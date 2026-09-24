import threading
import queue
import os
import av
import numpy as np
import ctypes

# Try to import nvJPEG decoder
try:
    from cuda_jpeg_decoder import CudaJpegDecoder
    _has_nvjpeg_decoder = True
except ImportError:
    _has_nvjpeg_decoder = False


def _rgb_to_nv12_gpu(rgb, h, w):
    import torch
    r = rgb[0].float()
    g = rgb[1].float()
    b = rgb[2].float()
    y = (0.299 * r + 0.587 * g + 0.114 * b).clamp(0, 255).byte().reshape(-1)
    r_pool = torch.nn.functional.avg_pool2d(r[None, None, :, :], 2)[0, 0]
    g_pool = torch.nn.functional.avg_pool2d(g[None, None, :, :], 2)[0, 0]
    b_pool = torch.nn.functional.avg_pool2d(b[None, None, :, :], 2)[0, 0]
    u = (-0.169 * r_pool - 0.331 * g_pool + 0.5 * b_pool + 128).clamp(0, 255).byte().reshape(-1)
    v = (0.5 * r_pool - 0.419 * g_pool - 0.081 * b_pool + 128).clamp(0, 255).byte().reshape(-1)
    uv = torch.stack([u, v], dim=1).reshape(-1)
    return torch.cat([y, uv])


class CudaCamera:
    def __init__(self, cam_type_state, stream_type, camera_id, gpu_output=None):
        self.cam_type_state = cam_type_state
        self.stream_type = stream_type
        self.cur_frame_id = 0
        self._gpu_output = gpu_output  # (device_ptr, size) for zero-copy
        try:
            import torch
            self._cuda_available = torch.cuda.is_available()
        except ImportError:
            self._cuda_available = False

        opts = {
            'input_format': 'mjpeg',
            'video_size': '1920x1080',
            'framerate': '20',
        }
        self.container = av.open(camera_id, 'r', format='video4linux2', options=opts)
        assert self.container.streams.video, f"Can't open video stream for camera {camera_id}"
        self.video_stream = self.container.streams.video[0]
        self.W = self.video_stream.codec_context.width
        self.H = self.video_stream.codec_context.height

    def read_frames(self):
        if not self._cuda_available:
            yield from self._read_frames_cpu()
        else:
            yield from self._read_frames_cuda()

    def _read_frames_cpu(self):
        for packet in self.container.demux(self.video_stream):
            for frame in packet.decode():
                yuv = frame.reformat(format='nv12').to_ndarray()
                yield yuv.data.tobytes()

    def _read_frames_cuda(self):
        # Initialize nvJPEG decoder for this camera
        dec = None
        if _has_nvjpeg_decoder:
            try:
                dec = CudaJpegDecoder()
                if not dec.create(self.W, self.H):
                    dec = None
            except Exception:
                dec = None

        if dec is not None:
            # If external GPU output provided, redirect decoder output
            if self._gpu_output:
                dec.set_external_output(self._gpu_output[0], self._gpu_output[1])

            for packet in self.container.demux(self.video_stream):
                raw_jpeg = bytes(packet)
                if not raw_jpeg or packet.size == 0:
                    continue
                dev_ptr = dec.decode_to_device(raw_jpeg)
                if dev_ptr:
                    # Copy from GPU to CPU bytes
                    yield dec.copy_to_host(dev_ptr)
                else:
                    yield b'\x00' * (self.W * self.H * 3 // 2)
            dec.close()
        else:
            # Fallback: CPU decode
            import torch
            from torchvision.io import decode_jpeg, ImageReadMode
            for packet in self.container.demux(self.video_stream):
                raw_jpeg = bytes(packet)
                if not raw_jpeg or packet.size == 0:
                    continue
                jpeg_tensor = torch.frombuffer(bytearray(raw_jpeg), dtype=torch.uint8)
                try:
                    rgb = decode_jpeg(jpeg_tensor, mode=ImageReadMode.RGB, device='cuda')
                except RuntimeError:
                    yield from self._read_frames_cpu()
                    return
                nv12 = _rgb_to_nv12_gpu(rgb, self.H, self.W)
                yield nv12.cpu().numpy().tobytes()


class CudaCameraMJPG:
    def __init__(self, cam_type_state, stream_type, camera_id, num_workers=None, max_queue_size=10, use_processes=False, direct_mode=False):
        import cv2
        try:
            camera_id = int(camera_id)
        except ValueError:
            pass

        self.cam_type_state = cam_type_state
        self.stream_type = stream_type
        self.cur_frame_id = 0
        try:
            import torch
            self._cuda_available = torch.cuda.is_available()
        except ImportError:
            self._cuda_available = False

        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise OSError(f"Unable to open camera device {camera_id}")

        self._configure_camera_format("MJPG")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_format = self._get_current_format()

        if direct_mode:
            self.stop_event = threading.Event()
            return

        if num_workers is None:
            num_workers = os.cpu_count() or 4
        self.num_workers = num_workers
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.output_queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()

        self.read_thread = threading.Thread(target=self._frame_reader, daemon=True)
        self.use_processes = use_processes
        self.read_thread.start()

    def _configure_camera_format(self, target_fourcc):
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*target_fourcc)
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 20)

    def _get_current_format(self):
        import cv2
        fourcc_code = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        return ''.join([chr((fourcc_code >> 8 * i) & 0xFF) for i in range(4)])

    def capture_direct(self):
        import cv2
        ret, frame = self.cap.read()
        if not ret:
            return None
        return self._bgr_to_nv12_gpu(frame) if self._cuda_available else self._bgr_to_nv12_cpu(frame)

    def _bgr_to_nv12_cpu(self, bgr_frame):
        import av as _av
        frame = _av.VideoFrame.from_ndarray(bgr_frame, format='bgr24')
        return frame.reformat(format='nv12').to_ndarray().data.tobytes()

    def _bgr_to_nv12_gpu(self, bgr_frame):
        import torch
        tensor = torch.from_numpy(bgr_frame).cuda().permute(2, 0, 1)
        rgb = tensor[[2, 1, 0]]
        nv12 = _rgb_to_nv12_gpu(rgb, self.H, self.W)
        return nv12.cpu().numpy().tobytes()

    def _frame_reader(self):
        import cv2
        while not self.stop_event.is_set():
            if self.frame_queue.full():
                self.stop_event.wait(0.01)
                continue
            ret, frame = self.cap.read()
            if not ret:
                self.stop_event.set()
                break
            try:
                self.frame_queue.put(frame, timeout=0.1)
            except queue.Full:
                continue

    def _bgr_to_nv12_worker(self, bgr_frame):
        if self._cuda_available:
            return self._bgr_to_nv12_gpu(bgr_frame)
        else:
            return self._bgr_to_nv12_cpu(bgr_frame)

    def read_frames(self):
        workers = []
        if self.use_processes:
            t = threading.Thread(target=self._frame_worker_process_pool)
            t.start()
            workers.append(t)
        else:
            for _ in range(self.num_workers):
                t = threading.Thread(target=self._frame_worker)
                t.start()
                workers.append(t)

        try:
            while not self.stop_event.is_set() or not self.output_queue.empty():
                try:
                    yuv_bytes = self.output_queue.get(timeout=0.5)
                    yield yuv_bytes
                    self.output_queue.task_done()
                except queue.Empty:
                    if self.stop_event.is_set():
                        break
        finally:
            self.stop_event.set()
            self.read_thread.join()
            for t in workers:
                t.join()
            self.cap.release()

    def _frame_worker(self):
        while not self.stop_event.is_set() or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            yuv_bytes = self._bgr_to_nv12_worker(frame)
            try:
                self.output_queue.put(yuv_bytes, timeout=0.1)
            except queue.Full:
                try:
                    _ = self.output_queue.get_nowait()
                    self.output_queue.put(yuv_bytes)
                except queue.Empty:
                    pass
            finally:
                self.frame_queue.task_done()

    def _frame_worker_process_pool(self):
        from concurrent.futures import ProcessPoolExecutor
        futures = set()
        executor = ProcessPoolExecutor(max_workers=self.num_workers)
        try:
            while not self.stop_event.is_set() or not self.frame_queue.empty() or futures:
                try:
                    while len(futures) < self.num_workers and not self.frame_queue.empty():
                        frame = self.frame_queue.get_nowait()
                        fut = executor.submit(self._bgr_to_nv12_cpu, frame)
                        futures.add(fut)
                        self.frame_queue.task_done()
                except queue.Empty:
                    pass

                done = [f for f in futures if f.done()]
                for f in done:
                    futures.remove(f)
                    try:
                        yuv_bytes = f.result()
                        try:
                            self.output_queue.put(yuv_bytes, timeout=0.1)
                        except queue.Full:
                            try:
                                _ = self.output_queue.get_nowait()
                                self.output_queue.put(yuv_bytes)
                            except queue.Empty:
                                pass
                    except Exception:
                        self.stop_event.set()
                        break
        finally:
            executor.shutdown(wait=True)

    def __del__(self):
        self.stop_event.set()
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
