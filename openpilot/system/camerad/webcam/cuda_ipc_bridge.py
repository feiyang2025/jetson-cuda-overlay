import ctypes, os, struct, mmap, socket, array, fcntl, time, errno, threading

SHM_DIR = '/dev/shm'

def _shm_path(stream_name):
    return f'{SHM_DIR}/cuda_ipc_{stream_name}'

def _sock_path(stream_name):
    return f'{SHM_DIR}/cuda_ipc_{stream_name}.sock'

class CudaIPCExport:
    """Server-side: allocate GPU buffer via VMM, export FD to importer via Unix socket."""

    def __init__(self, stream_name, size):
        self.stream_name = stream_name
        self.size = size
        self.device_ptr = None
        self.sock = None
        self.client_fd = -1
        self.cu = ctypes.CDLL('libcuda.so.1')
        self._ensure_cuda()
        self._setup_handles()

    def _ensure_cuda(self):
        try:
            from tinygrad.device import Device
            dev = Device['CUDA']
            ctx = ctypes.c_void_p(ctypes.addressof(dev.context.contents))
            self.cu.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
            self.cu.cuCtxSetCurrent.restype = ctypes.c_int
            self.cu.cuCtxSetCurrent(ctx)
        except Exception:
            pass

    def _setup_handles(self):
        import tinygrad.runtime.autogen.cuda as cuda_src
        cu = self.cu

        cu.cuMemGetAllocationGranularity.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_uint]
        cu.cuMemGetAllocationGranularity.restype = ctypes.c_int

        # Warm up: ensure CUDA context is fully active before VMM calls
        dev_id = ctypes.c_int()
        cu.cuCtxGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
        cu.cuCtxGetDevice.restype = ctypes.c_int
        cu.cuCtxGetDevice(ctypes.byref(dev_id))

        gran = ctypes.c_size_t()
        prop = cuda_src.CUmemAllocationProp()
        ctypes.memset(ctypes.addressof(prop), 0, ctypes.sizeof(prop))
        prop.type = cuda_src.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.requestedHandleTypes = cuda_src.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        prop.location.type = cuda_src.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = 0

        ret = cu.cuMemGetAllocationGranularity(ctypes.byref(gran), ctypes.byref(prop), 0)
        if ret != 0 or gran.value == 0:
            print(f"WARNING: cuMemGetAllocationGranularity returned {ret}, gran={gran.value}, using default 2MB")
            gran.value = 2097152

        padded = ((self.size + gran.value - 1) // gran.value) * gran.value

        va = ctypes.c_void_p()
        cu.cuMemAddressReserve(ctypes.byref(va), padded, 0, 0, 0)
        self._va = va
        self._padded = padded

        mem_handle = cuda_src.CUmemGenericAllocationHandle()
        cu.cuMemCreate(ctypes.byref(mem_handle), padded, ctypes.byref(prop), 0)
        cu.cuMemMap(va, padded, 0, mem_handle, 0)

        access = cuda_src.CUmemAccessDesc()
        ctypes.memset(ctypes.addressof(access), 0, ctypes.sizeof(access))
        access.location.type = cuda_src.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = 0
        access.flags = cuda_src.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        cu.cuMemSetAccess(va, padded, ctypes.byref(access), 1)

        self.device_ptr = va.value

        fd = ctypes.c_int()
        cu.cuMemExportToShareableHandle(
            ctypes.byref(fd), mem_handle,
            cuda_src.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0)
        cu.cuMemRelease(mem_handle)

        self._exported_fd = fd.value

        # Unix socket for FD passing
        sock_path = _sock_path(self.stream_name)
        try:
            os.unlink(sock_path)
        except FileNotFoundError:
            pass
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(sock_path)
        os.chmod(sock_path, 0o666)
        self.sock.listen(1)

        def _accept_thread():
            try:
                conn, _ = self.sock.accept()
                msg = [struct.pack('<Q', self.size)]
                fds = [array.array('i', [self._exported_fd])]
                conn.sendmsg(msg, [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds[0])])
                conn.close()
            except Exception:
                pass

        t = threading.Thread(target=_accept_thread, daemon=True)
        t.start()

        # Frame counter via mmap for cross-process visibility
        path = _shm_path(self.stream_name)
        with open(path, 'wb') as f:
            f.write(b'\x00' * 8)
        os.chmod(path, 0o666)
        f = os.open(path, os.O_RDWR)
        self._counter_mmap = mmap.mmap(f, 8, mmap.MAP_SHARED, mmap.PROT_WRITE)
        os.close(f)
        struct.pack_into('<Q', self._counter_mmap, 0, 0)

    def signal_frame(self, frame_id):
        struct.pack_into('<Q', self._counter_mmap, 0, frame_id)

    def close(self):
        if self.device_ptr:
            self._ensure_cuda()
            self.cu.cuMemAddressFree(self._va, self._padded)
            self.device_ptr = None
        if hasattr(self, '_counter_mmap'):
            self._counter_mmap.close()
        if self.sock:
            self.sock.close()
            try:
                os.unlink(_sock_path(self.stream_name))
            except FileNotFoundError:
                pass
        path = _shm_path(self.stream_name)
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


class CudaIPCImport:
    """Client-side: import GPU buffer via Unix socket FD transfer and VMM mapping."""

    def __init__(self, stream_name, size=None, timeout=10.0):
        self.stream_name = stream_name
        self.device_ptr = None
        self.size = size
        self._last_frame_id = 0
        self._shm_data = None
        self._cu = None
        self._va = None
        self._padded = 0
        self._setup(timeout)

    def _ensure_cuda(self):
        try:
            from tinygrad.device import Device
            dev = Device['CUDA']
            cu = ctypes.CDLL('libcuda.so.1')
            ctx = ctypes.c_void_p(ctypes.addressof(dev.context.contents))
            cu.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
            cu.cuCtxSetCurrent.restype = ctypes.c_int
            cu.cuCtxSetCurrent(ctx)
            return cu
        except Exception:
            return ctypes.CDLL('libcuda.so.1')

    def _setup(self, timeout=10.0):
        cu = self._ensure_cuda()
        self._cu = cu

        # Connect to Unix socket to receive FD
        sock_path = _sock_path(self.stream_name)
        deadline = time.time() + timeout
        fd = -1
        while time.time() < deadline:
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(2.0)
                sock.connect(sock_path)

                data = sock.recvmsg(8, socket.CMSG_SPACE(4))
                msg, ancdata, flags, addr = data
                for level, typ, raw in ancdata:
                    if level == socket.SOL_SOCKET and typ == socket.SCM_RIGHTS:
                        fds = array.array('i', raw[:4])
                        fd = fds[0]
                        if self.size is None:
                            self.size = struct.unpack('<Q', msg[:8])[0]
                sock.close()
                break
            except (FileNotFoundError, ConnectionRefusedError, socket.timeout, BlockingIOError, OSError) as e:
                time.sleep(0.3)
                continue

        if fd < 0:
            # Try reading from /dev/shm directly (legacy/timing fallback)
            raise RuntimeError('Could not connect to CUDA IPC server')

        # Import FD and map
        import tinygrad.runtime.autogen.cuda as cuda_src

        cu.cuMemGetAllocationGranularity.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_uint]
        cu.cuMemGetAllocationGranularity.restype = ctypes.c_int

        gran = ctypes.c_size_t()
        prop = cuda_src.CUmemAllocationProp()
        ctypes.memset(ctypes.addressof(prop), 0, ctypes.sizeof(prop))
        prop.type = cuda_src.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.requestedHandleTypes = cuda_src.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        prop.location.type = cuda_src.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = 0

        ret = cu.cuMemGetAllocationGranularity(
            ctypes.byref(gran), ctypes.byref(prop), 0)
        if ret != 0 or gran.value == 0:
            print(f"WARNING: cuMemGetAllocationGranularity returned {ret}, gran={gran.value}, using default 2MB")
            gran.value = 2097152
        self._padded = ((self.size + gran.value - 1) // gran.value) * gran.value

        # Import from FD
        import_handle = cuda_src.CUmemGenericAllocationHandle()
        cu.cuMemImportFromShareableHandle(
            ctypes.byref(import_handle), fd,
            cuda_src.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)

        # Reserve VA and map
        va = ctypes.c_void_p()
        cu.cuMemAddressReserve(ctypes.byref(va), self._padded, 0, 0, 0)
        cu.cuMemMap(va, self._padded, 0, import_handle, 0)

        access = cuda_src.CUmemAccessDesc()
        ctypes.memset(ctypes.addressof(access), 0, ctypes.sizeof(access))
        access.location.type = cuda_src.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = 0
        access.flags = cuda_src.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        cu.cuMemSetAccess(va, self._padded, ctypes.byref(access), 1)

        cu.cuMemRelease(import_handle)
        self._va = va
        self.device_ptr = va.value

        # Close the FD (not needed after import)
        os.close(fd)

        # mmap the shm file for fast counter polling
        path = _shm_path(self.stream_name)
        deadline2 = time.time() + 5.0
        while time.time() < deadline2:
            if os.path.exists(path):
                fd2 = os.open(path, os.O_RDONLY)
                self._shm_data = mmap.mmap(fd2, 8, mmap.MAP_SHARED, mmap.PROT_READ)
                os.close(fd2)
                break
            time.sleep(0.5)

    @property
    def frame_available(self):
        """Check if a new frame has been signaled. Returns frame_id or None."""
        if self._shm_data is None:
            return None
        val = struct.unpack('<Q', self._shm_data[:8])[0]
        if val != self._last_frame_id:
            self._last_frame_id = val
            return val
        return None

    def close(self):
        if self.device_ptr and self._cu:
            self._cu.cuMemAddressFree(self._va, self._padded)
            self.device_ptr = None
        if self._shm_data:
            self._shm_data.close()
            self._shm_data = None
