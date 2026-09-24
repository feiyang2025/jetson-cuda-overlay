"""
TensorRT model runner for openpilot modeld — ctypes-only implementation.

Uses trt_c_api.so (compiled C wrapper around TensorRT C++ API) instead of
the 'tensorrt' Python package.  No Python-version dependency, so it works
from any virtualenv.

Shares the same CUDA context as tinygrad (no context isolation).
All GPU memory and streams live in tinygrad's CUDA context.
Accepts host numpy arrays OR raw CUDA device pointers for zero-copy input.
"""
import ctypes
import numpy as np
import sys
from pathlib import Path

# ── Path to the compiled C API shared library ──────────────────────────
_TRT_C_API_SO = Path(__file__).resolve().parent / "trt_c_api.so"

# ── Load C API ─────────────────────────────────────────────────────────
_lib = ctypes.CDLL(str(_TRT_C_API_SO))

# ── C function signatures ──────────────────────────────────────────────
_lib.trt_engine_create.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_lib.trt_engine_create.restype = ctypes.c_void_p

_lib.trt_engine_execute.argtypes = [
    ctypes.c_void_p,                     # handle
    ctypes.POINTER(ctypes.c_ulonglong),  # input_addrs[]
    ctypes.c_ulonglong,                  # output_addr
    ctypes.c_ulonglong,                  # stream (CUstream handle)
]
_lib.trt_engine_execute.restype = ctypes.c_int

_lib.trt_engine_destroy.argtypes = [ctypes.c_void_p]
_lib.trt_engine_destroy.restype = None

_lib.trt_engine_num_inputs.argtypes = [ctypes.c_void_p]
_lib.trt_engine_num_inputs.restype = ctypes.c_int

_lib.trt_engine_num_outputs.argtypes = [ctypes.c_void_p]
_lib.trt_engine_num_outputs.restype = ctypes.c_int

_lib.trt_engine_input_name.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.trt_engine_input_name.restype = ctypes.c_char_p

_lib.trt_engine_output_name.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.trt_engine_output_name.restype = ctypes.c_char_p

_lib.trt_engine_input_dims.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),    # ndim_out
    ctypes.POINTER(ctypes.c_int64),  # dims_out[8]
    ctypes.POINTER(ctypes.c_int),    # dtype_out
]
_lib.trt_engine_input_dims.restype = ctypes.c_int

_lib.trt_engine_output_shape.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int64),  # dims_out[8]
]
_lib.trt_engine_output_shape.restype = ctypes.c_int

_lib.trt_engine_output_dtype.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.trt_engine_output_dtype.restype = ctypes.c_int


# ─── TRT dtype → numpy dtype mapping ──────────────────────
_TRT_DTYPE_TO_NP = {
    0: np.float32,   # kFLOAT
    1: np.float16,   # kHALF
    2: np.int8,      # kINT8
    3: np.int32,     # kINT32
    4: np.bool_,     # kBOOL
    5: np.uint8,     # kUINT8
    7: np.float16,   # kBF16 (stored as float16 in numpy)
}


class _CUDA:
    """Minimal ctypes wrapper around the CUDA driver API (libcuda.so)."""

    def __init__(self):
        self.lib = ctypes.CDLL("libcuda.so.1")

        for fn, argt, rest in [
            ("cuInit",
             [ctypes.c_uint],
             ctypes.c_int),
            ("cuDeviceGet",
             [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
             ctypes.c_int),
            ("cuDevicePrimaryCtxRetain",
             [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_int],
             ctypes.c_int),
            ("cuCtxGetCurrent",
             [ctypes.POINTER(ctypes.c_ulonglong)],
             ctypes.c_int),
            ("cuCtxPushCurrent",
             [ctypes.c_ulonglong],
             ctypes.c_int),
            ("cuCtxPopCurrent",
             [ctypes.POINTER(ctypes.c_ulonglong)],
             ctypes.c_int),
            ("cuCtxCreate_v2",
             [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_uint, ctypes.c_int],
             ctypes.c_int),
            ("cuDevicePrimaryCtxRetain",
             [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_int],
             ctypes.c_int),
            ("cuDevicePrimaryCtxRelease",
             [ctypes.c_int],
             ctypes.c_int),
            ("cuCtxSetCurrent",
             [ctypes.c_ulonglong],
             ctypes.c_int),
            ("cuCtxGetDevice",
             [ctypes.POINTER(ctypes.c_int)],
             ctypes.c_int),
            ("cuMemAlloc_v2",
             [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_size_t],
             ctypes.c_int),
            ("cuMemFree_v2",
             [ctypes.c_ulonglong],
             ctypes.c_int),
            ("cuMemcpyHtoDAsync_v2",
             [ctypes.c_ulonglong, ctypes.c_void_p, ctypes.c_size_t,
              ctypes.c_ulonglong],
             ctypes.c_int),
            ("cuMemcpyDtoHAsync_v2",
             [ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_size_t,
              ctypes.c_ulonglong],
             ctypes.c_int),
            ("cuStreamCreate",
             [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_uint],
             ctypes.c_int),
            ("cuStreamDestroy_v2",
             [ctypes.c_ulonglong],
             ctypes.c_int),
            ("cuStreamSynchronize",
             [ctypes.c_ulonglong],
             ctypes.c_int),
        ]:
            f = getattr(self.lib, fn)
            f.argtypes = argt
            f.restype = rest
            setattr(self, fn, f)


def _check(ret: int):
    if ret != 0:
        raise RuntimeError(f"CUDA driver error {ret}")


# ═══════════════════════════════════════════════════════════════════════
# Public class
# ═══════════════════════════════════════════════════════════════════════

class TensorRTModel:
    """TensorRT inference using ctypes-only C API (no Python tensorrt pkg).

    Loads the engine in the currently-active CUDA context (tinygrad's) and
    stays there — no context switching per inference call.
    """

    def __init__(self, engine_path: str):
        sys.stderr.write("[TRT_CTYPES] TensorRTModel init\n")
        self.cu = _CUDA()

        # ── Ensure CUDA driver is initialized ──
        _check(self.cu.cuInit(0))

        # ── 绑定 PRIMARY context ──
        # 根因(0902晚): modeld 进程 tinygrad/transform/TRT 三方混用, TRT 跟随 current context
        # 或私建 context 都导致引擎读到全零输入(唯一能跑的配置=primary, 独立测试与FOF均如此)。
        # 统一: 全部资源(retain primary)+调用时 push/pop, 与可工作的独立配置完全一致。
        ctx = ctypes.c_ulonglong()
        ret = self.cu.cuCtxGetCurrent(ctypes.byref(ctx))
        want_primary = True
        if ret == 0 and ctx.value != 0:
            # 查 current 是否已是 primary
            pri = ctypes.c_ulonglong()
            _check(self.cu.cuDevicePrimaryCtxRetain(ctypes.byref(pri), 0))
            self.cu.cuDevicePrimaryCtxRelease(0)
            want_primary = (pri.value != ctx.value)
            if not want_primary:
                ctx = pri
        if want_primary:
            _check(self.cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), 0))
            _check(self.cu.cuCtxSetCurrent(ctx))
        sys.stderr.write(f"[TRT_CTYPES] CUDA ctx=0x{ctx.value:x} primary={want_primary}\n")
        self._cuda_ctx = ctx.value  # save for context push before each inference

        # ── Read serialised engine ──
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        engine_buf = (ctypes.c_char * len(engine_data)).from_buffer_copy(engine_data)

        # ── Create TRT engine ──
        self._handle = _lib.trt_engine_create(engine_buf, len(engine_data))
        if not self._handle:
            raise RuntimeError(f"trt_engine_create failed for {engine_path}")

        # ── Query I/O metadata ──
        self.num_inputs = _lib.trt_engine_num_inputs(self._handle)
        self.num_outputs = _lib.trt_engine_num_outputs(self._handle)

        self.input_names: list[str] = []
        self.input_shapes: list[tuple] = []
        self.input_dtypes: dict[str, np.dtype] = {}  # engine-expected dtype per input
        for i in range(self.num_inputs):
            name = _lib.trt_engine_input_name(self._handle, i).decode()
            self.input_names.append(name)
            ndim = ctypes.c_int()
            dims_arr = (ctypes.c_int64 * 8)()
            dt = ctypes.c_int()
            _lib.trt_engine_input_dims(
                self._handle, i,
                ctypes.byref(ndim), dims_arr, ctypes.byref(dt),
            )
            self.input_shapes.append(tuple(dims_arr[: ndim.value]))
            self.input_dtypes[name] = _TRT_DTYPE_TO_NP.get(dt.value, np.float32)

        self.output_names: list[str] = []
        for i in range(self.num_outputs):
            name = _lib.trt_engine_output_name(self._handle, i).decode()
            self.output_names.append(name)

        # ── Output metadata (we only use output[0]) ──
        dims_out = (ctypes.c_int64 * 8)()
        nd = _lib.trt_engine_output_shape(self._handle, 0, dims_out)
        self.output_shape = tuple(dims_out[:nd])
        dtype_val = _lib.trt_engine_output_dtype(self._handle, 0)
        self.output_dtype_np = _TRT_DTYPE_TO_NP.get(dtype_val, np.float32)
        self.output_nbytes = int(np.prod(self.output_shape)) * np.dtype(self.output_dtype_np).itemsize

        # ── Create CUDA stream ──
        self.stream = ctypes.c_ulonglong()
        _check(self.cu.cuStreamCreate(ctypes.byref(self.stream), 0))

        # ── Allocate GPU output buffer ──
        self.gpu_output = ctypes.c_ulonglong()
        _check(self.cu.cuMemAlloc_v2(ctypes.byref(self.gpu_output), self.output_nbytes))

        # ── Pre-allocated GPU input buffers (used only for numpy fallback) ──
        self.input_gpu: dict[str, dict] = {}

    # ────────────────────────────────────────────────────────────────────

    def __call__(self, **kwargs) -> np.ndarray:
        """Run inference.

        Kwargs values can be:
          - ``int`` / ``numpy.integer``  →  raw CUDA device pointer (zero-copy)
          - ``numpy.ndarray``            →  host memory (H2D copy)

        Returns a flat float32 numpy array.
        """
        cu = self.cu

        # ── 进入 TRT 专属 context(所有绑定显存/拷贝/执行都在本 context 内) ──
        pushed = False
        if self._cuda_ctx:
            ctx_now = ctypes.c_ulonglong()
            cu.cuCtxGetCurrent(ctypes.byref(ctx_now))
            if ctx_now.value != self._cuda_ctx:
                _check(cu.cuCtxPushCurrent(self._cuda_ctx))
                pushed = True
            else:
                pushed = False

        # ── Build input address array for the C API ─────────────────
        input_addrs = (ctypes.c_ulonglong * self.num_inputs)()
        for i, name in enumerate(self.input_names):
            val = kwargs.get(name)
            if val is None:
                raise KeyError(f"Missing input tensor: {name}")

            if isinstance(val, (int, np.integer)):
                # Zero-copy: the caller already owns a CUDA buffer
                input_addrs[i] = int(val)

            elif isinstance(val, np.ndarray):
                # Convert host dtype to engine-expected dtype (e.g. fp32 → fp16).
                # Copying raw fp32 bytes into an fp16 binding reads garbage
                # (bit-pattern reinterpretation → feature amplification/NaN).
                if val.dtype != self.input_dtypes[name]:
                    val = np.ascontiguousarray(val, dtype=self.input_dtypes[name])
                # H2D copy — lazy-allocate a GPU buffer for this input
                info = self.input_gpu.get(name)
                if info is None:
                    nbytes = int(np.prod(val.shape)) * val.dtype.itemsize
                    ptr = ctypes.c_ulonglong()
                    _check(cu.cuMemAlloc_v2(ctypes.byref(ptr), nbytes))
                    info = {"ptr": ptr.value, "nbytes": nbytes}
                    self.input_gpu[name] = info
                _check(cu.cuMemcpyHtoDAsync_v2(
                    info["ptr"],
                    ctypes.c_void_p(val.ctypes.data),
                    info["nbytes"],
                    self.stream,
                ))
                input_addrs[i] = info["ptr"]

            else:
                raise TypeError(
                    f"Unsupported input type for {name}: {type(val).__name__}"
                )

        # ── Execute TRT inference (C API) ──────────────────────────
        ret = _lib.trt_engine_execute(
            self._handle,
            input_addrs,
            self.gpu_output.value,  # output_addr (int)
            self.stream.value,      # stream (int)
        )
        if ret != 0:
            raise RuntimeError("trt_engine_execute returned non-zero")

        # ── D2H: copy output back to CPU ───────────────────────────
        cpu_out = np.empty(self.output_shape, dtype=self.output_dtype_np)
        _check(cu.cuMemcpyDtoHAsync_v2(
            ctypes.c_void_p(cpu_out.ctypes.data),
            self.gpu_output,
            self.output_nbytes,
            self.stream,
        ))
        _check(cu.cuStreamSynchronize(self.stream))

        if self._cuda_ctx and pushed:
            popped = ctypes.c_ulonglong()
            _check(cu.cuCtxPopCurrent(ctypes.byref(popped)))

        return cpu_out.astype(np.float32).ravel()

    # ────────────────────────────────────────────────────────────────────

    def __del__(self):
        # 构造失败时部分属性不存在，全部用 getattr 守卫，避免 in-exception double fault
        exc = None
        if hasattr(self, "cu") and hasattr(self, "input_gpu"):
            for fn, args in [
                (getattr(_lib, "trt_engine_destroy", None), (getattr(self, "_handle", None),)),
                (getattr(self.cu, "cuStreamDestroy_v2", None), (getattr(self, "stream", None),)),
                (getattr(self.cu, "cuMemFree_v2", None), (getattr(self, "gpu_output", None),)),
            ]:
                if fn is None or args[0] is None:
                    continue
                try:
                    fn(*args)
                except Exception as e:
                    exc = e
            for info in self.input_gpu.values():
                try:
                    self.cu.cuMemFree_v2(info["ptr"])
                except Exception as e:
                    exc = e
        if exc is not None:
            sys.stderr.write(f"[TRT_CTYPES] __del__ error: {exc}\n")
