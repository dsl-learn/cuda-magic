"""Minimal raw CUDA driver API wrappers via nvidia-cuda-python (cuda.bindings).

This module exists because Triton does not expose cuModuleUnload.
When you need to call raw driver APIs (e.g. to verify that module-unload
still does NOT release a context-bound buffer), import from here instead
of scattering cuda.bindings boilerplate across the codebase.
"""

import base64
import ctypes
from cuda.bindings import driver as _drv

# Ensure the driver is initialized (idempotent).
_drv.cuInit(0)


def _maybe_decode(cubin) -> bytes:
    if isinstance(cubin, str):
        return base64.b64decode(cubin)
    if isinstance(cubin, bytes):
        return cubin
    raise TypeError(f"cubin must be bytes or str, got {type(cubin).__name__}")


def cu_module_load(cubin) -> int:
    """Load a cubin and return the module handle (int)."""
    data = _maybe_decode(cubin)
    status, mod = _drv.cuModuleLoadData(data)
    if status != _drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuModuleLoadData failed: {status}")
    return int(mod.value) if hasattr(mod, "value") else int(mod)


def cu_module_get_function(module: int, name: str | bytes) -> int:
    """Get a function handle from a loaded module."""
    if isinstance(name, str):
        name = name.encode()
    status, fn = _drv.cuModuleGetFunction(module, name)
    if status != _drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuModuleGetFunction({name!r}) failed: {status}")
    return int(fn.value) if hasattr(fn, "value") else int(fn)


def cu_module_unload(module: int) -> None:
    """Unload a module.  Does NOT release context-bound allocations."""
    status, = _drv.cuModuleUnload(module)
    if status != _drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuModuleUnload failed: {status}")


def cu_mem_get_info() -> tuple[int, int]:
    """Return (free_bytes, total_bytes) for the current device."""
    status, free, total = _drv.cuMemGetInfo()
    if status != _drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuMemGetInfo failed: {status}")
    return int(free), int(total)
