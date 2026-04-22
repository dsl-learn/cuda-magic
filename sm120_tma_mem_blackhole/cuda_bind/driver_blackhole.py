#!/usr/bin/env python3
"""
Verify the TMA black hole with raw CUDA driver APIs (via cuda_bind).

Same methodology as triton/blackhole_at_launch.py, but uses raw
nvidia-cuda-python bindings instead of Triton's driver utils.

Usage:
  python sm120_tma_mem_blackhole/cuda_bind/driver_blackhole.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRITON_DIR = ROOT / "triton"
if str(TRITON_DIR) not in sys.path:
    sys.path.insert(0, str(TRITON_DIR))

import torch
import triton
from triton.backends.nvidia.driver import CudaLauncher

from cuda_bind import cu_module_load, cu_module_get_function, cu_module_unload
from common.mem_utils import nvml_init, nvml_used

# Triton needs an allocator for global_scratch allocations.
triton.set_allocator(
    lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8)
)

_nvml_handle = nvml_init(torch.cuda.current_device())


def snapshot(tag: str) -> None:
    torch.cuda.synchronize()
    used_mib = nvml_used(_nvml_handle) / (1024 * 1024)
    print(f"  [{tag:<50s}] {used_mib:>8.1f} MiB")


def main() -> None:
    torch.cuda.init()

    import tma_copy
    kernel = tma_copy.tma_cluster_kernel

    c = torch.randn((128, 128), device="cuda", dtype=torch.float16)
    d = torch.empty_like(c)

    print("=" * 72)
    print("Raw CUDA driver API: cuModuleLoad -> launch -> cuModuleUnload")
    print("=" * 72)

    # A. Compile only
    compiled = kernel.warmup(c, d, 128, 128, TILE_M=128, TILE_N=128, grid=(1,))
    snapshot("A. after warmup (compile only, module=None)")

    # B. Raw driver load
    tma_mod = cu_module_load(compiled.asm["cubin"])
    tma_func = cu_module_get_function(tma_mod, compiled.metadata.name)
    snapshot("B. after cuModuleLoadData + cuModuleGetFunction")

    # C. Inject handles
    compiled.module = tma_mod
    compiled.function = tma_func
    compiled._run = CudaLauncher(compiled.src, compiled.metadata)

    # D. Launch
    kernel[(1, 1)](c, d, 128, 128, TILE_M=128, TILE_N=128)
    snapshot("C. after cuLaunchKernelEx (launch returned)")

    torch.cuda.synchronize()
    snapshot("D. after kernel completes (sync)")

    # E. Raw driver unload (proves buffer is context-bound)
    cu_module_unload(tma_mod)
    snapshot("E. after cuModuleUnload")

    torch.cuda.empty_cache()
    snapshot("F. after torch.cuda.empty_cache")

    del c, d
    torch.cuda.empty_cache()
    snapshot("G. after del tensors + empty_cache")

    print()
    print("Key finding: even cuModuleUnload does NOT release the buffer.")
    print("The allocation is context-bound, not module-bound.")


if __name__ == "__main__":
    main()
