#!/usr/bin/env python3
"""
Verify the TMA black hole with raw CUDA driver APIs (via cuda_bind).

This is a variant of triton/experiment_blackhole_at_launch.py that uses the
raw driver wrappers in cuda_bind instead of Triton's utils.load_binary.
It proves the same two facts:

  1. The ~3.7 GiB allocation happens at FIRST LAUNCH, not at module load.
  2. cuModuleUnload does NOT release it — the buffer is context-bound.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# tma_copy lives under the triton/ sub-package
TRITON_DIR = ROOT / "triton"
if str(TRITON_DIR) not in sys.path:
    sys.path.insert(0, str(TRITON_DIR))

import torch
import triton
from triton.backends.nvidia.driver import CudaLauncher

from cuda_bind import (
    cu_mem_get_info,
    cu_module_get_function,
    cu_module_load,
    cu_module_unload,
)

# ---------------------------------------------------------------------------
# Triton needs an allocator for global_scratch allocations.
# ---------------------------------------------------------------------------
triton.set_allocator(
    lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8)
)


def mem_mb() -> float:
    """Global GPU memory currently used (MiB)."""
    free, total = cu_mem_get_info()
    return (total - free) / 1024 / 1024


def snapshot(tag: str) -> None:
    torch.cuda.synchronize()
    print(f"  [{tag:50s}] {mem_mb():8.1f} MiB")


def main() -> None:
    torch.cuda.init()

    import tma_copy
    tma_cluster_kernel = tma_copy.tma_cluster_kernel

    print("=" * 72)
    print("Experiment: raw CUDA driver API load + Triton launch")
    print("=" * 72)

    # ---- Phase A: compile only (no module load) ----------------------------
    c = torch.randn((128, 128), device="cuda", dtype=torch.float16)
    d = torch.empty_like(c)

    compiled = tma_cluster_kernel.warmup(
        c, d, 128, 128, TILE_M=128, TILE_N=128, SPIN_NS=0, grid=(1,)
    )
    snapshot("A. after warmup (compile only, module=None)")

    # ---- Phase B: raw driver API load ---------------------------------------
    tma_mod = cu_module_load(compiled.asm["cubin"])
    tma_func = cu_module_get_function(tma_mod, "tma_cluster_kernel")
    snapshot("B. after cuModuleLoadData + cuModuleGetFunction")

    # ---- Phase C: inject handles so Triton skips load_binary ----------------
    compiled.module = tma_mod
    compiled.function = tma_func
    compiled._run = CudaLauncher(compiled.src, compiled.metadata)

    # ---- Phase D: launch (Triton goes straight to cuLaunchKernelEx) --------
    tma_cluster_kernel[(1, 1)](c, d, 128, 128, TILE_M=128, TILE_N=128, SPIN_NS=0)
    snapshot("C. after cuLaunchKernelEx (launch returned)")

    torch.cuda.synchronize()
    snapshot("D. after kernel completes (sync)")

    # ---- Phase E: raw driver API unload -------------------------------------
    # Even cuModuleUnload (raw CUDA driver API) does NOT release the ~3.7 GiB
    # black-hole buffer.  It is context-bound, not module-bound.
    cu_module_unload(tma_mod)
    snapshot("E. after cuModuleUnload")

    torch.cuda.empty_cache()
    snapshot("F. after torch.cuda.empty_cache")

    del c, d
    torch.cuda.empty_cache()
    snapshot("G. after del tensors + empty_cache")

    print()
    print("=" * 72)
    print("Done.  See source comments for interpretation.")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Interpretation (for readers of the source code):
#
#   Step                          Expected delta
#   -----------------------------------------------------------
#   A. compile only               +2   MiB  (Triton host bookkeeping)
#   B. cuModuleLoadData           +80  MiB  (cubin code segment in device mem)
#   C. cuLaunchKernelEx           +3728 MiB  <-- THE BLACK HOLE
#   D. after sync                 0    MiB  (kernel completes normally)
#   E. cuModuleUnload             0    MiB  (buffer is context-bound)
#   F. torch.cuda.empty_cache     0    MiB
#   G. del tensors + empty_cache  0    MiB
#
# Key difference from the Triton-driver version:
#   This script calls the RAW CUDA driver API cuModuleUnload.
#   Even that does NOT release the black-hole buffer — proving the allocation
#   is truly context-bound, not merely hidden behind a missing Triton API.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
