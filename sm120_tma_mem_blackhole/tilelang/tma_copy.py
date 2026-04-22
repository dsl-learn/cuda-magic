"""
Minimal TileLang TMA copy + black-hole probe for RTX 5090 (SM120).

Usage:
  python sm120_tma_mem_blackhole/tilelang/tma_copy_simple.py
"""

import os
import sys
from pathlib import Path

# Add repo root so we can import common/
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Cache must be set before importing tilelang
CACHE_ROOT = Path(__file__).parent / "tilelang_cache"
CACHE_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("TILELANG_CACHE_DIR", str(CACHE_ROOT))

import torch
import tilelang
import tilelang.language as T

from common.mem_utils import nvml_init, nvml_used, fmt_gb


# --------------------------------------------------------------------------- #
# Kernel: TMA copy via shared memory staging
# --------------------------------------------------------------------------- #
@tilelang.jit(out_idx=[-1])
def tma_copy(M, N, block_M, block_N, dtype):
    @T.prim_func
    def kern(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            shared = T.alloc_shared((block_M, block_N), dtype)
            T.copy(A[by * block_M, bx * block_N], shared)
            T.copy(shared, B[by * block_M, bx * block_N])
    return kern


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    TILE_M = 128
    TILE_N = 128
    M, N = 1024, 2048
    DTYPE = T.bfloat16

    # Warm-up CUDA context so baseline is stable
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    handle = nvml_init(torch.cuda.current_device())

    before = nvml_used(handle)
    print(f"[baseline] {fmt_gb(before)}")

    a = torch.randn((M, N), device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()

    after_alloc = nvml_used(handle)
    print(f"[after alloc input] {fmt_gb(after_alloc)}  (d={fmt_gb(after_alloc - before)})")

    print("\n=== launching TileLang TMA copy kernel ===")
    k = tma_copy(M, N, TILE_M, TILE_N, DTYPE)
    b = k(a)  # out_idx=[-1] -> output returned
    torch.cuda.synchronize()

    after_launch = nvml_used(handle)
    print(f"[after launch] {fmt_gb(after_launch)}  (d vs baseline={fmt_gb(after_launch - before)})")
    print(f"\n[BLACK HOLE] launch - inputs = {fmt_gb(after_launch - after_alloc)}")

    ok = torch.equal(a, b)
    print(f"{'OK' if ok else 'MISMATCH'}: identity copy")
