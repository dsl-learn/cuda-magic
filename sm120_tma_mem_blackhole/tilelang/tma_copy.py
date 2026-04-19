import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CACHE_ROOT = Path(__file__).parent / "tilelang_cache"
CACHE_ROOT.mkdir(exist_ok=True)
os.environ.setdefault('TILELANG_CACHE_DIR', str(CACHE_ROOT))

import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def mean_center(M, N, block_M, block_N, dtype):
    @T.prim_func
    def kern(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), dtype)
            B_shared = T.alloc_shared((block_M, block_N), dtype)
            T.copy(A[by * block_M, bx * block_N], A_shared)
            for ly, lx in T.Parallel(block_M, block_N):
                B_shared[ly, lx] = A_shared[ly, lx]
            T.copy(B_shared, B[by * block_M, bx * block_N])
    return kern


k = mean_center(1024, 2048, 128, 128, T.bfloat16)
src = k.get_kernel_source()
print(src)
print('---')
print('has cp.async.bulk:', 'cp.async.bulk' in src)
print('has tma_load:', 'tma_load' in src.lower())
