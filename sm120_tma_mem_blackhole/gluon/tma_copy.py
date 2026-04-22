"""
Gluon TMA memory-black-hole reproducer for RTX 5090 (SM120).

Probes whether a Gluon TMA kernel pulls in the ~3.7 GiB CUDA-driver syscall
buffer by watching NVML device memory before and after launch.

Usage:
  python sm120_tma_mem_blackhole/gluon/tma_copy.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier

from common.benchmark import run_blackhole_test

TILE_M = 128
TILE_N = 128


# --------------------------------------------------------------------------- #
# Gluon TMA copy kernel: async_copy_global_to_shared -> async_copy_shared_to_global.
# --------------------------------------------------------------------------- #
@gluon.jit
def tma_copy_kernel(in_desc, out_desc, TILE_M: gl.constexpr, TILE_N: gl.constexpr):
    smem_layout: gl.constexpr = in_desc.layout
    smem = gl.allocate_shared_memory(in_desc.dtype, [TILE_M, TILE_N], smem_layout)

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)

    mbarrier.expect(bar, in_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(in_desc, [0, 0], bar, smem)

    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    tma.async_copy_shared_to_global(out_desc, [0, 0], smem)
    tma.store_wait(pendings=0)


def main():
    def runner(a, b):
        block_shape = [TILE_M, TILE_N]
        layout = gl.NVMMASharedLayout.get_default_for(block_shape, gl.float16)
        in_desc = TensorDescriptor.from_tensor(a, block_shape, layout)
        out_desc = TensorDescriptor.from_tensor(b, block_shape, layout)
        grid = (1, 1)
        tma_copy_kernel[grid](in_desc, out_desc, TILE_M, TILE_N, num_warps=1)

    run_blackhole_test(runner, "Gluon")


if __name__ == "__main__":
    main()
