"""
TMA memory-black-hole reproducer for RTX 5090 (SM120).

Probes whether a Triton TMA kernel pulls in the ~3.7 GiB CUDA-driver syscall
buffer by watching NVML device memory before and after launch.

Usage:
  python sm120_tma_mem_blackhole/triton/tma_copy.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import triton
import triton.language as tl

from common.cuda_utils import make_triton_allocator
from common.benchmark import run_blackhole_test

DEVICE = triton.runtime.driver.active.get_active_torch_device()
TILE_M = 128
TILE_N = 128


# --------------------------------------------------------------------------- #
# TMA cluster kernel: tl.make_tensor_descriptor (Triton 3.4-3.6 emits
# shared::cluster scope regardless of num_ctas).
# --------------------------------------------------------------------------- #
@triton.jit
def tma_cluster_kernel(src_ptr, dst_ptr, M, N,
                       TILE_M: tl.constexpr, TILE_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    src_desc = tl.make_tensor_descriptor(
        src_ptr, shape=[M, N], strides=[N, 1],
        block_shape=[TILE_M, TILE_N],
    )
    dst_desc = tl.make_tensor_descriptor(
        dst_ptr, shape=[M, N], strides=[N, 1],
        block_shape=[TILE_M, TILE_N],
    )
    tile = src_desc.load([pid_m * TILE_M, pid_n * TILE_N])
    dst_desc.store([pid_m * TILE_M, pid_n * TILE_N], tile)


def main():
    def runner(a, b):
        triton.set_allocator(make_triton_allocator())
        grid = (1, 1)
        tma_cluster_kernel[grid](a, b, TILE_M, TILE_N,
                                 TILE_M=TILE_M, TILE_N=TILE_N)

    run_blackhole_test(runner, "Triton")


if __name__ == "__main__":
    main()
