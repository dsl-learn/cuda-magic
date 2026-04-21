"""
Triton Runner PTX patch: leverage triton_runner's source-override to feed patched PTX back.

Requires ``triton_runner`` (https://github.com/.../triton-runner) and a call to
``configure_jit_backend()`` so that ``ptx_src=...`` is accepted by the JIT.

Instead of calling ptxas and cuLaunchKernel manually, we compile once,
patch the PTX string, then pass ``source_type="ptx_src"`` back to Triton
so it reuses the patched assembly through the normal driver path.

Usage:
  python sm120_tma_mem_blackhole/triton/tma_triton_runner_patch.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import triton
import triton.language as tl

import triton_runner
triton_runner.configure_jit_backend()

from common.benchmark import run_blackhole_test

TILE_M = 128
TILE_N = 128


# --------------------------------------------------------------------------- #
# Same kernel as tma_copy.py's tma_cluster_kernel.
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


def runner(a, b):
    """Compile -> extract PTX -> patch -> relaunch via triton_runner source override."""
    M, N = a.shape
    grid = (triton.cdiv(M, TILE_M), triton.cdiv(N, TILE_N))

    # 1. Normal compile to extract generated PTX.
    compiled = tma_cluster_kernel.warmup(
        a, b, M, N, TILE_M=TILE_M, TILE_N=TILE_N, grid=grid
    )
    ptx = compiled.asm["ptx"]

    NEEDLE = "cp.async.bulk.tensor.2d.shared::cluster.global"
    REPLACE = "cp.async.bulk.tensor.2d.shared::cta.global"
    assert NEEDLE in ptx, "PTX does not contain shared::cluster -- nothing to patch"
    ptx_patched = ptx.replace(NEEDLE, REPLACE)

    # 2. Feed patched PTX back through triton_runner's source override.
    #    Triton handles ptxas, module loading, and launch for us.
    tma_cluster_kernel[grid](
        a, b, M, N,
        TILE_M=TILE_M, TILE_N=TILE_N,
        ptx_src=ptx_patched,
        metadata_json=compiled.metadata,
    )


def main():
    triton.set_allocator(
        lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8)
    )
    run_blackhole_test(runner, "TritonRunnerPatch")


if __name__ == "__main__":
    main()
