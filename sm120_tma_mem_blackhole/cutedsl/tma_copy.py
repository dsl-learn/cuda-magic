"""
CuTe DSL TMA memory-black-hole reproducer for RTX 5090 (SM120).

Probes whether a CuTe DSL TMA kernel pulls in the ~3.7 GiB CUDA-driver syscall
buffer by watching NVML device memory before and after launch.

Note: the compiled kernel livelocks on SM120 (sync never returns), so we read
NVML immediately after launch returns and skip synchronize.

Usage:
  python sm120_tma_mem_blackhole/cutedsl/tma_copy.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

# CuTe DSL env setup (must happen before import cutlass)
_CACHE = Path(__file__).parent / "cute_cache"
_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("CUTE_DSL_CACHE_DIR", str(_CACHE / "mlir_cache"))
os.environ.setdefault("CUTE_DSL_DUMP_DIR", str(_CACHE))
if "CUTE_DSL_ARCH" not in os.environ:
    mj, mn = torch.cuda.get_device_capability(0)
    os.environ["CUTE_DSL_ARCH"] = f"sm_{mj}{mn}"

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu import cpasync

from common.mem_utils import fmt_gb, nvml_init, nvml_used

TILE_M = 128
TILE_N = 128


class _TmaIdentityCopy:
    @cute.jit
    def __call__(self, mA, mB):
        self.dtype = mA.element_type
        tile = (TILE_M, TILE_N)
        smem_layout = cute.make_ordered_layout(tile, order=(1, 0))

        tma_load_atom, tma_gA = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mA, smem_layout, tile,
        )
        tma_store_atom, tma_gB = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mB, smem_layout, tile,
        )

        M = cute.size(mA, mode=[0])
        N = cute.size(mA, mode=[1])
        grid = (
            (M + TILE_M - 1) // TILE_M,
            (N + TILE_N - 1) // TILE_N,
            1,
        )

        self.kernel(
            tma_load_atom, tma_gA,
            tma_store_atom, tma_gB,
            smem_layout,
        ).launch(grid=grid, block=[128, 1, 1])

    @cute.kernel
    def kernel(
        self,
        tma_load_atom, tma_gA,
        tma_store_atom, tma_gB,
        smem_layout,
    ):
        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        smem = utils.SmemAllocator()
        sA = smem.allocate_tensor(self.dtype, smem_layout, byte_alignment=128)
        mbar_ptr = smem.allocate_array(cutlass.Int64, num_elems=1)

        gA_tiled = cute.flat_divide(tma_gA, (TILE_M, TILE_N))
        gB_tiled = cute.flat_divide(tma_gB, (TILE_M, TILE_N))

        bSG_sA, bSG_gA = cpasync.tma_partition(
            tma_load_atom, 0, cute.make_layout(1),
            cute.group_modes(sA, 0, cute.rank(sA)),
            cute.group_modes(gA_tiled, 0, 2),
        )
        bSG_sB, bSG_gB = cpasync.tma_partition(
            tma_store_atom, 0, cute.make_layout(1),
            cute.group_modes(sA, 0, cute.rank(sA)),
            cute.group_modes(gB_tiled, 0, 2),
        )

        tile_bytes = cute.size_in_bytes(self.dtype, smem_layout)

        if tidx == 0:
            cute.arch.mbarrier_init(mbar_ptr, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        if tidx == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, tile_bytes)
            cute.copy(
                tma_load_atom,
                bSG_gA[(None, bidx, bidy)],
                bSG_sA[(None,)],
                tma_bar_ptr=mbar_ptr,
            )
        cute.arch.mbarrier_wait(mbar_ptr, phase=0)

        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()

        if tidx == 0:
            cute.copy(
                tma_store_atom,
                bSG_sB[(None,)],
                bSG_gB[(None, bidx, bidy)],
            )
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=False)


def main():
    torch.cuda.init()
    handle = nvml_init(torch.cuda.current_device())

    nvml_before = nvml_used(handle)
    print(f"[CuTeDSL][nvml] BEFORE any alloc: {fmt_gb(nvml_before)}")

    a = torch.randn((TILE_M, TILE_N), dtype=torch.bfloat16, device="cuda")
    b = torch.empty_like(a)
    a_ct = from_dlpack(a).mark_layout_dynamic(leading_dim=1)
    b_ct = from_dlpack(b).mark_layout_dynamic(leading_dim=1)
    torch.cuda.synchronize()

    nvml_after_inputs = nvml_used(handle)
    print(f"[CuTeDSL][nvml] after inputs: {fmt_gb(nvml_after_inputs)} "
          f"(d={fmt_gb(nvml_after_inputs - nvml_before)})")

    copy = _TmaIdentityCopy()
    compiled = cute.compile(copy, a_ct, b_ct)

    print("\n=== launching CuTeDSL TMA kernel ===")
    compiled(a_ct, b_ct)
    # Skip sync: kernel livelocks on SM120. The blackhole allocation happens
    # at launch-queue time, so NVML already captures it.
    nvml_after = nvml_used(handle)
    print(f"[CuTeDSL][nvml] after launch: {fmt_gb(nvml_after)} "
          f"(d vs baseline={fmt_gb(nvml_after - nvml_before)})")
    print(f"\n[CuTeDSL] Black hole jump (launch - inputs): "
          f"{fmt_gb(nvml_after - nvml_after_inputs)}")
    print("Note: kernel livelocks on SM120; sync skipped.")


if __name__ == "__main__":
    main()
