"""
Minimal CUTE DSL TMA load/store primitive, as a drop-in mental model
for Triton's `tl._experimental_descriptor_load` / `_store`.

Why this avoids the 5090 / SM120 `shared::cluster` memory leak:
  `cpasync.make_tiled_tma_atom(CopyBulkTensorTileG2SOp(), ..., num_multicast=1)`
  with a single-CTA kernel (no cluster, no multicast) emits
    cp.async.bulk.tensor.Nd.shared::cta.global.tile.mbarrier::complete_tx::bytes
  i.e. the inline UTMALDG.2D SASS form. No `__cuda_syscall_*` reference gets
  baked into the ELF, so the driver never pre-allocates the ~3.7 GiB buffer.

Run:
  python cutedsl/cute_tma_copy.py
"""

import os
import sys
from pathlib import Path

# Allow `import common` from a sibling directory.
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from common.cuda_utils import verify_no_syscall

# Pin the CUTE DSL cache + dump locations to a single repo-local folder so the
# generated cubin is easy to find for post-hoc verification. Set these BEFORE
# importing cutlass — env_manager reads them at import time.
CACHE_ROOT = Path(__file__).parent / "cute_cache"
CACHE_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("CUTE_DSL_CACHE_DIR", str(CACHE_ROOT / "mlir_cache"))
os.environ.setdefault("CUTE_DSL_DUMP_DIR", str(CACHE_ROOT))
os.environ.setdefault("CUTE_DSL_KEEP_CUBIN", "1")
os.environ.setdefault("CUTE_DSL_KEEP_PTX", "1")

# Auto-set CUTE_DSL_ARCH from the current device before importing cutlass,
# so CopyBulkTensorTileG2SOp's arch check picks up e.g. sm_120 on 5090.
if "CUTE_DSL_ARCH" not in os.environ and torch.cuda.is_available():
    _maj, _min = torch.cuda.get_device_capability(0)
    os.environ["CUTE_DSL_ARCH"] = f"sm_{_maj}{_min}"

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu import cpasync


TILE_M = 128
TILE_N = 128


class TmaIdentityCopy:
    """gB = gA, done tile-by-tile via TMA load → smem → TMA store."""

    def __init__(self, tile_mn=(TILE_M, TILE_N)):
        self.tile_m, self.tile_n = tile_mn

    @cute.jit
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor):
        self.dtype = mA.element_type
        tile = (self.tile_m, self.tile_n)

        # SMEM layout for one tile. Column-major so the fast dim matches
        # GMEM's innermost (row-major torch tensor becomes (M,N) with
        # stride (N,1), mark_layout_dynamic below pins leading_dim=0).
        smem_layout = cute.make_ordered_layout(tile, order=(1, 0))

        # TMA load atom. num_multicast=1 → shared::cta (no syscall path).
        tma_load_atom, tma_gA = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mA, smem_layout, tile,
        )
        # TMA store atom. S2G never uses cluster scope anyway.
        tma_store_atom, tma_gB = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mB, smem_layout, tile,
        )

        M = cute.size(mA, mode=[0])
        N = cute.size(mA, mode=[1])
        grid = (
            (M + self.tile_m - 1) // self.tile_m,
            (N + self.tile_n - 1) // self.tile_n,
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
        tma_load_atom: cute.CopyAtom, tma_gA: cute.Tensor,
        tma_store_atom: cute.CopyAtom, tma_gB: cute.Tensor,
        smem_layout: cute.Layout,
    ):
        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        # --- SMEM: one tile + one mbarrier ------------------------------
        smem = utils.SmemAllocator()
        sA = smem.allocate_tensor(self.dtype, smem_layout, byte_alignment=128)
        mbar_ptr = smem.allocate_array(cutlass.Int64, num_elems=1)

        # Partition TMA tensors into (TMA_slice, tile_coord) form for a
        # single-CTA (no cluster) launch: cta_coord=0, cta_layout=Layout(1).
        gA_tiled = cute.flat_divide(tma_gA, (self.tile_m, self.tile_n))
        gB_tiled = cute.flat_divide(tma_gB, (self.tile_m, self.tile_n))

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

        # --- TMA load --------------------------------------------------
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

        # --- (put your compute over sA here) ---------------------------
        # For demo: identity copy, so sA is already the tile to store.

        # --- TMA store -------------------------------------------------
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
    torch.manual_seed(0)
    dev = torch.cuda.current_device()
    cc = torch.cuda.get_device_capability(dev)
    print(f"device: {torch.cuda.get_device_name(dev)}  cc: sm_{cc[0]}{cc[1]}")
    if cc[0] < 9:
        print("TMA requires sm_90+. Skipping launch (compile-only).")

    M, N = 1024, 2048
    assert M % TILE_M == 0 and N % TILE_N == 0
    A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    B = torch.empty_like(A)

    a_ct = from_dlpack(A).mark_layout_dynamic(leading_dim=1)
    b_ct = from_dlpack(B).mark_layout_dynamic(leading_dim=1)

    copy = TmaIdentityCopy()
    compiled = cute.compile(copy, a_ct, b_ct)
    print("compile: OK")

    # Scan the cubin regardless of whether we can launch — the whole point of
    # the cluster-scope leak is that it happens at *load* time, so the ELF
    # symbol scan is the diagnosis that matters.
    verify_no_syscall(CACHE_ROOT)

    if cc[0] < 9:
        return
    compiled(a_ct, b_ct)
    torch.cuda.synchronize()

    ok = torch.equal(A, B)
    print(f"identity copy correct: {ok}")


if __name__ == "__main__":
    main()
