"""
cuTile TMA reproducer + cubin analysis for RTX 5090 (SM120).

A reduce+broadcast kernel forces cuTile to stage the tile through smem,
which emits UTMALDG.2D (TMA).  On SM120 this does NOT trigger the ~3.7 GiB
black hole because cuTile's codegen correctly uses shared::cta scope.

This script both measures NVML and scans the generated cubin to prove
the TMA path is clean (no __cuda_syscall_* symbols).

Usage:
  python sm120_tma_mem_blackhole/cutile/cutile_tma_copy.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import cuda.tile as ct
import cuda.tile._cext as _ctx

from common.cuda_utils import warmup_cuda_context, resolve_cuobjdump, dump_cubins, analyze_cubin
from common.mem_utils import fmt_gb, nvml_init, nvml_used

CACHE = Path(__file__).parent / "cutile_cache"
CACHE.mkdir(exist_ok=True)
_ctx.default_tile_context.config.cache_dir = str(CACHE)

ConstInt = ct.Constant[int]
TILE_M = 128
TILE_N = 128


@ct.kernel
def tma_reduce_broadcast_kernel(src, dst, TM: ConstInt, TN: ConstInt):
    """Row-wise mean-center. Forces tile through smem -> cuTile picks TMA."""
    bidx = ct.bid(0)
    bidy = ct.bid(1)
    t = ct.load(src, index=(bidx, bidy), shape=(TM, TN))
    s = ct.sum(t, axis=1)
    out = t - ct.expand_dims(s, axis=1) / TN
    ct.store(dst, index=(bidx, bidy), tile=out.astype(t.dtype))


def main():
    warmup_cuda_context()
    handle = nvml_init(torch.cuda.current_device())

    nvml_before = nvml_used(handle)
    print(f"[cuTile][nvml] BEFORE any alloc: {fmt_gb(nvml_before)}")

    a = torch.randn((TILE_M, TILE_N), dtype=torch.bfloat16, device="cuda")
    b = torch.empty_like(a)
    torch.cuda.synchronize()

    nvml_after_inputs = nvml_used(handle)
    print(f"[cuTile][nvml] after inputs: {fmt_gb(nvml_after_inputs)} "
          f"(d={fmt_gb(nvml_after_inputs - nvml_before)})")

    print("\n=== launching cuTile TMA kernel (reduce+broadcast) ===")
    stream = torch.cuda.current_stream()
    grid = (1, 1, 1)
    ct.launch(stream, grid, tma_reduce_broadcast_kernel,
              (a, b, TILE_M, TILE_N))
    torch.cuda.synchronize()

    nvml_after = nvml_used(handle)
    print(f"[cuTile][nvml] after launch: {fmt_gb(nvml_after)} "
          f"(d vs baseline={fmt_gb(nvml_after - nvml_before)})")
    print(f"\n[cuTile] Black hole jump (launch - inputs): "
          f"{fmt_gb(nvml_after - nvml_after_inputs)}")

    ref = (a.float() - a.float().mean(dim=1, keepdim=True)).to(torch.bfloat16)
    err = (b.float() - ref.float()).abs().max() / ref.float().abs().max().clamp_min(1e-6)
    ok = err.item() < 1e-1
    print(f"{'OK' if ok else 'MISMATCH'}: row-mean-center "
          f"{'matches' if ok else 'does NOT match'} torch (rel_err={err.item():.2e})")

    # ---- cubin analysis ----------------------------------------------------
    print("\n=== cubin analysis ===")
    dump_dir = CACHE / "_dumped"
    paths = dump_cubins(CACHE / "cache.db", dump_dir)
    print(f"dumped {len(paths)} cubin(s) from cache.db")
    cuobjdump_path = resolve_cuobjdump()
    for p in paths:
        r = analyze_cubin(p, cuobjdump_path=cuobjdump_path)
        print(f"\n{p.name}")
        print(f"  UTMALDG.*:      {r['UTMALDG']}")
        print(f"  UBLKCP.*:       {r['UBLKCP']}")
        print(f"  CALL.ABS.NOINC: {r['CALL_ABS']}")
        print(f"  LDG.E:          {r['LDG']}")
        print(f"  STG.E:          {r['STG']}")
        if r["syscall_refs"]:
            print(f"  ELF syscall refs ({len(r['syscall_refs'])}) — WOULD LEAK:")
            for h in r["syscall_refs"][:5]:
                print(f"    {h}")
        else:
            print("  ELF syscall refs: NONE (clean)")


if __name__ == "__main__":
    main()
