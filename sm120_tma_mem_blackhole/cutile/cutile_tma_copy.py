"""
Minimal cuTile TMA workload on sm_120 — diagnose whether the CUTE-DSL
blackhole + livelock reproduce under NVIDIA's own cuTile tile-DSL.

A naive `ct.load` + `ct.store` identity copy is lowered by cuTile to
`LDG.E.128` / `STG.E.128` on sm_120 — the compiler sees no reason to
stage a pass-through tile through smem. To actually exercise the TMA
path we need a kernel that forces the tile into shared memory. The
minimal pattern that does so (see `_probe_tma_trigger.py`) is a
row-reduce + broadcast-subtract (row-wise mean-center):

    t  = load(src)              # (TM, TN) -> smem via TMA
    s  = sum(t, axis=1)         # (TM,)
    out = t - s / TN            # broadcast back
    store(dst, out)

cuTile emits 4× `UTMALDG.2D` for this, zero `LDG.E`/`STG.E`. If CUTE
DSL's sm_120 bug were a hardware/driver problem, this kernel should
also:
  - pre-allocate ~3.7 GiB at first launch (blackhole)
  - hang in the mbarrier phase wait (livelock)
  - emit `__cuda_syscall_cp_async_bulk_tensor_*` in the cubin ELF

We check all three and report.
"""

import time
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CACHE_ROOT = Path(__file__).parent / "cutile_cache"
CACHE_ROOT.mkdir(exist_ok=True)

import torch
import cuda.tile as ct
import cuda.tile._cext as _ctx

_ctx.default_tile_context.config.cache_dir = str(CACHE_ROOT)

MiB = 1024 * 1024
ConstInt = ct.Constant[int]

TILE_M = 128
TILE_N = 128

from common.cuda_utils import resolve_cuobjdump, dump_cubins, analyze_cubin
from common.mem_utils import snapshot


@ct.kernel
def tma_mean_center_2d(src, dst, TM: ConstInt, TN: ConstInt):
    """Row-wise mean-center. Forces the tile through smem -> cuTile picks TMA."""
    bidx = ct.bid(0)
    bidy = ct.bid(1)
    t = ct.load(src, index=(bidx, bidy), shape=(TM, TN))
    s = ct.sum(t, axis=1)                          # (TM,)
    out = t - ct.expand_dims(s, axis=1) / TN       # broadcast back
    ct.store(dst, index=(bidx, bidy), tile=out.astype(t.dtype))


def main() -> None:
    torch.cuda.init()
    _ = torch.empty(1, device="cuda"); del _
    torch.cuda.synchronize()
    cc = torch.cuda.get_device_capability(0)
    print(f"device: {torch.cuda.get_device_name(0)}  cc: sm_{cc[0]}{cc[1]}")
    snapshot("A. ctx only")

    M, N = 1024, 2048
    assert M % TILE_M == 0 and N % TILE_N == 0
    src = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    dst = torch.empty_like(src)
    snapshot("B. tensors alloc")

    stream = torch.cuda.current_stream()
    grid = (M // TILE_M, N // TILE_N, 1)

    t0 = time.time()
    ct.launch(stream, grid, tma_mean_center_2d, (src, dst, TILE_M, TILE_N))
    t_launch = time.time() - t0
    snapshot("C. post first launch")

    t1 = time.time()
    torch.cuda.synchronize()
    t_sync = time.time() - t1
    snapshot("D. post synchronize")

    print(f"\nfirst-launch wall: {t_launch:.3f}s")
    print(f"synchronize  wall: {t_sync:.3f}s (>60s would mean livelock)")

    # Second launch — should hit cache, no further driver alloc.
    t2 = time.time()
    ct.launch(stream, grid, tma_mean_center_2d, (src, dst, TILE_M, TILE_N))
    torch.cuda.synchronize()
    print(f"second-launch+sync: {time.time()-t2:.3f}s")
    snapshot("E. post second launch")

    # Correctness: compare against torch row-wise mean-center in bf16.
    # (fp32 reference would blow up relative error due to bf16 row-sum rounding.)
    ref = (src.float() - src.float().mean(dim=1, keepdim=True)).to(torch.bfloat16)
    err = (dst.float() - ref.float()).abs().max() / ref.float().abs().max().clamp_min(1e-6)
    print(f"\nmax relative err (vs bf16 ref): {err.item():.3e}")

    print("\n=== cubin analysis ===")
    dump_dir = CACHE_ROOT / "_dumped"
    paths = dump_cubins(CACHE_ROOT / "cache.db", dump_dir)
    print(f"dumped {len(paths)} cubin(s) from cache.db")
    verdict_tma = False
    verdict_clean = True
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
            print(f"  ELF syscall refs ({len(r['syscall_refs'])}):")
            for h in r["syscall_refs"][:5]:
                print(f"    {h}")
            verdict_clean = False
        else:
            print("  ELF syscall refs: NONE")
        if r["UTMALDG"] > 0 or r["UBLKCP"] > 0:
            verdict_tma = True

    print("\n=== verdict ===")
    print(f"  TMA actually used?             {'YES' if verdict_tma else 'NO'}")
    print(f"  cubin clean (no syscall sym)?  {'YES' if verdict_clean else 'NO'}")
    print(f"  kernel livelocked?             {'YES' if t_sync > 60 else 'NO'}")
    print(f"  correct?                       {'YES' if err.item() < 1e-1 else 'NO'}")


if __name__ == "__main__":
    main()
