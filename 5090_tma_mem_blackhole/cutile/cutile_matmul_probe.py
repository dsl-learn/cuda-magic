"""
cuTile matmul probe — forces cuTile to emit TMA on sm_120.

The plain ct.load/ct.store identity copy (see cutile_tma_copy.py) compiled to
LDG.E.128 / STG.E.128 on sm_120, not TMA. Matmul with ct.mma pipelining is the
canonical TMA-consumer in cuTile. This script runs a minimal matmul, then
inspects the cached cubin for:

  1. UTMALDG.* / UBLKCP.* in SASS  -> confirms TMA is actually used
  2. __cuda_syscall_cp_async_bulk_* in ELF symbols -> would indicate blackhole
  3. kernel completion + correctness -> would indicate no livelock
  4. driver_used delta -> would quantify any blackhole

Reference: cutile-learn/tutorials/03-matrix-multiplication.py.
"""

import time
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CACHE_ROOT = Path(__file__).parent / "cutile_matmul_cache"
CACHE_ROOT.mkdir(exist_ok=True)

import torch
import cuda.tile as ct
import cuda.tile._cext as _ctx

_ctx.default_tile_context.config.cache_dir = str(CACHE_ROOT)

MiB = 1024 * 1024
ConstInt = ct.Constant[int]

from common.cuda_utils import resolve_cuobjdump, dump_cubins, analyze_cubin
from common.mem_utils import snapshot


@ct.kernel
def matmul_kernel(A, B, C,
                  TILE_M: ConstInt,
                  TILE_N: ConstInt,
                  TILE_K: ConstInt):
    bidx = ct.bid(0)
    bidy = ct.bid(1)
    acc = ct.full((TILE_M, TILE_N), 0, dtype=ct.float32)
    num_k = ct.num_tiles(A, axis=1, shape=(TILE_M, TILE_K))
    for k in range(num_k):
        a = ct.load(A, index=(bidx, k), shape=(TILE_M, TILE_K))
        b = ct.load(B, index=(k, bidy), shape=(TILE_K, TILE_N))
        acc = ct.mma(a, b, acc)
    ct.store(C, index=(bidx, bidy), tile=ct.astype(acc, C.dtype))


def main() -> None:
    torch.cuda.init()
    _ = torch.empty(1, device="cuda"); del _
    torch.cuda.synchronize()
    cc = torch.cuda.get_device_capability(0)
    print(f"device: {torch.cuda.get_device_name(0)}  cc: sm_{cc[0]}{cc[1]}")
    snapshot("A. ctx only")

    # sm_120 autotuner config from cutile-learn: 128x64 with K=64 (fits occupancy=1)
    TM, TN, TK = 128, 64, 64
    M, N, K = 512, 512, 512  # multiple tiles to force K-loop + TMA pipelining

    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    C = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    snapshot("B. tensors alloc")

    stream = torch.cuda.current_stream()
    grid = (M // TM, N // TN, 1)
    print(f"grid={grid}, tile=({TM},{TN},{TK})")

    t0 = time.time()
    ct.launch(stream, grid, matmul_kernel, (A, B, C, TM, TN, TK))
    t_launch = time.time() - t0
    snapshot("C. post first launch")

    t1 = time.time()
    torch.cuda.synchronize()
    t_sync = time.time() - t1
    snapshot("D. post synchronize")

    print(f"\nfirst-launch wall:  {t_launch:.3f}s")
    print(f"first sync  wall:  {t_sync:.3f}s (>60s = livelock)")

    # Correctness
    C_ref = (A.float() @ B.float()).to(torch.bfloat16)
    rel_err = (C.float() - C_ref.float()).abs().max() / C_ref.float().abs().max()
    print(f"max relative err:  {rel_err.item():.3e}")

    # Dump cached cubins + analyze
    print("\n=== cubin analysis ===")
    dump_dir = CACHE_ROOT / "_dumped"
    paths = dump_cubins(CACHE_ROOT / "cache.db", dump_dir)
    print(f"dumped {len(paths)} cubin(s) from cache.db")
    cuobjdump_path = resolve_cuobjdump()
    for p in paths:
        print(f"\n{p.name}  ({p.stat().st_size} bytes)")
        r = analyze_cubin(p, cuobjdump_path=cuobjdump_path)
        print(f"  UTMALDG.*:      {r['UTMALDG']}")
        print(f"  UBLKCP.*:       {r['UBLKCP']}")
        print(f"  CALL.ABS.NOINC: {r['CALL_ABS']}")
        print(f"  LDG.*:          {r['LDG']}")
        print(f"  STG.*:          {r['STG']}")
        if r["syscall_refs"]:
            print(f"  ELF syscall refs ({len(r['syscall_refs'])}):")
            for h in r["syscall_refs"][:5]:
                print(f"    {h}")
        else:
            print("  ELF syscall refs: NONE")


if __name__ == "__main__":
    main()
