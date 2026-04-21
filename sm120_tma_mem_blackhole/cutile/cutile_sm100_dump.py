"""
Dump cuTile PTX forced to sm_100 codegen on an sm_120 host.

Monkey-patches cuda.tile._compile.get_sm_arch to return "sm_100" so the
compiler targets Blackwell-datacenter instead of the local RTX 5090 (sm_120).

This checks whether cluster-scope TMA (multicast::cluster) appears in the
PTX when num_ctas=2 is requested. The actual launch on sm_120 will fail,
but the PTX dump happens before ptxas, so we capture it.

Usage:
  python sm120_tma_mem_blackhole/cutile/cutile_sm100_dump.py
"""

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE = Path(__file__).parent
PTX_DIR = BASE / "cutile_sm100_ptx_cache"
PTX_DIR.mkdir(exist_ok=True)
os.environ["PTX_DUMP_DIR"] = str(PTX_DIR)

CACHE = BASE / "cutile_sm100_cache"
if CACHE.exists():
    shutil.rmtree(CACHE)
CACHE.mkdir()

import torch
import cuda.tile as ct
import cuda.tile._cext as _cext
import cuda.tile._compile as _comp

_cext.default_tile_context.config.cache_dir = str(CACHE)

# Force sm_100 codegen
_comp.get_sm_arch = lambda: "sm_100"

ConstInt = ct.Constant[int]


@ct.kernel(num_ctas=ct.ByTarget(sm_100=2))
def matmul_sm100(A, B, C,
                 TM: ConstInt, TN: ConstInt, TK: ConstInt):
    bidx = ct.bid(0)
    bidy = ct.bid(1)
    acc = ct.full((TM, TN), 0, dtype=ct.float32)
    num_k = ct.num_tiles(A, axis=1, shape=(TM, TK))
    for k in range(num_k):
        a = ct.load(A, index=(bidx, k), shape=(TM, TK))
        b = ct.load(B, index=(k, bidy), shape=(TK, TN))
        acc = ct.mma(a, b, acc)
    ct.store(C, index=(bidx, bidy), tile=ct.astype(acc, C.dtype))


def main():
    torch.cuda.init()
    TM, TN, TK = 128, 64, 64
    M, N, K = 512, 512, 512
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    C = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    try:
        ct.launch(
            torch.cuda.current_stream(),
            (M // TM, N // TN, 1),
            matmul_sm100, (A, B, C, TM, TN, TK),
        )
        torch.cuda.synchronize()
    except Exception as e:
        print(f"launch failed (expected on sm_120 host): {type(e).__name__}: {str(e)[:200]}")

    print("=== dumped PTX ===")
    for p in sorted(PTX_DIR.glob("*.ptx")):
        text = p.read_text()
        cluster = text.count("shared::cluster")
        multicast = text.count("multicast::cluster")
        print(f"  {p.name}  ({p.stat().st_size} bytes)")
        print(f"    shared::cluster:   {cluster}")
        print(f"    multicast::cluster: {multicast}")


if __name__ == "__main__":
    main()
