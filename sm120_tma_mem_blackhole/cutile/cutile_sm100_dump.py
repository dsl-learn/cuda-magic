"""
Force cuTile to codegen for sm_100 on an sm_120 host, to check whether
the matmul kernel (which has `num_ctas=ct.ByTarget(sm_100=2)`) triggers
`.multicast::cluster` or other cluster-scope TMA on Blackwell-datacenter.

We monkeypatch `cuda.tile._compile.get_sm_arch` to return 'sm_100a', which
makes tileiras target sm_100 for cubin emission. The actual launch on the
local sm_120 device will fail, but we only need the PTX that the wrapper
captures before ptxas assembles.
"""
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE = Path(__file__).parent
os.environ["PTX_DUMP_DIR"] = str(BASE / "cutile_sm100_ptx_cache")
Path(os.environ["PTX_DUMP_DIR"]).mkdir(exist_ok=True)

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
_orig = _comp.get_sm_arch
_comp.get_sm_arch = lambda: "sm_100"
print(f"forced get_sm_arch: {_comp.get_sm_arch()}")

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
    stream = torch.cuda.current_stream()
    grid = (M // TM, N // TN, 1)
    try:
        ct.launch(stream, grid, matmul_sm100, (A, B, C, TM, TN, TK))
        torch.cuda.synchronize()
    except Exception as e:
        # Expected: running sm_100 cubin on sm_120 device will fail,
        # but the PTX was already dumped by the wrapper.
        print(f"launch/sync failed (expected on sm_120 host): {type(e).__name__}: {str(e)[:200]}")

    print("\n=== dumped PTX ===")
    for p in sorted(Path(os.environ["PTX_DUMP_DIR"]).glob("*.ptx")):
        print(f"  {p}  ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
