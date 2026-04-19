"""
Probe which ct.load knob pushes cuTile into TMA on sm_120 for a pure copy.
"""

import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import cuda.tile as ct
import cuda.tile._cext as _ctx

BASE = Path(__file__).parent
ConstInt = ct.Constant[int]

from common.cuda_utils import resolve_cuobjdump, count_sass

CUOBJDUMP = resolve_cuobjdump()


def dump_cubin(cache):
    """Return the most recent ELF blob from a cuTile sqlite cache."""
    db = cache / "cache.db"
    if not db.exists():
        return None
    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT blob FROM cache ORDER BY atime DESC").fetchall()
    conn.close()
    for (blob,) in rows:
        if blob[:4] == b"\x7fELF":
            p = cache / "probe.cubin"
            p.write_bytes(blob)
            return p
    return None


def trial(name, kernel, setup, grid_fn, args_fn):
    cache = BASE / f"tma_probe_{name}"
    if cache.exists():
        shutil.rmtree(cache)
    cache.mkdir()
    _ctx.default_tile_context.config.cache_dir = str(cache)
    args = setup()
    ct.launch(torch.cuda.current_stream(), grid_fn(args), kernel, args_fn(args))
    torch.cuda.synchronize()
    cubin = dump_cubin(cache)
    if cubin is None:
        print(f"[{name:<35}] no cubin")
        return
    c = count_sass(cubin, {
        "UTMALDG": "UTMALDG",
        "UBLKCP": "UBLKCP",
        "CALL_ABS": "CALL.ABS",
        "LDG.E": " LDG.E",
        "STG.E": " STG.E",
    }, cuobjdump_path=CUOBJDUMP)
    verdict = "TMA" if c["UTMALDG"] > 0 or c["UBLKCP"] > 0 else "LDG/STG"
    print(f"[{name:<35}] {verdict:<8} UTMALDG={c['UTMALDG']} UBLKCP={c['UBLKCP']} LDG.E={c['LDG.E']} STG.E={c['STG.E']}")


# --- kernel variants ---

@ct.kernel
def k_plain(src, dst, TM: ConstInt, TN: ConstInt):
    bx, by = ct.bid(0), ct.bid(1)
    t = ct.load(src, index=(bx, by), shape=(TM, TN))
    ct.store(dst, index=(bx, by), tile=t)


@ct.kernel
def k_latency_hi(src, dst, TM: ConstInt, TN: ConstInt):
    bx, by = ct.bid(0), ct.bid(1)
    t = ct.load(src, index=(bx, by), shape=(TM, TN), latency=10)
    ct.store(dst, index=(bx, by), tile=t)


@ct.kernel
def k_latency_lo(src, dst, TM: ConstInt, TN: ConstInt):
    bx, by = ct.bid(0), ct.bid(1)
    t = ct.load(src, index=(bx, by), shape=(TM, TN), latency=1)
    ct.store(dst, index=(bx, by), tile=t)


@ct.kernel
def k_two_loads_add(src, src2, dst, TM: ConstInt, TN: ConstInt):
    # two tiles + an op -> compiler might route through smem
    bx, by = ct.bid(0), ct.bid(1)
    a = ct.load(src, index=(bx, by), shape=(TM, TN), latency=10)
    b = ct.load(src2, index=(bx, by), shape=(TM, TN), latency=10)
    ct.store(dst, index=(bx, by), tile=a + b)


@ct.kernel
def k_reduce_broadcast(src, dst, TM: ConstInt, TN: ConstInt):
    # reduce then broadcast back -> forces tile in smem
    bx, by = ct.bid(0), ct.bid(1)
    t = ct.load(src, index=(bx, by), shape=(TM, TN), latency=10)
    s = ct.sum(t, axis=1)          # (TM,)
    out = t - ct.expand_dims(s, axis=1) / TN
    ct.store(dst, index=(bx, by), tile=out.astype(t.dtype))


@ct.kernel
def k_large_tile(src, dst, TM: ConstInt, TN: ConstInt):
    # tile big enough that LDG.128 would need many instructions -> TMA is
    # plausibly the compiler's preferred strategy.
    bx, by = ct.bid(0), ct.bid(1)
    t = ct.load(src, index=(bx, by), shape=(TM, TN), latency=10)
    ct.store(dst, index=(bx, by), tile=t)


def main():
    torch.cuda.init()
    M, N = 1024, 2048

    def _setup(TM, TN, dt=torch.bfloat16):
        def _s():
            A = torch.randn(M, N, dtype=dt, device="cuda")
            B = torch.empty_like(A)
            return A, B, TM, TN
        return _s

    def _setup2(TM, TN, dt=torch.bfloat16):
        def _s():
            A = torch.randn(M, N, dtype=dt, device="cuda")
            A2 = torch.randn(M, N, dtype=dt, device="cuda")
            B = torch.empty_like(A)
            return A, A2, B, TM, TN
        return _s

    def gf(args):
        M_, N_ = args[0].shape
        TM_, TN_ = args[-2], args[-1]
        return (M_ // TM_, N_ // TN_, 1)

    trials = [
        ("plain_128x128",            k_plain,           _setup(128, 128)),
        ("latency=10_128x128",       k_latency_hi,      _setup(128, 128)),
        ("latency=1_128x128",        k_latency_lo,      _setup(128, 128)),
        ("two_loads_add_128x128",    k_two_loads_add,   _setup2(128, 128)),
        ("reduce_broadcast_128x128", k_reduce_broadcast,_setup(128, 128)),
        ("large_256x256",            k_large_tile,      _setup(256, 256)),
        ("large_512x128",            k_large_tile,      _setup(512, 128)),
    ]
    for name, kernel, setup in trials:
        trial(name, kernel, setup, gf, lambda a: a)


if __name__ == "__main__":
    main()
