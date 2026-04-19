"""
TMA memory-black-hole reproducer for RTX 5090 (SM120).

https://gist.github.com/Harry-Chen/38c0f47ce3eff4469db4a310e763e949

Probes whether a given Triton kernel pulls in the ~3.7 GiB CUDA-driver
TMA buffer on RTX 5090 (SM120) by watching NVML device memory before and
after the kernel is loaded + launched.

Each mode must be run in a FRESH process. The driver only allocates the
internal buffer the first time a kernel whose ELF references
`__cuda_syscall_cp_async_bulk*` is loaded into the context; once it's
warm, later kernels in the same process won't reproduce the jump.

Modes:

  plain       tl.load + tl.store           -- baseline, no TMA
  tma_cluster tl.make_tensor_descriptor    -- Triton lowers G2S to
              shared::cluster scope on 3.4-3.6, pulling in the ~3.7 GiB
              driver syscall buffer (the black hole)

Usage -- legacy single-process (run each mode in a SEPARATE process):
  python triton/gluon_tma_copy.py --mode plain
  python triton/gluon_tma_copy.py --mode tma_cluster

Usage -- process-isolation (parent spawns a fresh child per mode, SIGKILLs
after launch, verifies the 3.7 GiB is fully returned on ctx teardown --
same methodology as experiment_memleak_ctx_bound.py):
  python triton/gluon_tma_copy.py --isolate
  python triton/gluon_tma_copy.py --isolate --mode tma_cluster

Expected -- legacy mode (d measures *only* the blackhole, baselined after
tensors are allocated):
  plain       d ~ 0   GiB
  tma_cluster d ~ 3.7 GiB  (shared::cluster -> syscall buffer)

Expected -- isolate mode (d(T0-T2) is the full parent-visible delta, so it
*includes* the ~0.5 GiB CUDA-context baseline that every child pays):
  plain       d(T0-T2) ~ 0.5 GiB  ctx only
  tma_cluster d(T0-T2) ~ 4.2 GiB  ctx + 3.7 GiB blackhole
  reclaim(T4-T0) ~ 0 for every mode -- SIGKILL returns all memory.

To see the workaround (PTX patch shared::cluster -> shared::cta), run:
  python triton/tma_cta_patch.py
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

import triton
import triton.language as tl

from common.mem_utils import (
    fmt_gb, smi_used_bytes, smi_proc_mem,
    nvml_init, nvml_used, HAS_NVML, DeviceMemorySampler,
)
from common.proc_utils import wait_line, iso_sample

DEVICE = triton.runtime.driver.active.get_active_torch_device()

TILE_M = 128
TILE_N = 128


# --------------------------------------------------------------------------- #
# A device-side busy-wait so NVML / nvidia-smi (~1 Hz) can sample memory
# while the kernel is still resident on the GPU.
# --------------------------------------------------------------------------- #
@triton.jit
def _spin_ns(ns_deadline_delta):
    tl.inline_asm_elementwise(
        asm=(
            "{\n"
            " .reg .pred %p0;\n"
            " .reg .u64 %ts, %tn, %td;\n"
            " mov.u64 %ts, %globaltimer;\n"
            " cvt.u64.u32 %td, $1;\n"
            " add.u64 %td, %ts, %td;\n"
            "SPIN_LOOP:\n"
            " mov.u64 %tn, %globaltimer;\n"
            " setp.lt.u64 %p0, %tn, %td;\n"
            " @%p0 bra SPIN_LOOP;\n"
            " mov.u32 $0, $1;\n"
            "}"
        ),
        constraints="=r,r",
        args=[ns_deadline_delta],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


# --------------------------------------------------------------------------- #
# plain : tl.load + tl.store, one tile per program.
# --------------------------------------------------------------------------- #
@triton.jit
def plain_kernel(src_ptr, dst_ptr, N: tl.constexpr,
                 TILE_M: tl.constexpr, TILE_N: tl.constexpr,
                 SPIN_NS: tl.constexpr = 0):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    ptrs = src_ptr + offs_m[:, None] * N + offs_n[None, :]
    tile = tl.load(ptrs)
    tl.store(dst_ptr + offs_m[:, None] * N + offs_n[None, :], tile)
    if SPIN_NS > 0:
        _spin_ns(SPIN_NS)


# --------------------------------------------------------------------------- #
# tma_cluster : tl.make_tensor_descriptor (Triton 3.4-3.6 emit
#   shared::cluster scope regardless of num_ctas).
# --------------------------------------------------------------------------- #
@triton.jit
def tma_cluster_kernel(src_ptr, dst_ptr, M, N,
                       TILE_M: tl.constexpr, TILE_N: tl.constexpr,
                       SPIN_NS: tl.constexpr = 0):
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
    if SPIN_NS > 0:
        _spin_ns(SPIN_NS)


# --------------------------------------------------------------------------- #
# Allocation logging (Triton-specific)
# --------------------------------------------------------------------------- #
TMA_ALLOC_LOG = []


def make_logging_allocator():
    def alloc_fn(size, alignment, stream):
        TMA_ALLOC_LOG.append(size)
        print(f"  [alloc_fn] size={size/1024/1024:.2f} MiB "
              f"align={alignment} stream={stream}")
        return torch.empty(size, device="cuda", dtype=torch.int8)
    return alloc_fn


def snapshot(tag):
    torch.cuda.synchronize()
    print(f"[{tag}] torch alloc={fmt_gb(torch.cuda.memory_allocated())} "
          f"reserved={fmt_gb(torch.cuda.memory_reserved())} "
          f"peak_alloc={fmt_gb(torch.cuda.max_memory_allocated())}")


# --------------------------------------------------------------------------- #
# Runners
# --------------------------------------------------------------------------- #
def run_plain(a, b, spin_ns=0):
    M, N = a.shape
    grid = (triton.cdiv(M, TILE_M), triton.cdiv(N, TILE_N))
    plain_kernel[grid](a, b, N, TILE_M=TILE_M, TILE_N=TILE_N, SPIN_NS=spin_ns)


def run_tma_cluster(a, b, spin_ns=0):
    M, N = a.shape
    triton.set_allocator(make_logging_allocator())
    grid = (triton.cdiv(M, TILE_M), triton.cdiv(N, TILE_N))
    tma_cluster_kernel[grid](a, b, M, N,
                             TILE_M=TILE_M, TILE_N=TILE_N, SPIN_NS=spin_ns)


MODE_TABLE = {
    "plain": ("plain  (tl.load / tl.store)", run_plain),
    "tma_cluster": ("TMA  tl.make_tensor_descriptor  (shared::cluster -- black hole)",
                    run_tma_cluster),
}


# --------------------------------------------------------------------------- #
# Child runner: minimal work to trigger the lazy driver allocation, then park.
# The kernel (tma_cluster in particular) livelocks on SM120, so we never call
# torch.cuda.synchronize() after launch -- we only need the alloc to hit the
# driver queue, which happens when the Triton dispatch returns.
# --------------------------------------------------------------------------- #
def run_child(mode: str, shape: tuple[int, int]) -> None:
    torch.cuda.init()
    _ = torch.empty(1, device="cuda"); del _
    torch.cuda.synchronize()
    print(f"CHILD_PID={os.getpid()}", flush=True)
    print("CTX_READY", flush=True)

    M, N = shape
    assert M % TILE_M == 0 and N % TILE_N == 0, "shape must be a multiple of tile"
    a = torch.randn((M, N), device=DEVICE, dtype=torch.float16)
    b = torch.empty_like(a)

    _, runner = MODE_TABLE[mode]
    print("COMPILED", flush=True)

    runner(a, b, spin_ns=0)
    # DO NOT synchronize: tma_cluster will livelock on SM120.
    print("LAUNCHED", flush=True)
    print("PARKED", flush=True)
    while True:
        time.sleep(3600)


# --------------------------------------------------------------------------- #
# Parent orchestrator: for each mode, spawn a fresh child, walk T0..T4
# samples, SIGKILL, verify reclaim. Proves the 3.7 GiB leak is strictly
# context-bound (same conclusion as experiment_memleak_ctx_bound.py).
# --------------------------------------------------------------------------- #
def _run_one_isolated(mode: str, shape: tuple[int, int]) -> dict:
    """Run a single mode under isolation; return row for the summary table."""
    label = MODE_TABLE[mode][0]
    print(f"\n{'=' * 72}")
    print(f"=== isolated mode: {mode}  ({label}) ===")
    print(f"{'=' * 72}")

    row: dict = {"mode": mode, "status": "OK"}
    row["T0"], _ = iso_sample("T0.baseline")

    proc = subprocess.Popen(
        [sys.executable, "-u", os.path.abspath(__file__),
         "--child", "--mode", mode, "--shape", str(shape[0]), str(shape[1])],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        try:
            wait_line(proc, "CTX_READY", timeout=60)
        except (RuntimeError, TimeoutError) as e:
            print(f"  !! {e}")
            row["status"] = "CRASH_BEFORE_CTX"
            return row
        time.sleep(0.5)
        row["T1"], _ = iso_sample("T1.ctx_ready", pid=proc.pid)

        try:
            wait_line(proc, "COMPILED", timeout=60)
        except (RuntimeError, TimeoutError) as e:
            print(f"  !! {e}")
            row["status"] = "COMPILE_FAIL"
            return row
        try:
            wait_line(proc, "LAUNCHED", timeout=60)
        except (RuntimeError, TimeoutError) as e:
            print(f"  !! {e}")
            row["status"] = "LAUNCH_FAIL"
            return row
        time.sleep(2.0)  # let the driver reflect the allocation in nvidia-smi
        row["T2"], row["T2_pid"] = iso_sample("T2.post_launch", pid=proc.pid)

        print("  sending SIGKILL ...")
        proc.send_signal(signal.SIGKILL)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("  !! child did not die within 10s")
        row["T3"], _ = iso_sample("T3.killed", pid=proc.pid)

        time.sleep(2.0)
        row["T4"], _ = iso_sample("T4.post_kill", pid=proc.pid)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGKILL)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
    return row


def run_orchestrator(modes: list[str], shape: tuple[int, int]) -> None:
    print(f"Process-isolation test over modes: {modes}")
    print(f"Shape per child: {shape[0]}x{shape[1]} fp16 "
          f"({shape[0] * shape[1] * 2 / 1024 / 1024:.2f} MiB)")

    rows = [_run_one_isolated(m, shape) for m in modes]

    print(f"\n{'=' * 110}")
    print("=== summary (isolated) ===")
    print(f"{'=' * 110}")
    hdr = (f"{'mode':<13} {'status':<18} "
           f"{'T0_free':>10} {'T1_ctx':>10} {'T2_launch':>10} "
           f"{'T3_kill':>10} {'T4_post':>10} "
           f"{'d(T0-T2)':>10} {'reclaim(T4-T0)':>16}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r["status"] != "OK":
            print(f"{r['mode']:<13} {r['status']:<18}")
            continue
        delta = r["T0"] - r["T2"]        # positive => allocation grew
        reclaim = r["T4"] - r["T0"]      # ~ 0  => fully reclaimed on kill
        print(f"{r['mode']:<13} {r['status']:<18} "
              f"{r['T0']:>10.1f} {r['T1']:>10.1f} {r['T2']:>10.1f} "
              f"{r['T3']:>10.1f} {r['T4']:>10.1f} "
              f"{delta:>10.1f} {reclaim:>16.1f}")
    print("\nUnits: MiB. d(T0-T2) = free memory lost while child was alive.")
    print("reclaim(T4-T0) ~ 0 confirms the allocation is CUDA-context-bound")
    print("and is fully released by SIGKILL -- matches experiment_memleak_ctx_bound.")


# --------------------------------------------------------------------------- #
# Legacy single-process runner (original behaviour).
# --------------------------------------------------------------------------- #
def run_legacy(args: argparse.Namespace) -> None:
    label, runner = MODE_TABLE[args.mode]
    M, N = args.shape
    assert M % TILE_M == 0 and N % TILE_N == 0, "shape must be a multiple of tile"
    print(f"=== mode: {label} ===")
    print(f"shape: {M}x{N} fp16 ({M*N*2/1024/1024:.2f} MiB)\n")

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    handle = nvml_init(torch.cuda.current_device())
    source = "nvml" if HAS_NVML else "nvidia-smi (pynvml not installed)"
    nvml_before = nvml_used(handle)
    print(f"[{source}] BEFORE any alloc: {fmt_gb(nvml_before)}")

    a = torch.randn((M, N), device=DEVICE, dtype=torch.float16)
    b = torch.empty_like(a)
    torch.cuda.synchronize()

    nvml_after_inputs = nvml_used(handle)
    snapshot("after inputs")
    print(f"[nvml] after inputs: {fmt_gb(nvml_after_inputs)} "
          f"(d={fmt_gb(nvml_after_inputs - nvml_before)})")

    # --- first launch + spin so nvidia-smi can sample ---
    spin_ns = int(args.spin_seconds * 1e9)
    print(f"\n=== launching {label}, spinning ~{args.spin_seconds:.0f}s "
          "(check nvidia-smi in another shell) ===")
    runner(a, b, spin_ns=spin_ns)
    print(f"  right after launch (pre-sync): "
          f"nvml={fmt_gb(nvml_used(handle))}  "
          f"smi={fmt_gb(smi_used_bytes(torch.cuda.current_device()))}")
    for i in range(int(args.spin_seconds)):
        time.sleep(1.0)
        print(f"  t+{i+1}s  nvml={fmt_gb(nvml_used(handle))}  "
              f"smi={fmt_gb(smi_used_bytes(torch.cuda.current_device()))}")
    torch.cuda.synchronize()
    nvml_after_first = nvml_used(handle)
    print(f"  after sync: nvml={fmt_gb(nvml_after_first)} "
          f"(d vs baseline={fmt_gb(nvml_after_first - nvml_before)})")

    # --- background sampler across 20 reruns (steady state) ---
    print("\n=== 20 more launches (steady state) ===")
    sampler = DeviceMemorySampler(torch.cuda.current_device(), interval_s=0.002)
    sampler.start()
    for _ in range(20):
        runner(a, b, spin_ns=0)
    torch.cuda.synchronize()
    sampler.stop()
    snapshot("after 20 launches")
    print(f"[nvml] peak over 20 launches: {fmt_gb(sampler.peak())} "
          f"(d vs baseline={fmt_gb(sampler.peak() - nvml_before)})")

    # --- summary ---
    print(f"\n=== summary ({label}) ===")
    print(f"alloc_fn called {len(TMA_ALLOC_LOG)} time(s), "
          f"total requested = {fmt_gb(sum(TMA_ALLOC_LOG))}")
    print(f"torch  alloc final: {fmt_gb(torch.cuda.memory_allocated())}")
    print(f"torch  reserv final: {fmt_gb(torch.cuda.memory_reserved())}")
    print(f"nvml   used  final: {fmt_gb(nvml_used(handle))}")
    print(f"smi    used  final: "
          f"{fmt_gb(smi_used_bytes(torch.cuda.current_device()))}")
    me = os.getpid()
    procs = smi_proc_mem(torch.cuda.current_device())
    print(f"smi this-process ({me}): {fmt_gb(procs.get(me, 0))}")
    print(f"nvml d vs baseline: "
          f"{fmt_gb(nvml_used(handle) - nvml_before)}")
    print("(Excess beyond torch reserved = CUDA-context / JIT / TMA tables / "
          "driver syscall buffer.)")

    # --- correctness check (identity copy) ---
    ok = torch.equal(a, b)
    print(f"\n{'OK' if ok else 'MISMATCH'}: identity copy "
          f"{'matches' if ok else 'does NOT match'} torch")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=list(MODE_TABLE.keys()),
                        default="plain",
                        help="Which kernel variant to exercise. In legacy "
                             "(single-process) mode, run the script once "
                             "per mode in SEPARATE processes.")
    parser.add_argument("--shape", nargs=2, type=int, default=[TILE_M, TILE_N],
                        metavar=("M", "N"),
                        help="Source tensor shape. Must be multiples of "
                             f"{TILE_M}x{TILE_N}.")
    parser.add_argument("--spin-seconds", type=float, default=6.0,
                        help="(legacy mode) seconds the first launch "
                             "busy-waits so you can eyeball nvidia-smi.")
    parser.add_argument("--isolate", action="store_true",
                        help="Parent orchestrator: spawn a fresh child per "
                             "mode, SIGKILL after launch, observe GPU-memory "
                             "reclaim. Mirrors experiment_memleak_ctx_bound.")
    parser.add_argument("--modes", nargs="+", choices=list(MODE_TABLE.keys()),
                        default=None,
                        help="(with --isolate) subset of modes to test; "
                             "defaults to all.")
    parser.add_argument("--child", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        return run_child(args.mode, tuple(args.shape))

    if args.isolate:
        if args.modes is not None:
            modes = args.modes
        elif any(f"--mode" == a or a.startswith("--mode=") for a in sys.argv[1:]):
            # User explicitly passed --mode alongside --isolate: run just that one.
            modes = [args.mode]
        else:
            modes = list(MODE_TABLE.keys())
        return run_orchestrator(modes, tuple(args.shape))

    run_legacy(args)


if __name__ == "__main__":
    main()
