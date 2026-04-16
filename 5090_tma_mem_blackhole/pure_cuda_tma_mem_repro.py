"""
Triton / Gluon counterpart to pure_cuda_tma_mem_repro.cu.

https://gist.github.com/Harry-Chen/38c0f47ce3eff4469db4a310e763e949

Probes whether a given Triton kernel pulls in the ~3.7 GiB CUDA-driver
TMA buffer on RTX 5090 (SM120) by watching NVML device memory before and
after the kernel is loaded + launched.

Each mode must be run in a FRESH process. The driver only allocates the
internal buffer the first time a kernel whose ELF references
`__cuda_syscall_cp_async_bulk*` is loaded into the context; once it's
warm, later kernels in the same process won't reproduce the jump.

Modes (mapped to pure_cuda_tma_mem_repro.cu rows):

  plain           tl.load + tl.store                 — baseline, no TMA
  tma_cta         tl.make_tensor_descriptor, 1 CTA   — whatever scope the
                  current branch's codegen picks for the descriptor path
                  (on post-fix branches this is shared::cta; on older
                  branches it can still be shared::cluster → black hole)
  tma_gluon_cta   Gluon tma.async_copy_global_to_shared with
                  multicast=False, num_ctas=1       — the EXPLICIT bypass:
                  Gluon lets you pass `multicast=False` directly, so the
                  lowering emits shared::cta regardless of branch state.
                  Use this mode's structure as the workaround when
                  tl.make_tensor_descriptor on your branch still triggers
                  the syscall buffer.
  tma_multicast   Gluon tma.async_copy with multicast=True, num_ctas=2
                  — emits shared::cluster → driver pulls in the
                  ~3.7 GiB syscall buffer (the black hole itself)

Usage:
  python tma_black_hole_probe.py --mode plain          2>&1 | tee plain.log
  python tma_black_hole_probe.py --mode tma_cta        2>&1 | tee tma_cta.log
  python tma_black_hole_probe.py --mode tma_gluon_cta  2>&1 | tee tma_gluon.log
  python tma_black_hole_probe.py --mode tma_multicast  2>&1 | tee tma_mc.log

Compare the 'nvml Δ vs baseline' line across logs:
  tma_gluon_cta  ≈ plain          (safe — no syscall buffer)
  tma_multicast  ≈ plain + 3.7G   (the shared::cluster tax)
  tma_cta        depends on your branch; if ≈ tma_multicast, your branch
                 still emits shared::cluster for single-CTA descriptors
                 and you should switch to the tma_gluon_cta structure.
"""

import argparse
import os
import subprocess
import threading
import time

import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

TILE_M = 128
TILE_N = 128
TILE_BYTES = TILE_M * TILE_N * 2  # fp16


# --------------------------------------------------------------------------- #
# A device-side busy-wait so NVML / nvidia-smi (≈1 Hz) can sample memory
# while the kernel is still resident on the GPU.
# --------------------------------------------------------------------------- #
@triton.jit
def _spin_ns(ns_deadline_delta):
    tl.inline_asm_elementwise(
        asm=(
            "{\n"
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
# tma_cta : tl.make_tensor_descriptor, single-CTA launch.
# On current Triton main(eb5efe2) this lowers to
#   cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes
# (see https://github.com/triton-lang/triton/blob/main/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp#L1253,
# related https://github.com/triton-lang/triton/blob/f1d668bfc/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp#L1418
# shared::cta is chosen when neither multicast nor crossCTABarrier hold).
# --------------------------------------------------------------------------- #
@triton.jit
def tma_cta_kernel(src_ptr, dst_ptr, M, N,
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
# tma_multicast : Gluon, num_ctas=2, multicast=True.
# The `multicast=True` flag flips the lowering to
#   cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::... .multicast::cluster
# which is what pulls in __cuda_syscall_cp_async_bulk_tensor_2d_tile_unicast and
# forces the ~3.7 GiB driver buffer.
# --------------------------------------------------------------------------- #
_HAS_GLUON = True
try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
except ImportError:
    _HAS_GLUON = False


if _HAS_GLUON:

    # ------------------------------------------------------------------- #
    # Gluon single-CTA identity copy, multicast=False.
    # This is the explicit workaround: Gluon exposes the `multicast` flag
    # directly, so we bypass whatever default tl.make_tensor_descriptor
    # picks on the current branch and force shared::cta scope.
    # ------------------------------------------------------------------- #
    @gluon.jit
    def tma_gluon_cta_kernel(in_desc, out_desc,
                             TILE_M: gl.constexpr, TILE_N: gl.constexpr):
        pid_m = gl.program_id(0)
        pid_n = gl.program_id(1)
        off_m = pid_m * TILE_M
        off_n = pid_n * TILE_N

        smem = gl.allocate_shared_memory(in_desc.dtype, in_desc.block_shape,
                                         in_desc.layout)
        bar = mbarrier.allocate_mbarrier()
        mbarrier.init(bar, count=1)
        mbarrier.expect(bar, in_desc.nbytes_per_cta)
        # multicast=False is the default; spelled out here for clarity.
        tma.async_copy_global_to_shared(in_desc, [off_m, off_n], bar, smem,
                                        multicast=False)
        mbarrier.wait(bar, phase=0, deps=[smem])
        tma.async_copy_shared_to_global(out_desc, [off_m, off_n], smem)

    @gluon.jit
    def tma_multicast_kernel(in_desc, out_desc):
        gl.static_assert(gl.num_ctas() == 2)
        smem = gl.allocate_shared_memory(in_desc.dtype, in_desc.block_shape,
                                         in_desc.layout)
        bar = mbarrier.allocate_mbarrier()
        mbarrier.init(bar, count=1)
        mbarrier.expect(bar, in_desc.nbytes_per_cta)
        tma.async_copy_global_to_shared(in_desc, [0, 0], bar, smem,
                                        multicast=True)
        mbarrier.wait(bar, phase=0, deps=[smem])
        tma.async_copy_shared_to_global(out_desc, [0, 0], smem)


# --------------------------------------------------------------------------- #
# NVML / nvidia-smi memory probes (identical spirit to tma_black_hole.py).
# --------------------------------------------------------------------------- #
try:
    import pynvml as _pynvml  # type: ignore[import-not-found]
    HAS_NVML = True
except ImportError:
    _pynvml = None
    HAS_NVML = False


def smi_used_bytes(device_index=0):
    out = subprocess.check_output([
        "nvidia-smi", f"--id={device_index}",
        "--query-gpu=memory.used", "--format=csv,noheader,nounits",
    ]).decode().strip()
    return int(out) * 1024 * 1024


def smi_proc_mem(device_index=0):
    out = subprocess.check_output([
        "nvidia-smi", f"--id={device_index}",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]).decode().strip()
    result = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        pid, mib = [x.strip() for x in line.split(",")]
        result[int(pid)] = int(mib) * 1024 * 1024
    return result


def nvml_init(device_index):
    if HAS_NVML:
        _pynvml.nvmlInit()
        return _pynvml.nvmlDeviceGetHandleByIndex(device_index)
    return device_index


def nvml_used(handle):
    if HAS_NVML and handle is not None:
        return _pynvml.nvmlDeviceGetMemoryInfo(handle).used
    return smi_used_bytes(handle if isinstance(handle, int) else 0)


class DeviceMemorySampler(threading.Thread):
    """Poll NVML in the background to catch transient allocations."""

    def __init__(self, device_index=0, interval_s=0.002):
        super().__init__(daemon=True)
        if not HAS_NVML:
            interval_s = max(interval_s, 0.2)
        self._handle = nvml_init(device_index)
        self._interval = interval_s
        self._stop = threading.Event()
        self.samples = []

    def run(self):
        t0 = time.time()
        while not self._stop.is_set():
            self.samples.append((time.time() - t0, nvml_used(self._handle)))
            time.sleep(self._interval)

    def stop(self):
        self._stop.set()
        self.join()

    def peak(self):
        return max(s[1] for s in self.samples) if self.samples else 0


TMA_ALLOC_LOG = []


def make_logging_allocator():
    def alloc_fn(size, alignment, stream):
        TMA_ALLOC_LOG.append(size)
        print(f"  [alloc_fn] size={size/1024/1024:.2f} MiB "
              f"align={alignment} stream={stream}")
        return torch.empty(size, device="cuda", dtype=torch.int8)
    return alloc_fn


def fmt_gb(n):
    return f"{n / 1024**3:.3f} GiB"


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


def run_tma_cta(a, b, spin_ns=0):
    M, N = a.shape
    triton.set_allocator(make_logging_allocator())
    grid = (triton.cdiv(M, TILE_M), triton.cdiv(N, TILE_N))
    tma_cta_kernel[grid](a, b, M, N,
                         TILE_M=TILE_M, TILE_N=TILE_N, SPIN_NS=spin_ns)


def run_tma_gluon_cta(a, b, spin_ns=0):
    if not _HAS_GLUON:
        raise RuntimeError("Gluon not available in this Triton build")
    M, N = a.shape
    # Single-CTA layout: no cga_layout → defaults to 1 CTA.
    layout = gl.NVMMASharedLayout.get_default_for([TILE_M, TILE_N], gl.float16)
    in_desc = TensorDescriptor.from_tensor(a, [TILE_M, TILE_N], layout)
    out_desc = TensorDescriptor.from_tensor(b, [TILE_M, TILE_N], layout)
    grid = (triton.cdiv(M, TILE_M), triton.cdiv(N, TILE_N))
    tma_gluon_cta_kernel[grid](in_desc, out_desc,
                               TILE_M=TILE_M, TILE_N=TILE_N,
                               num_warps=4, num_ctas=1)
    # Gluon kernel doesn't take SPIN_NS; the syscall buffer (if any) is
    # already observable from the kernel-load step alone.


def run_tma_multicast(a, b, spin_ns=0):
    if not _HAS_GLUON:
        raise RuntimeError("Gluon not available in this Triton build")
    # One multicast group (cga_layout=[[0, 0]] means both CTAs see the full
    # tile). Launch exactly one cluster; the kernel body asserts num_ctas==2.
    layout = gl.NVMMASharedLayout.get_default_for(list(a.shape), gl.float16,
                                                  cga_layout=[[0, 0]])
    in_desc = TensorDescriptor.from_tensor(a, list(a.shape), layout)
    out_desc = TensorDescriptor.from_tensor(b, list(a.shape), layout)
    tma_multicast_kernel[(1,)](in_desc, out_desc, num_warps=4, num_ctas=2)


MODE_TABLE = {
    "plain": ("plain  (tl.load / tl.store)", run_plain),
    "tma_cta": ("TMA via tl.make_tensor_descriptor  (branch-default scope)",
                run_tma_cta),
    "tma_gluon_cta": ("TMA via Gluon, multicast=False  (forced shared::cta)",
                      run_tma_gluon_cta),
    "tma_multicast": ("TMA via Gluon, multicast=True  (shared::cluster → syscall)",
                      run_tma_multicast),
}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=list(MODE_TABLE.keys()),
                        default="tma_cta",
                        help="Which kernel variant to exercise. Run the "
                             "script once per mode in SEPARATE processes.")
    parser.add_argument("--shape", nargs=2, type=int, default=[TILE_M, TILE_N],
                        metavar=("M", "N"),
                        help="Source tensor shape. Must be multiples of "
                             f"{TILE_M}x{TILE_N}.")
    parser.add_argument("--spin-seconds", type=float, default=6.0,
                        help="Seconds the first launch busy-waits, so you "
                             "can eyeball nvidia-smi from another shell.")
    args = parser.parse_args()

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
          f"(Δ={fmt_gb(nvml_after_inputs - nvml_before)})")

    # --- first launch + spin so nvidia-smi can sample ---
    spin_ns = int(args.spin_seconds * 1e9)
    print(f"\n=== launching {label}, spinning ~{args.spin_seconds:.0f}s "
          "(check nvidia-smi in another shell) ===")
    gluon_modes = ("tma_gluon_cta", "tma_multicast")
    runner(a, b, spin_ns=0 if args.mode in gluon_modes else spin_ns)
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
          f"(Δ vs baseline={fmt_gb(nvml_after_first - nvml_before)})")

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
          f"(Δ vs baseline={fmt_gb(sampler.peak() - nvml_before)})")

    # --- summary ---
    print(f"\n=== summary ({label}) ===")
    print(f"alloc_fn called {len(TMA_ALLOC_LOG)} time(s), "
          f"total requested = {fmt_gb(sum(TMA_ALLOC_LOG))}  "
          f"(nonzero only for tma_cta, which uses a host-side workspace)")
    print(f"torch  alloc final: {fmt_gb(torch.cuda.memory_allocated())}")
    print(f"torch  reserv final: {fmt_gb(torch.cuda.memory_reserved())}")
    print(f"nvml   used  final: {fmt_gb(nvml_used(handle))}")
    print(f"smi    used  final: "
          f"{fmt_gb(smi_used_bytes(torch.cuda.current_device()))}")
    me = os.getpid()
    procs = smi_proc_mem(torch.cuda.current_device())
    print(f"smi this-process ({me}): {fmt_gb(procs.get(me, 0))}")
    print(f"nvml Δ vs baseline: "
          f"{fmt_gb(nvml_used(handle) - nvml_before)}")
    print("(Excess beyond torch reserved = CUDA-context / JIT / TMA tables / "
          "driver syscall buffer.)")

    # --- correctness check (identity copy) ---
    if args.mode in ("plain", "tma_cta", "tma_gluon_cta"):
        ok = torch.equal(a, b)
        print(f"\n{'OK' if ok else 'MISMATCH'}: identity copy "
              f"{'matches' if ok else 'does NOT match'} torch")


if __name__ == "__main__":
    main()
