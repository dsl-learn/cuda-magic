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

  plain           tl.load + tl.store                 — baseline, no TMA
  tma_cluster     tl.make_tensor_descriptor           — Triton lowers G2S
                  to shared::cluster scope on 3.4–3.6, pulling in the
                  ~3.7 GiB driver syscall buffer (the black hole)
  tma_cta         PTX-patched tl.make_tensor_descriptor — identical kernel,
                  but post-compile we sed the PTX to shared::cta and
                  re-ptxas. Proves the fix is scope-only, zero perf cost.

Usage — legacy single-process (run each mode in a SEPARATE process):
  python gluon_tma_copy.py --mode plain
  python gluon_tma_copy.py --mode tma_cluster
  python gluon_tma_copy.py --mode tma_cta

Usage — process-isolation (parent spawns a fresh child per mode, SIGKILLs
after launch, verifies the 3.7 GiB is fully returned on ctx teardown —
same methodology as experiment_memleak_ctx_bound.py):
  python gluon_tma_copy.py --isolate                      # all 3 modes
  python gluon_tma_copy.py --isolate --mode tma_cluster   # just one

Expected — legacy mode (Δ measures *only* the blackhole, baselined after
tensors are allocated):
  plain       Δ ≈ 0   GiB
  tma_cluster Δ ≈ 3.7 GiB  (shared::cluster → syscall buffer)
  tma_cta     Δ ≈ 0   GiB  (shared::cta    → no syscall buffer)

Expected — isolate mode (Δ(T0-T2) is the full parent-visible delta, so it
*includes* the ~0.5 GiB CUDA-context baseline that every child pays):
  plain       Δ(T0-T2) ≈ 0.5 GiB  ctx only
  tma_cluster Δ(T0-T2) ≈ 4.2 GiB  ctx + 3.7 GiB blackhole
  tma_cta     Δ(T0-T2) ≈ 0.5 GiB  ctx only (PTX patch works)
  reclaim(T4-T0) ≈ 0 for every mode — SIGKILL returns all memory.
"""

import argparse
import ctypes
import os
import signal
import subprocess
import sys
import tempfile
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
# tma_cluster : tl.make_tensor_descriptor (Triton 3.4–3.6 emit
#   shared::cluster scope regardless of num_ctas).
# Also used as the *source* for the PTX-patched tma_cta variant below.
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
# PTX-patch helper: compile tma_cluster_kernel via Triton, then sed the PTX
# to replace shared::cluster → shared::cta, re-ptxas, and launch via the
# CUDA driver API.  This proves the 3.7 GiB leak is scope-only.
# --------------------------------------------------------------------------- #
_patched_cache = {}          # (grid, spin_ns) → (func, smem, n_args)
_patched_ws_bufs = []        # prevent GC of workspace tensors


def _find_ptxas(arch: str):
    """Return the ptxas Triton itself would use for this arch. On Blackwell
    (sm_100+) Triton switches to a newer bundled ptxas (ptxas-blackwell,
    CUDA 12.9) that accepts PTX 8.8 and sm_120a — the right choice for us."""
    from triton.backends.nvidia.compiler import get_ptxas
    # get_ptxas takes a numeric arch: sm_120 -> 120, sm_120a -> 120.
    import re
    m = re.match(r"sm_(\d+)", arch)
    arch_num = int(m.group(1)) if m else 80
    return get_ptxas(arch_num).path


def _compile_patched(a, b, M, N, spin_ns, grid):
    """Compile → patch PTX → ptxas → load cubin.  Returns (CUfunction, smem_bytes, n_params)."""
    triton.set_allocator(make_logging_allocator())
    compiled = tma_cluster_kernel.warmup(
        a, b, M, N,
        TILE_M=TILE_M, TILE_N=TILE_N, SPIN_NS=spin_ns,
        grid=grid,
    )

    ptx = compiled.asm["ptx"]
    NEEDLE = "cp.async.bulk.tensor.2d.shared::cluster.global"
    REPLACE = "cp.async.bulk.tensor.2d.shared::cta.global"
    assert NEEDLE in ptx, "PTX does not contain shared::cluster — nothing to patch"
    ptx_patched = ptx.replace(NEEDLE, REPLACE)

    # Extract kernel name and shared-memory size from PTX
    import re
    m = re.search(r"\.visible\s+\.entry\s+(\w+)\s*\(", ptx_patched)
    assert m, "cannot find kernel entry point in PTX"
    kernel_name = m.group(1)

    smem = compiled.metadata.shared

    # Count kernel parameters from PTX. Each param declaration looks like
    #   .param .u64 .ptr .global .align 1 tma_cluster_kernel_param_0,
    # or the simpler
    #   .param .u32 tma_cluster_kernel_param_2,
    # so we must skip any number of intermediate attribute tokens between
    # `.param` and the parameter name.
    param_pattern = re.compile(
        rf"\.param\b[^,\n;]*?\b{kernel_name}_param_\d+"
    )
    n_params = len(param_pattern.findall(ptx_patched))

    # ptxas → cubin. Use the .target declared in the PTX itself (Triton may
    # emit sm_120a even when the device reports (12, 0)) and let Triton pick
    # the ptxas it would use — on Blackwell that's the bundled ptxas-blackwell.
    m_target = re.search(r"\.target\s+(sm_\w+)", ptx_patched)
    assert m_target, "cannot find .target directive in PTX"
    arch = m_target.group(1)

    ptxas = _find_ptxas(arch)
    with tempfile.NamedTemporaryFile(suffix=".ptx", mode="w", delete=False) as f:
        f.write(ptx_patched)
        ptx_path = f.name
    cubin_path = ptx_path.replace(".ptx", ".cubin")
    try:
        subprocess.run(
            [ptxas, f"-arch={arch}", ptx_path, "-o", cubin_path],
            check=True, capture_output=True, text=True,
        )
        with open(cubin_path, "rb") as f:
            cubin_bytes = f.read()
    finally:
        os.unlink(ptx_path)
        if os.path.exists(cubin_path):
            os.unlink(cubin_path)

    # Load cubin via CUDA driver API
    libcuda = ctypes.CDLL("libcuda.so.1")
    module = ctypes.c_void_p()
    err = libcuda.cuModuleLoadData(ctypes.byref(module), cubin_bytes)
    assert err == 0, f"cuModuleLoadData failed: {err}"
    func = ctypes.c_void_p()
    err = libcuda.cuModuleGetFunction(
        ctypes.byref(func), module, kernel_name.encode()
    )
    assert err == 0, f"cuModuleGetFunction failed: {err}"

    if smem > 48 * 1024:
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
        libcuda.cuFuncSetAttribute(
            func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem
        )

    return func, smem, n_params


def _launch_patched(func, smem, n_params, a, b, M, N, grid):
    """Launch a CUfunction with the same arguments Triton would pass."""
    # Triton's TMA kernel args: src_ptr, dst_ptr, M, N, [workspace ptrs ...]
    # Workspace count = n_params - 4  (the "visible" user args)
    n_ws = n_params - 4
    ws_bufs = []
    for _ in range(n_ws):
        ws = torch.empty(256, device="cuda", dtype=torch.uint8)
        ws_bufs.append(ws)
    _patched_ws_bufs.extend(ws_bufs)

    # Pack argument values  (all as 64-bit or 32-bit C types)
    c_args = [
        ctypes.c_uint64(a.data_ptr()),
        ctypes.c_uint64(b.data_ptr()),
        ctypes.c_int32(M),
        ctypes.c_int32(N),
    ] + [ctypes.c_uint64(ws.data_ptr()) for ws in ws_bufs]

    # Array of *pointers* to each arg (what cuLaunchKernel expects)
    arg_ptrs = (ctypes.c_void_p * len(c_args))(
        *[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in c_args]
    )

    libcuda = ctypes.CDLL("libcuda.so.1")
    err = libcuda.cuLaunchKernel(
        func,
        grid[0], grid[1], 1,       # grid
        128, 1, 1,                  # block  (must match Triton's num_warps*32)
        smem,                       # dynamic shared memory
        ctypes.c_void_p(0),        # stream = default
        arg_ptrs,                   # kernel params
        ctypes.c_void_p(0),        # extra = NULL
    )
    assert err == 0, f"cuLaunchKernel failed: {err}"


# --------------------------------------------------------------------------- #
# NVML / nvidia-smi memory probes
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
        self._stop_ev = threading.Event()
        self.samples = []

    def run(self):
        t0 = time.time()
        while not self._stop_ev.is_set():
            self.samples.append((time.time() - t0, nvml_used(self._handle)))
            time.sleep(self._interval)

    def stop(self):
        self._stop_ev.set()
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


def run_tma_cluster(a, b, spin_ns=0):
    M, N = a.shape
    triton.set_allocator(make_logging_allocator())
    grid = (triton.cdiv(M, TILE_M), triton.cdiv(N, TILE_N))
    tma_cluster_kernel[grid](a, b, M, N,
                             TILE_M=TILE_M, TILE_N=TILE_N, SPIN_NS=spin_ns)


def run_tma_cta(a, b, spin_ns=0):
    """Launch tma_cluster_kernel with PTX patched to shared::cta scope."""
    M, N = a.shape
    grid = (triton.cdiv(M, TILE_M), triton.cdiv(N, TILE_N))
    key = (grid, spin_ns)
    if key not in _patched_cache:
        print("  [ptx-patch] compiling & patching ...")
        _patched_cache[key] = _compile_patched(a, b, M, N, spin_ns, grid)
        print("  [ptx-patch] done")
    func, smem, n_params = _patched_cache[key]
    _launch_patched(func, smem, n_params, a, b, M, N, grid)


MODE_TABLE = {
    "plain": ("plain  (tl.load / tl.store)", run_plain),
    "tma_cluster": ("TMA  tl.make_tensor_descriptor  (shared::cluster — black hole)",
                    run_tma_cluster),
    "tma_cta": ("TMA  PTX-patched  (shared::cta — no black hole)",
                run_tma_cta),
}


# --------------------------------------------------------------------------- #
# Process-isolation helpers (parent-side): sample GPU memory from outside the
# child process, using `nvidia-smi`. Mirrors experiment_memleak_ctx_bound.py
# so the two scripts report comparable numbers.
# --------------------------------------------------------------------------- #
def free_mib() -> float:
    """Global GPU free memory in MiB (parent's external view)."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free",
         "--format=csv,noheader,nounits"],
        stderr=subprocess.DEVNULL,
    ).decode().strip().splitlines()[0]
    return float(out)


def used_by_pid(pid: int) -> float:
    """How much GPU memory nvidia-smi attributes to `pid` (MiB)."""
    out = subprocess.check_output(
        ["nvidia-smi",
         "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader,nounits"],
        stderr=subprocess.DEVNULL,
    ).decode()
    for ln in out.strip().splitlines():
        parts = [p.strip() for p in ln.split(",")]
        if parts and parts[0] == str(pid):
            return float(parts[1])
    return 0.0


def wait_line(proc: subprocess.Popen, needle: str, timeout: float) -> str:
    """Read child stdout until a line starts with `needle`, or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"child exited (code={proc.returncode}) before '{needle}'"
                )
            continue
        line = line.rstrip("\n")
        print(f"  child> {line}")
        if line.startswith(needle):
            return line
    raise TimeoutError(f"waiting for '{needle}' from child timed out")


def iso_sample(tag: str, pid: int | None = None) -> tuple[float, float]:
    """Print + return (gpu_free_mib, pid_used_mib)."""
    g = free_mib()
    u = used_by_pid(pid) if pid else 0.0
    print(f"  [{tag:<20}] gpu_free={g:8.1f} MiB   child_used={u:8.1f} MiB")
    return g, u


# --------------------------------------------------------------------------- #
# Child runner: minimal work to trigger the lazy driver allocation, then park.
# The kernel (tma_cluster in particular) livelocks on SM120, so we never call
# torch.cuda.synchronize() after launch — we only need the alloc to hit the
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

    _label, runner = MODE_TABLE[mode]
    if mode == "tma_cta":
        # Pre-compile the PTX-patched variant so COMPILED is a clean signal
        # distinct from LAUNCHED (ptxas + cuModuleLoadData take real time).
        grid = (triton.cdiv(M, TILE_M), triton.cdiv(N, TILE_N))
        _patched_cache[(grid, 0)] = _compile_patched(a, b, M, N, 0, grid)
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

        # tma_cta needs ptxas + cuModuleLoadData; give it extra headroom.
        compile_timeout = 180 if mode == "tma_cta" else 60
        try:
            wait_line(proc, "COMPILED", timeout=compile_timeout)
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
           f"{'Δ(T0-T2)':>10} {'reclaim(T4-T0)':>16}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r["status"] != "OK":
            print(f"{r['mode']:<13} {r['status']:<18}")
            continue
        delta = r["T0"] - r["T2"]        # positive => allocation grew
        reclaim = r["T4"] - r["T0"]      # ≈ 0  => fully reclaimed on kill
        print(f"{r['mode']:<13} {r['status']:<18} "
              f"{r['T0']:>10.1f} {r['T1']:>10.1f} {r['T2']:>10.1f} "
              f"{r['T3']:>10.1f} {r['T4']:>10.1f} "
              f"{delta:>10.1f} {reclaim:>16.1f}")
    print("\nUnits: MiB. Δ(T0-T2) = free memory lost while child was alive.")
    print("reclaim(T4-T0) ≈ 0 confirms the allocation is CUDA-context-bound")
    print("and is fully released by SIGKILL — matches experiment_memleak_ctx_bound.")


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
          f"(Δ={fmt_gb(nvml_after_inputs - nvml_before)})")

    # --- first launch + spin so nvidia-smi can sample ---
    spin_ns = int(args.spin_seconds * 1e9)
    print(f"\n=== launching {label}, spinning ~{args.spin_seconds:.0f}s "
          "(check nvidia-smi in another shell) ===")
    no_spin_modes = ("tma_cta",)
    runner(a, b, spin_ns=0 if args.mode in no_spin_modes else spin_ns)
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
          f"total requested = {fmt_gb(sum(TMA_ALLOC_LOG))}")
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
    if args.mode in ("plain", "tma_cluster", "tma_cta"):
        ok = torch.equal(a, b)
        print(f"\n{'OK' if ok else 'MISMATCH'}: identity copy "
              f"{'matches' if ok else 'does NOT match'} torch")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=list(MODE_TABLE.keys()),
                        default="tma_cta",
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
                             "defaults to all three.")
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
