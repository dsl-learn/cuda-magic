# Usage:
#   # Run the TMA variant
#   python 5090_tma_mem_blackhole.py --kernel tma    2>&1 | tee tma.log
#
#   # Run the plain (non-TMA) variant — must be a FRESH process, otherwise
#   # the CUDA context / Triton runtime is already warm and the one-time
#   # memory cost we're trying to measure won't show up.
#   python 5090_tma_mem_blackhole.py --kernel plain  2>&1 | tee plain.log
#
# Then compare the "nvml Δ vs baseline" line between tma.log and plain.log
# to see how much of the extra device-memory usage is TMA-specific vs generic
# CUDA / Triton initialization.

import os
import subprocess
import threading
import time

import triton
import triton.language as tl
import torch

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _spin_ns(ns_deadline_delta):
    """Busy-wait for `ns_deadline_delta` nanoseconds by polling %globaltimer.
    Works on any sm_>=35 GPU; no 1ms-per-call cap like nanosleep."""
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


@triton.jit
def matmul_kernel_make_tensor_desciptor(a_ptr, b_ptr, c_ptr,  #
                                        M, N, K,  #
                                        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                                        BLOCK_SIZE_K: tl.constexpr,  #
                                        SLEEP_NS: tl.constexpr = 0,  #
                                        ):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for n in range(tl.cdiv(N, BLOCK_SIZE_N)):
        a = a_desc.load([pid_m * BLOCK_SIZE_M, n * BLOCK_SIZE_N])
        b = b_desc.load([n * BLOCK_SIZE_N, pid_k * BLOCK_SIZE_K])
        accumulator = tl.dot(a, b, acc=accumulator)

    accumulator = accumulator.to(tl.float16)
    c_desc.store([pid_m * BLOCK_SIZE_M, pid_k * BLOCK_SIZE_K], accumulator)

    # Keep the kernel resident so NVML / nvidia-smi (≈1 Hz refresh) can sample
    # device memory while TMA descriptor tables / cubin / context are live.
    if SLEEP_NS > 0:
        # Split into chunks — some drivers cap a single globaltimer delta.
        _spin_ns(SLEEP_NS)


# --------------------------------------------------------------------------- #
# Non-TMA baseline kernel — same shape/block sizes, plain tl.load / tl.store.
# Used to isolate whether the big NVML delta is TMA-specific or just generic
# Triton / CUDA context initialization.
# --------------------------------------------------------------------------- #


@triton.jit
def matmul_kernel_plain(a_ptr, b_ptr, c_ptr,  #
                        N, K,  #
                        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                        BLOCK_SIZE_K: tl.constexpr,  #
                        SLEEP_NS: tl.constexpr = 0,  #
                        ):
    # M is not used inside the kernel because shapes are assumed to be
    # multiples of the block sizes (no bounds masking needed).
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    # Shapes are assumed to be multiples of block sizes (true for the
    # M=1024, N=512, K=256 / BLOCK 128/64/64 setup below). Dropping masks
    # avoids the fp8 `other=0` cast issue that `tl.load(..., mask=, other=)`
    # triggers, since Triton can't currently cast int32 literal 0 to fp8e5.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for n in range(tl.cdiv(N, BLOCK_SIZE_N)):
        a_ptrs = a_ptr + (offs_m[:, None] * N + (n * BLOCK_SIZE_N + offs_n)[None, :])
        b_ptrs = b_ptr + ((n * BLOCK_SIZE_N + offs_n)[:, None] * K + offs_k[None, :])
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator = tl.dot(a, b, acc=accumulator)

    c_ptrs = c_ptr + (offs_m[:, None] * K + offs_k[None, :])
    tl.store(c_ptrs, accumulator.to(tl.float16))

    if SLEEP_NS > 0:
        _spin_ns(SLEEP_NS)


def matmul_plain(a, b, sleep_ns=0):
    M, N = a.shape
    N, K = b.shape
    c = torch.empty((M, K), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']),
                         triton.cdiv(K, META['BLOCK_SIZE_K']),)
    matmul_kernel_plain[grid](
        a, b, c,
        N, K,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_K=64,
        BLOCK_SIZE_N=64,
        SLEEP_NS=sleep_ns,
    )
    return c


# --------------------------------------------------------------------------- #
# Memory probes
# --------------------------------------------------------------------------- #

# Total bytes Triton's host-side TMA allocator was asked to provide.
TMA_ALLOC_LOG = []


def make_logging_allocator():
    """Wrap Triton's TMA workspace allocator so we can see every request."""

    def alloc_fn(size, alignment, stream):
        TMA_ALLOC_LOG.append(size)
        print(f"  [alloc_fn] size={size/1024/1024:.2f} MiB "
              f"align={alignment} stream={stream}")
        return torch.empty(size, device="cuda", dtype=torch.int8)

    return alloc_fn


# NVML is optional — fall back to shelling out to nvidia-smi if pynvml /
# nvidia-ml-py isn't installed. Both read the same underlying driver data.
try:
    import pynvml as _pynvml  # type: ignore[import-not-found]  # pynvml or nvidia-ml-py
    HAS_NVML = True
except ImportError:
    _pynvml = None
    HAS_NVML = False


def nvml_used_bytes(handle):
    if HAS_NVML and handle is not None:
        return _pynvml.nvmlDeviceGetMemoryInfo(handle).used
    # Fallback: ask nvidia-smi. `handle` is just the device index in this path.
    idx = handle if isinstance(handle, int) else 0
    return smi_used_bytes(idx)


def nvml_init_handle(device_index):
    """Return an NVML handle, or just the device index if NVML is unavailable."""
    if HAS_NVML:
        _pynvml.nvmlInit()
        return _pynvml.nvmlDeviceGetHandleByIndex(device_index)
    return device_index


def smi_used_bytes(device_index=0):
    """Shell out to nvidia-smi. Same data as NVML but lets you sanity-check
    that pynvml isn't lying, and shows per-process attribution via
    --query-compute-apps."""
    out = subprocess.check_output([
        "nvidia-smi",
        f"--id={device_index}",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]).decode().strip()
    return int(out) * 1024 * 1024  # MiB -> bytes


def smi_proc_mem(device_index=0):
    """Return {pid: used_bytes} for this device, same source as nvidia-smi."""
    out = subprocess.check_output([
        "nvidia-smi",
        f"--id={device_index}",
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


class DeviceMemorySampler(threading.Thread):
    """Poll real device memory via NVML in the background to catch transient
    allocations that happen while the kernel is launching / running.

    PyTorch's torch.cuda.memory_* only sees allocations that go through the
    caching allocator. Anything Triton allocates directly via cuMemAlloc (or
    via the CUDA context growing for JIT / TMA descriptor tables) is
    invisible to PyTorch — but shows up in nvidia-smi and NVML.
    """

    def __init__(self, device_index=0, interval_s=0.002):
        super().__init__(daemon=True)
        # If NVML isn't available we fall back to nvidia-smi; bump the
        # interval so we don't fork a subprocess every 2 ms.
        if not HAS_NVML:
            interval_s = max(interval_s, 0.2)
        self._handle = nvml_init_handle(device_index)
        self._interval = interval_s
        self._stop_event = threading.Event()
        self.samples = []  # list[(t, used_bytes)]

    def run(self):
        t0 = time.time()
        while not self._stop_event.is_set():
            self.samples.append((time.time() - t0, nvml_used_bytes(self._handle)))
            time.sleep(self._interval)

    def stop(self):
        self._stop_event.set()
        self.join()

    def peak(self):
        return max(s[1] for s in self.samples) if self.samples else 0

    def baseline(self):
        return self.samples[0][1] if self.samples else 0


def fmt_gb(n):
    return f"{n / 1024**3:.3f} GiB"


def snapshot(tag):
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    peak = torch.cuda.max_memory_allocated()
    print(f"[{tag}] torch alloc={fmt_gb(alloc)} reserved={fmt_gb(reserved)} "
          f"peak_alloc={fmt_gb(peak)}")


# --------------------------------------------------------------------------- #
# Kernel wrapper
# --------------------------------------------------------------------------- #

def matmul(a, b, sleep_ns=0):
    M, N = a.shape
    N, K = b.shape

    c = torch.empty((M, K), device=a.device, dtype=torch.float16)

    triton.set_allocator(make_logging_allocator())

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']),
                         triton.cdiv(K, META['BLOCK_SIZE_K']),)

    matmul_kernel_make_tensor_desciptor[grid](
        a, b, c,
        M, N, K,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_K=64,
        BLOCK_SIZE_N=64,
        SLEEP_NS=sleep_ns,
    )
    return c


# --------------------------------------------------------------------------- #
# Main: measure what the TMA path actually costs
# --------------------------------------------------------------------------- #

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=["tma", "plain"], default="tma",
                        help="Which kernel to benchmark. Run the script twice "
                             "(once with each) in SEPARATE processes to compare "
                             "— within a single process the second kernel "
                             "won't show the context-init cost.")
    parser.add_argument("--spin-seconds", type=float, default=8.0,
                        help="Seconds the first kernel busy-waits on device, "
                             "so you can eyeball nvidia-smi from another shell.")
    args = parser.parse_args()

    if args.kernel == "tma":
        kernel_fn = matmul
        label = "TMA (make_tensor_descriptor)"
    else:
        kernel_fn = matmul_plain
        label = "plain (tl.load / tl.store)"
    print(f"=== running kernel variant: {label} ===\n")

    M, N, K = 1024, 512, 256

    # --- 1. Baseline device memory (before anything Triton) ---
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    handle = nvml_init_handle(torch.cuda.current_device())
    source = "nvml" if HAS_NVML else "nvidia-smi (pynvml not installed)"
    nvml_before = nvml_used_bytes(handle)
    print(f"[{source}] device used BEFORE inputs: {fmt_gb(nvml_before)}")

    a = torch.randn((M, N), device=DEVICE, dtype=torch.float16).to(torch.float8_e5m2)
    b = torch.randn((N, K), device=DEVICE, dtype=torch.float16).to(torch.float8_e5m2)
    torch.cuda.synchronize()

    nvml_after_inputs = nvml_used_bytes(handle)
    snapshot("after inputs")
    print(f"[nvml] device used after inputs: {fmt_gb(nvml_after_inputs)} "
          f"(Δ={fmt_gb(nvml_after_inputs - nvml_before)})")

    # --- 1b. Launch a kernel that busy-waits, so you can eyeball nvidia-smi ---
    # While this runs (~8s), in another terminal do:
    #   watch -n 0.5 "nvidia-smi --query-compute-apps=pid,used_memory --format=csv"
    spin_ns = int(args.spin_seconds * 1e9)
    print(f"\n=== launching {label} that spins for ~{args.spin_seconds:.0f}s "
          f"(check nvidia-smi now) ===")
    _ = kernel_fn(a, b, sleep_ns=spin_ns)
    print(f"  right after launch (before sync): nvml used = "
          f"{fmt_gb(nvml_used_bytes(handle))}, "
          f"smi used = {fmt_gb(smi_used_bytes(torch.cuda.current_device()))}")
    # Poll from the host while the kernel is still resident.
    for i in range(int(args.spin_seconds)):
        time.sleep(1.0)
        print(f"  t+{i+1}s   nvml={fmt_gb(nvml_used_bytes(handle))}  "
              f"smi={fmt_gb(smi_used_bytes(torch.cuda.current_device()))}")
    torch.cuda.synchronize()
    print(f"  after sync: nvml used = {fmt_gb(nvml_used_bytes(handle))}")

    # --- 2. Run kernel WITH a background NVML sampler to catch transients ---
    print("\n=== first kernel launch (includes JIT compile) ===")
    sampler = DeviceMemorySampler(torch.cuda.current_device(), interval_s=0.002)
    sampler.start()
    c = kernel_fn(a, b)
    torch.cuda.synchronize()
    sampler.stop()
    snapshot("after 1st launch")
    print(f"[nvml] peak during 1st launch: {fmt_gb(sampler.peak())} "
          f"(Δ over baseline: {fmt_gb(sampler.peak() - nvml_before)})")
    print(f"[nvml] device used right after 1st launch: "
          f"{fmt_gb(nvml_used_bytes(handle))}")

    # --- 3. Repeat to separate one-time (JIT / context) cost from per-call ---
    print("\n=== 20 more launches (steady state) ===")
    sampler2 = DeviceMemorySampler(torch.cuda.current_device(), interval_s=0.002)
    sampler2.start()
    for _ in range(20):
        c = kernel_fn(a, b)
    torch.cuda.synchronize()
    sampler2.stop()
    snapshot("after 20 launches")
    print(f"[nvml] peak during 20 launches: {fmt_gb(sampler2.peak())}")
    print(f"[nvml] device used after 20 launches: "
          f"{fmt_gb(nvml_used_bytes(handle))}")

    # --- 4. Summary ---
    total_tma_req = sum(TMA_ALLOC_LOG)
    print(f"\n=== summary ({label}) ===")
    print(f"alloc_fn was called {len(TMA_ALLOC_LOG)} time(s), "
          f"total requested = {fmt_gb(total_tma_req)}  "
          f"(should be 0 for plain kernel)")
    print(f"torch.cuda.memory_allocated final: "
          f"{fmt_gb(torch.cuda.memory_allocated())}")
    print(f"torch.cuda.memory_reserved  final: "
          f"{fmt_gb(torch.cuda.memory_reserved())}")
    print(f"nvml used final: {fmt_gb(nvml_used_bytes(handle))}")
    print(f"smi  used final: {fmt_gb(smi_used_bytes(torch.cuda.current_device()))}"
          f"   (cross-check: should match nvml)")
    me = os.getpid()
    proc = smi_proc_mem(torch.cuda.current_device())
    print(f"smi  this-process ({me}): {fmt_gb(proc.get(me, 0))}")
    print(f"smi  all procs on device: "
          f"{ {p: fmt_gb(v) for p, v in proc.items()} }")
    print(f"nvml Δ vs baseline: "
          f"{fmt_gb(nvml_used_bytes(handle) - nvml_before)}")
    print("(If nvml Δ >> torch reserved, the excess is CUDA-context / JIT /"
          " TMA tables outside PyTorch's caching allocator.)")

    # --- 5. Correctness check (kept from original) ---
    torch_output = torch.matmul(a.to(torch.float16), b.to(torch.float16))
    ok = torch.allclose(c.float(), torch_output.float(), atol=0.125, rtol=0)
    print("\n✅ Triton and Torch match" if ok else "\n❌ Triton and Torch differ")


if __name__ == "__main__":
    main()
