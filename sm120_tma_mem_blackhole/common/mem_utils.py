"""Memory probing utilities: torch allocator + nvidia-smi / NVML snapshots."""

import os
import subprocess
import threading
import time
from pathlib import Path

import torch


MiB = 1024 * 1024


def snapshot(tag: str) -> None:
    """Print driver / torch memory state."""
    free, total = torch.cuda.mem_get_info()
    driver_used = (total - free) / MiB
    torch_alloc = torch.cuda.memory_allocated() / MiB
    torch_resv = torch.cuda.memory_reserved() / MiB
    print(
        f"[{tag:<24}] "
        f"free={free/MiB:8.1f} MiB  "
        f"driver_used={driver_used:8.1f} MiB  "
        f"torch_alloc={torch_alloc:7.1f} MiB  "
        f"torch_reserved={torch_resv:7.1f} MiB"
    )


def smi_used_mb() -> float:
    """GPU memory attributed to the current process by nvidia-smi (MiB)."""
    pid = os.getpid()
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode()
    except Exception:
        return float("nan")
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if parts and parts[0] == str(pid):
            return float(parts[1])
    return 0.0


def free_mib() -> float:
    """Global GPU free memory (MiB)."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free",
         "--format=csv,noheader,nounits"],
        stderr=subprocess.DEVNULL,
    ).decode().strip().splitlines()[0]
    return float(out)


def used_by_pid(pid: int) -> float:
    """GPU memory attributed to a specific PID by nvidia-smi (MiB)."""
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


def fmt_gb(n: int | float) -> str:
    return f"{n / 1024**3:.3f} GiB"


def smi_used_bytes(device_index: int = 0) -> int:
    """Global GPU used memory from nvidia-smi (bytes)."""
    out = subprocess.check_output([
        "nvidia-smi", f"--id={device_index}",
        "--query-gpu=memory.used", "--format=csv,noheader,nounits",
    ]).decode().strip()
    return int(out) * MiB


def smi_proc_mem(device_index: int = 0) -> dict[int, int]:
    """Per-process GPU memory usage from nvidia-smi, returns {pid: bytes}."""
    out = subprocess.check_output([
        "nvidia-smi", f"--id={device_index}",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]).decode().strip()
    result: dict[int, int] = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        pid, mib = [x.strip() for x in line.split(",")]
        result[int(pid)] = int(mib) * MiB
    return result


# ---- NVML (optional) ----
try:
    import pynvml as _pynvml  # type: ignore[import-not-found]
    HAS_NVML = True
except ImportError:
    _pynvml = None
    HAS_NVML = False


def nvml_init(device_index: int = 0):
    if HAS_NVML:
        _pynvml.nvmlInit()
        return _pynvml.nvmlDeviceGetHandleByIndex(device_index)
    return device_index


def nvml_used(handle):
    if HAS_NVML and handle is not None and not isinstance(handle, int):
        return _pynvml.nvmlDeviceGetMemoryInfo(handle).used
    return smi_used_bytes(handle if isinstance(handle, int) else 0)


class DeviceMemorySampler(threading.Thread):
    """Poll NVML / nvidia-smi in the background to capture peak memory."""

    def __init__(self, device_index: int = 0, interval_s: float = 0.002):
        super().__init__(daemon=True)
        if not HAS_NVML:
            interval_s = max(interval_s, 0.2)
        self._handle = nvml_init(device_index)
        self._interval = interval_s
        self._stop_ev = threading.Event()
        self.samples: list[tuple[float, int]] = []

    def run(self):
        t0 = time.time()
        while not self._stop_ev.is_set():
            self.samples.append((time.time() - t0, nvml_used(self._handle)))
            time.sleep(self._interval)

    def stop(self):
        self._stop_ev.set()
        self.join()

    def peak(self) -> int:
        return max(s[1] for s in self.samples) if self.samples else 0
