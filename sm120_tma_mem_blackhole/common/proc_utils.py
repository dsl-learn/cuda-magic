"""Subprocess experiment helpers: wait_line, iso_sample, etc."""

import subprocess
import time

from .mem_utils import free_mib, used_by_pid


def wait_line(proc: subprocess.Popen, needle: str, timeout: float) -> str:
    """Read child stdout until a line starts with needle, or timeout / child exit."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                raise RuntimeError(f"child died early before '{needle}'")
            continue
        line = line.strip()
        print(f"  child> {line}")
        if line.startswith(needle):
            return line
    raise TimeoutError(f"waiting for '{needle}' timed out")


def iso_sample(tag: str, pid: int | None = None) -> tuple[float, float]:
    """Isolation snapshot: print and return (gpu_free_mib, pid_used_mib)."""
    g = free_mib()
    u = used_by_pid(pid) if pid else 0.0
    print(f"  [{tag:<20}] gpu_free={g:8.1f} MiB   child_used={u:8.1f} MiB")
    return g, u
