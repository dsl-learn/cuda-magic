"""
Experiment 2: subprocess-isolated lifetime test.

Since the kernel livelocks on SM120 and never completes, we cannot observe
"after kernel completion" in the same process. Instead, we isolate the leak
inside a child process and watch the parent's view of GPU free memory across
the child's lifecycle.

Timeline:
  T0  baseline (no child)
  T1  child spawned, ctx built, before any launch
  T2  child has launched the TMA kernel (blackhole allocated)
  T3  parent SIGKILLs child
  T4  2s post-kill

The key question:  does T4's free memory return to T0?
  - yes -> the 3.7 GiB is strictly bound to the CUDA context; only process
           teardown releases it. Nothing an in-process API can do.
  - no  -> worse: driver-level leak surviving ctx destruction.

Child prints its PID and a sentinel "READY_FOR_LAUNCH" / "LAUNCHED" so the
parent knows when to sample.
"""

import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.mem_utils import free_mib, used_by_pid
from common.proc_utils import wait_line, iso_sample

HERE = Path(__file__).parent.parent  # repo root
CHILD_SCRIPT = Path(__file__).parent / "_leaky_child.py"  # put child script inside cutedsl/


def write_child():
    CHILD_SCRIPT.write_text(r'''
import os, sys, time
from pathlib import Path

# _leaky_child.py lives in cutedsl/, so repo root is its grandparent.
repo = Path(__file__).parent.parent
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))

import torch

CACHE_ROOT = repo / "cutedsl" / "cute_cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CUTE_DSL_CACHE_DIR", str(CACHE_ROOT / "mlir_cache"))
os.environ.setdefault("CUTE_DSL_DUMP_DIR", str(CACHE_ROOT))
os.environ.setdefault("CUTE_DSL_KEEP_CUBIN", "1")
if "CUTE_DSL_ARCH" not in os.environ:
    mj, mn = torch.cuda.get_device_capability(0)
    os.environ["CUTE_DSL_ARCH"] = f"sm_{mj}{mn}"

torch.cuda.init()
_ = torch.empty(1, device="cuda"); del _
torch.cuda.synchronize()
print(f"CHILD_PID={os.getpid()}", flush=True)
print("CTX_READY", flush=True)

from cutedsl.cute_tma_copy import TmaIdentityCopy
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

M, N = 1024, 2048
A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
B = torch.empty_like(A)
a_ct = from_dlpack(A).mark_layout_dynamic(leading_dim=1)
b_ct = from_dlpack(B).mark_layout_dynamic(leading_dim=1)
compiled = cute.compile(TmaIdentityCopy(), a_ct, b_ct)
print("COMPILED", flush=True)

# Fire-and-forget: we know the kernel livelocks on SM120. We only need the
# allocation to trigger, which happens at launch-queue time.
compiled(a_ct, b_ct)
print("LAUNCHED", flush=True)

# Park. Parent will SIGKILL us when it's done sampling.
while True:
    time.sleep(3600)
''')


def sample(tag: str, pid: int | None = None) -> None:
    gpu_free = free_mib()
    pid_used = used_by_pid(pid) if pid else 0.0
    print(f"  [{tag:<20}] gpu_free={gpu_free:8.1f} MiB   child_used={pid_used:8.1f} MiB")


def main() -> None:
    write_child()

    print("=== T0: baseline (no child) ===")
    sample("T0.baseline")

    proc = subprocess.Popen(
        [sys.executable, "-u", str(CHILD_SCRIPT)],
        cwd=str(HERE),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        print("\n=== T1: child spawned, ctx ready ===")
        wait_line(proc, "CTX_READY", timeout=60)
        # Extract PID from earlier line — re-parse from smi instead.
        pid = proc.pid
        time.sleep(0.5)
        sample("T1.ctx_ready", pid)

        print("\n=== waiting for COMPILED + LAUNCHED ===")
        wait_line(proc, "COMPILED", timeout=120)
        wait_line(proc, "LAUNCHED", timeout=60)
        # Give driver a beat to reflect the allocation in nvidia-smi.
        time.sleep(2.0)

        print("\n=== T2: post-launch (blackhole expected) ===")
        sample("T2.post_launch", pid)

        print("\n=== T3: SIGKILL child ===")
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)
        sample("T3.killed", pid)

        print("\n=== T4: 2s post-kill ===")
        time.sleep(2.0)
        sample("T4.post_kill", pid)

        print("\n=== 5s post-kill (final) ===")
        time.sleep(3.0)
        sample("T4+3s.final", pid)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGKILL)
            proc.wait(timeout=5)

    print("\nInterpretation:")
    print("  T4.final vs T0.baseline  == same  -> leak is context-bound,")
    print("                                         only process death frees it.")
    print("  T4.final vs T0.baseline  much less -> driver-level leak beyond ctx.")


if __name__ == "__main__":
    main()
