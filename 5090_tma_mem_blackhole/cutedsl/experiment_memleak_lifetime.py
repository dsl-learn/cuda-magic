"""
Experiment: does the driver-side TMA cluster-scope allocation get released
while the process is still alive?

Hypothesis: the ~3.7 GiB buffer is bound to the CUDA context, not to the
module/function/kernel launch, so nothing short of tearing down the context
(= exiting the process) will return it.

We take memory snapshots at 5 stages with three orthogonal views:
  1. torch.cuda.mem_get_info()        — driver free/total (ground truth)
  2. nvidia-smi pid=self used_memory  — driver view of THIS process
  3. torch.cuda.memory_allocated()    — PyTorch caching allocator only

Expected shape if the hypothesis holds:
  A (ctx only)   : free ~ baseline
  B (compiled)   : free drops ~3.7 GiB    <-- leak appears here
  C (launched)   : no further drop
  D (del+gc+empty_cache) : **free unchanged** (the tell)
  E (ipc_collect): still unchanged
"""

import gc
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from common.mem_utils import snapshot, smi_used_mb


def main():
    assert torch.cuda.is_available()
    torch.cuda.init()
    # Force ctx creation with a tiny alloc so baseline is real.
    _ = torch.empty(1, device="cuda"); del _
    torch.cuda.synchronize()
    snapshot("A. ctx only")

    # --- Lazy import so ctx/baseline reading isn't polluted by CUTE init. ----
    CACHE_ROOT = Path(__file__).parent / "cute_cache"
    CACHE_ROOT.mkdir(exist_ok=True)
    os.environ.setdefault("CUTE_DSL_CACHE_DIR", str(CACHE_ROOT / "mlir_cache"))
    os.environ.setdefault("CUTE_DSL_DUMP_DIR", str(CACHE_ROOT))
    os.environ.setdefault("CUTE_DSL_KEEP_CUBIN", "1")
    os.environ.setdefault("CUTE_DSL_KEEP_PTX", "1")
    if "CUTE_DSL_ARCH" not in os.environ:
        mj, mn = torch.cuda.get_device_capability(0)
        os.environ["CUTE_DSL_ARCH"] = f"sm_{mj}{mn}"

    from cutedsl.cute_tma_copy import TmaIdentityCopy
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    M, N = 1024, 2048
    A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    B = torch.empty_like(A)
    a_ct = from_dlpack(A).mark_layout_dynamic(leading_dim=1)
    b_ct = from_dlpack(B).mark_layout_dynamic(leading_dim=1)

    copy = TmaIdentityCopy()
    compiled = cute.compile(copy, a_ct, b_ct)
    torch.cuda.synchronize()
    snapshot("B. after compile")

    # --- Phase C: split launch vs. synchronize, with a background poller
    # sampling memory every 0.5s. This reveals whether (a) driver is slowly
    # allocating the 3.7 GiB blackhole during launch/load, or (b) the kernel
    # itself is livelocked (allocation would spike once, then flatline).
    stop_evt = threading.Event()

    def _poll():
        i = 0
        while not stop_evt.is_set():
            snapshot(f"   C.poll[{i:02d}]")
            i += 1
            stop_evt.wait(0.5)

    poller = threading.Thread(target=_poll, daemon=True)
    poller.start()

    t0 = time.time()
    print(">>> launching compiled()")
    compiled(a_ct, b_ct)
    print(f">>> compiled() returned in {time.time()-t0:.2f}s; calling synchronize()")
    t1 = time.time()
    torch.cuda.synchronize()
    print(f">>> synchronize() returned in {time.time()-t1:.2f}s")

    stop_evt.set()
    poller.join()
    snapshot("C. after launch")

    del compiled, copy, a_ct, b_ct, A, B
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    snapshot("D. del+gc+empty_cache")

    torch.cuda.ipc_collect()
    torch.cuda.synchronize()
    snapshot("E. ipc_collect")

    print("\nProcess still alive. The diff (E - A) is the leak bound to this CUDA context.")
    print("Exit the process to confirm the driver returns it.")


if __name__ == "__main__":
    main()
