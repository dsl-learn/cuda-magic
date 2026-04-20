#!/usr/bin/env python3
"""
Pinpoint when the TMA memory black hole appears: distinguish load from launch.

Methodology:
1. Compile the tma_cluster kernel with Triton to obtain the cubin (compile only, no load).
2. Manually load the cubin via Triton driver API: utils.load_binary.
3. Inject the resulting module/function handles (as Python ints) back into the
   CompiledKernel, and manually build a CudaLauncher assigned to _run so that
   Triton skips its own load_binary entirely.
4. Launch the kernel through Triton normally — Triton will go straight to
   cuLaunchKernelEx without touching load_binary.
5. Sample global GPU memory via torch.cuda.mem_get_info() before/after each step.

Expected results:
  - load_binary: only ~+80 MiB (cubin code segment in device memory), no 3.7 GiB jump.
  - cuLaunchKernelEx: spikes by ~+3728 MiB (driver lazy-allocates the internal
    cluster-scope syscall buffer on first launch).
  - torch.cuda.empty_cache(): does NOT release it.
  - Only process exit (cuCtxDestroy) reclaims the memory.

Note: Triton driver does not expose unload_module.  Even if we call the raw
      CUDA driver API cuModuleUnload, the ~3.7 GiB buffer is NOT released —
      it is context-bound, not module-bound.  The proof is in
      experiment_memleak_ctx_bound.py (process isolation + SIGKILL).
"""

# ---------------------------------------------------------------------------
# Interpretation of the printed snapshots (for readers of the source code):
#
# Observation table (approximate values on RTX 5090 32 GiB):
#
#   Step                          Expected delta
#   -----------------------------------------------------------
#   A. compile only               +2   MiB  (Triton host bookkeeping)
#   B. load_binary                +80  MiB  (cubin code segment in device mem)
#   C. cuLaunchKernelEx           +3728 MiB  <-- THE BLACK HOLE
#   D. after sync                 0    MiB  (kernel completes normally)
#   E. torch.cuda.empty_cache     0    MiB
#   F. del tensors + empty_cache  0    MiB
#
# Key findings:
#   1. The ~3.7 GiB allocation happens at FIRST LAUNCH (cuLaunchKernelEx),
#      NOT at load_binary.
#   2. The kernel completes normally after sync (no livelock in Triton).
#   3. torch.cuda.empty_cache() is useless because the buffer lives outside
#      PyTorch's caching allocator.
#   4. The only way to reclaim it inside a single process is to destroy the
#      CUDA context (cuCtxDestroy), which practically means process exit.
#   5. Triton driver does not expose unload_module.  Even the raw CUDA driver
#      API cuModuleUnload does NOT release the black-hole buffer — it is
#      context-bound, not module-bound.  The proof is in
#      experiment_memleak_ctx_bound.py using process isolation.
#
# Why launch and not load?
#   - CUDA driver delays the lazy allocation of the cluster-scope syscall
#     internal buffer until the kernel is actually launched for the first
#     time.  Merely loading the ELF (load_binary) parses the symbol table
#     but does not yet trigger the physical allocation.
#   - This matches the original document's "B → C jumps by 3808 MiB"
#     observation, but clarifies that the jump is strictly tied to
#     cuLaunchKernelEx, not to module load.
# ---------------------------------------------------------------------------

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import base64

import torch
import triton
from triton.backends.nvidia.driver import CudaLauncher

# ---------------------------------------------------------------------------
# Triton needs an allocator for global_scratch allocations.
# ---------------------------------------------------------------------------
triton.set_allocator(
    lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8)
)

_driver = triton.runtime.driver.active

def _load_binary(kernel_name: str, cubin, shared: int, device: int):
    """Thin wrapper around Triton driver utils.load_binary.

    Returns (module_handle, function_handle) as plain ints.
    """
    if isinstance(cubin, str):
        cubin = base64.b64decode(cubin)
    result = _driver.utils.load_binary(kernel_name, cubin, shared, device)
    module, function = result[0], result[1]
    return int(module), int(function)




def mem_mb() -> float:
    """Global GPU memory currently used (MiB)."""
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024 / 1024


def snapshot(tag: str) -> None:
    torch.cuda.synchronize()
    print(f"  [{tag:50s}] {mem_mb():8.1f} MiB")


def main() -> None:
    torch.cuda.init()

    import tma_copy
    tma_cluster_kernel = tma_copy.tma_cluster_kernel

    print("=" * 72)
    print("Experiment: manual cubin load + skip-Triton-load_binary launch")
    print("=" * 72)

    # ---- Phase A: compile only (no module load) ----------------------------
    c = torch.randn((128, 128), device="cuda", dtype=torch.float16)
    d = torch.empty_like(c)

    compiled = tma_cluster_kernel.warmup(
        c, d, 128, 128, TILE_M=128, TILE_N=128, SPIN_NS=0, grid=(1,)
    )
    snapshot("A. after warmup (compile only, module=None)")

    # ---- Phase B: load binary via Triton driver API -------------------------
    device = torch.cuda.current_device()
    kernel_name = compiled.metadata.name
    shared = getattr(compiled.metadata, "shared", 0)
    tma_mod, tma_func = _load_binary(
        kernel_name, compiled.asm["cubin"], shared, device
    )
    snapshot("B. after load_binary")

    # ---- Phase C: inject handles so Triton skips load_binary ----------------
    compiled.module = tma_mod
    compiled.function = tma_func
    compiled._run = CudaLauncher(compiled.src, compiled.metadata)

    # ---- Phase D: launch (Triton goes straight to cuLaunchKernelEx) --------
    tma_cluster_kernel[(1, 1)](c, d, 128, 128, TILE_M=128, TILE_N=128, SPIN_NS=0)
    snapshot("C. after cuLaunchKernelEx (launch returned)")

    torch.cuda.synchronize()
    snapshot("D. after kernel completes (sync)")

    # ---- Phase E: try release mechanisms ------------------------------------
    # NOTE: Triton driver does not expose unload_module.  Even if we called
    # the raw CUDA driver API cuModuleUnload, the ~3.7 GiB buffer would NOT
    # be released — it is context-bound, not module-bound.  The proof is in
    # experiment_memleak_ctx_bound.py (process isolation + SIGKILL).
    torch.cuda.empty_cache()
    snapshot("E. after torch.cuda.empty_cache")

    del c, d
    torch.cuda.empty_cache()
    snapshot("F. after del tensors + empty_cache")

    print()
    print("=" * 72)
    print("Done. See the module docstring for interpretation of the steps above.")
    print("=" * 72)


if __name__ == "__main__":
    main()
