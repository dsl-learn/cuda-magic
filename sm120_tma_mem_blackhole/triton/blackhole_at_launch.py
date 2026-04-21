"""
Pinpoint the TMA black hole: compile -> load -> launch step-by-step.

Proves the ~3.7 GiB driver buffer is lazy-allocated at cuLaunchKernelEx,
not at module load, and cannot be released by empty_cache.

Usage:
  python sm120_tma_mem_blackhole/triton/blackhole_at_launch.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import base64
import torch
import triton
from triton.backends.nvidia.driver import CudaLauncher

from common.mem_utils import fmt_gb, nvml_init, nvml_used

# Triton needs an allocator for global_scratch allocations.
triton.set_allocator(
    lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8)
)

_nvml_handle = nvml_init(torch.cuda.current_device())

_driver = triton.runtime.driver.active


def _load_binary(kernel_name: str, cubin, shared: int, device: int):
    """Manually load cubin via Triton driver API. Returns (module, function) ints."""
    if isinstance(cubin, str):
        cubin = base64.b64decode(cubin)
    result = _driver.utils.load_binary(kernel_name, cubin, shared, device)
    return int(result[0]), int(result[1])


def snapshot(tag: str) -> None:
    torch.cuda.synchronize()
    used_mib = nvml_used(_nvml_handle) / (1024 * 1024)
    print(f"  [{tag:<50s}] {used_mib:>8.1f} MiB")


def main() -> None:
    torch.cuda.init()

    # Import the kernel from the sibling tma_copy module.
    import tma_copy
    kernel = tma_copy.tma_cluster_kernel

    c = torch.randn((128, 128), device="cuda", dtype=torch.float16)
    d = torch.empty_like(c)

    print("=" * 72)
    print("Step-by-step: compile -> load_binary -> launch -> sync -> empty_cache")
    print("=" * 72)

    # A. Compile only (no module load)
    compiled = kernel.warmup(c, d, 128, 128, TILE_M=128, TILE_N=128, grid=(1,))
    snapshot("A. after warmup (compile only, module=None)")

    # B. Manually load cubin
    device = torch.cuda.current_device()
    mod, func = _load_binary(
        compiled.metadata.name, compiled.asm["cubin"],
        getattr(compiled.metadata, "shared", 0), device,
    )
    snapshot("B. after load_binary")

    # C. Inject handles so Triton skips its own load_binary
    compiled.module = mod
    compiled.function = func
    compiled._run = CudaLauncher(compiled.src, compiled.metadata)

    # D. Launch (goes straight to cuLaunchKernelEx)
    kernel[(1, 1)](c, d, 128, 128, TILE_M=128, TILE_N=128)
    snapshot("C. after cuLaunchKernelEx (launch returned)")

    torch.cuda.synchronize()
    snapshot("D. after kernel completes (sync)")

    # E. Try release mechanisms
    torch.cuda.empty_cache()
    snapshot("E. after torch.cuda.empty_cache")

    del c, d
    torch.cuda.empty_cache()
    snapshot("F. after del tensors + empty_cache")

    print()
    print("Key finding: the +~3.7 GiB jump happens at step C (launch), not B (load).")
    print("The buffer is context-bound; only process exit can reclaim it.")


if __name__ == "__main__":
    main()
