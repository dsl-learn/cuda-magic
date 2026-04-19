"""
Load the patched cubin (shared::cta) via CUDA driver API and verify:
1. No 3.7 GiB memory spike (module load does not trigger lazy-alloc)
2. SASS contains UTMALDG.2D, not CALL.ABS.NOINC
"""

import ctypes
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from common.cuda_utils import resolve_cuobjdump
from common.mem_utils import nvml_init, nvml_used, fmt_gb

CUBIN = Path(__file__).parent / "cute_cache" / "fixed.cubin"
KERNEL_NAME = (
    "kernel_cutlass_kernel_cutedslcute_tma_copyTmaIdentityCopy_object_at__"
    "CopyAtom_ThrID10_TVLayoutSrc11638401_TVLayoutDst11638401_Valuetypebf16_"
    "tensor00o1110_CopyAtom_ThrID10_TVLayoutSrc11638_0"
)


def main():
    if not CUBIN.exists():
        print("Run verify_cta_fix.py first to generate fixed.cubin")
        sys.exit(1)

    dev = torch.cuda.current_device()
    handle = nvml_init(dev)

    # Baseline
    torch.cuda.synchronize()
    nvml_before = nvml_used(handle)
    print(f"[nvml] BEFORE module load: {fmt_gb(nvml_before)}")

    # Load cubin via driver API
    libcuda = ctypes.CDLL("libcuda.so.1")
    module = ctypes.c_void_p()
    with open(CUBIN, "rb") as f:
        cubin_bytes = f.read()

    err = libcuda.cuModuleLoadData(ctypes.byref(module), cubin_bytes)
    assert err == 0, f"cuModuleLoadData failed: {err}"

    nvml_after_load = nvml_used(handle)
    print(f"[nvml] AFTER  module load: {fmt_gb(nvml_after_load)} "
          f"(d={fmt_gb(nvml_after_load - nvml_before)})")

    # Resolve function
    func = ctypes.c_void_p()
    err = libcuda.cuModuleGetFunction(
        ctypes.byref(func), module, KERNEL_NAME.encode()
    )
    assert err == 0, f"cuModuleGetFunction failed: {err}"
    print(f"[driver] function resolved: {KERNEL_NAME[:60]}...")

    # SASS scan
    cuobjdump = resolve_cuobjdump()
    out = subprocess.check_output(
        [cuobjdump, "--dump-sass", str(CUBIN)],
        stderr=subprocess.STDOUT,
    ).decode("utf-8", errors="replace")

    utmal = sum(1 for ln in out.splitlines() if "UTMALDG.2D" in ln)
    calls = sum(1 for ln in out.splitlines() if "CALL.ABS.NOINC" in ln)
    print(f"[sass] UTMALDG.2D count: {utmal}")
    print(f"[sass] CALL.ABS.NOINC count: {calls}")

    # Conclusion
    delta = nvml_after_load - nvml_before
    if delta < 200 * 1024 * 1024 and calls == 0 and utmal > 0:  # < 200 MiB
        print("\n=== PASS: shared::cta patch fixes both leak and livelock root cause ===")
        print("  - No 3.7 GiB lazy-alloc at module load time")
        print("  - SASS uses native UTMALDG.2D, no driver syscall stubs")
    else:
        print("\n=== FAIL or INCONCLUSIVE ===")
        print(f"  delta={fmt_gb(delta)}, calls={calls}, utmal={utmal}")


if __name__ == "__main__":
    main()
