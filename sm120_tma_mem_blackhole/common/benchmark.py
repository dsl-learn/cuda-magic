"""Shared benchmark harness for the TMA black-hole reproducer."""

import torch

from common.cuda_utils import warmup_cuda_context
from common.mem_utils import (
    fmt_gb, nvml_init, nvml_used, HAS_NVML,
)


def run_blackhole_test(runner, label: str,
                       shape: tuple[int, int] = (128, 128),
                       dtype=torch.float16) -> None:
    """
    Standard black-hole measurement:
      1. warm-up CUDA context
      2. NVML baseline
      3. allocate input / output tensors
      4. run ``runner(a, b)``
      5. sync + read NVML again
      6. print deltas
    """
    warmup_cuda_context()

    handle = nvml_init(torch.cuda.current_device())
    source = "nvml" if HAS_NVML else "nvidia-smi (pynvml not installed)"

    nvml_before = nvml_used(handle)
    print(f"[{label}][{source}] BEFORE any alloc: {fmt_gb(nvml_before)}")

    a = torch.randn(shape, device="cuda", dtype=dtype)
    b = torch.empty_like(a)
    torch.cuda.synchronize()

    nvml_after_inputs = nvml_used(handle)
    print(f"[{label}][nvml] after inputs: {fmt_gb(nvml_after_inputs)} "
          f"(d={fmt_gb(nvml_after_inputs - nvml_before)})")

    print(f"\n=== launching {label} TMA cluster kernel ===")
    runner(a, b)
    torch.cuda.synchronize()

    nvml_after = nvml_used(handle)
    print(f"[{label}][nvml] after launch: {fmt_gb(nvml_after)} "
          f"(d vs baseline={fmt_gb(nvml_after - nvml_before)})")
    print(f"\n[{label}] Black hole jump (launch - inputs): "
          f"{fmt_gb(nvml_after - nvml_after_inputs)}")

    ok = torch.equal(a, b)
    print(f"{'OK' if ok else 'MISMATCH'}: identity copy "
          f"{'matches' if ok else 'does NOT match'} torch")
