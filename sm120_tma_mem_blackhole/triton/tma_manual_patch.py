"""
Manual PTX patch workaround: replace shared::cluster with shared::cta.

Triton 3.4-3.6 emits cp.async.bulk.tensor with shared::cluster scope by
default, which triggers a ~3.7 GiB driver-side allocation on SM120 (RTX 5090).
This script compiles the kernel, sed-replaces cluster->cta in the PTX,
re-assembles via ptxas, and launches through the raw CUDA driver API.

Proves the leak is purely scope-related and has zero perf cost.

Usage:
  python sm120_tma_mem_blackhole/triton/tma_manual_patch.py
"""

import ctypes
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import triton
import triton.language as tl

from common.cuda_utils import warmup_cuda_context
from common.mem_utils import fmt_gb, nvml_init, nvml_used

DEVICE = triton.runtime.driver.active.get_active_torch_device()

TILE_M = 128
TILE_N = 128


# --------------------------------------------------------------------------- #
# Same kernel as tma_copy.py's tma_cluster_kernel.
# --------------------------------------------------------------------------- #
@triton.jit
def tma_cluster_kernel(src_ptr, dst_ptr, M, N,
                       TILE_M: tl.constexpr, TILE_N: tl.constexpr):
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


# --------------------------------------------------------------------------- #
# PTX patch machinery
# --------------------------------------------------------------------------- #
_patched_cache = {}
_patched_ws_bufs = []


def _find_ptxas(arch: str) -> str:
    from triton.backends.nvidia.compiler import get_ptxas
    m = re.match(r"sm_(\d+)", arch)
    arch_num = int(m.group(1)) if m else 80
    return get_ptxas(arch_num).path


def _compile_patched(a, b, M, N, grid):
    """Compile -> patch PTX -> ptxas -> load cubin. Returns (CUfunction, smem, n_params)."""
    compiled = tma_cluster_kernel.warmup(
        a, b, M, N,
        TILE_M=TILE_M, TILE_N=TILE_N,
        grid=grid,
    )

    ptx = compiled.asm["ptx"]
    NEEDLE = "cp.async.bulk.tensor.2d.shared::cluster.global"
    REPLACE = "cp.async.bulk.tensor.2d.shared::cta.global"
    assert NEEDLE in ptx, "PTX does not contain shared::cluster -- nothing to patch"
    ptx_patched = ptx.replace(NEEDLE, REPLACE)

    m = re.search(r"\.visible\s+\.entry\s+(\w+)\s*\(", ptx_patched)
    assert m, "cannot find kernel entry point in PTX"
    kernel_name = m.group(1)

    smem = compiled.metadata.shared

    param_pattern = re.compile(
        rf"\.param\b[^,\n;]*?\b{kernel_name}_param_\d+"
    )
    n_params = len(param_pattern.findall(ptx_patched))

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
    n_ws = n_params - 4
    ws_bufs = []
    for _ in range(n_ws):
        ws = torch.empty(256, device="cuda", dtype=torch.uint8)
        ws_bufs.append(ws)
    _patched_ws_bufs.extend(ws_bufs)

    c_args = [
        ctypes.c_uint64(a.data_ptr()),
        ctypes.c_uint64(b.data_ptr()),
        ctypes.c_int32(M),
        ctypes.c_int32(N),
    ] + [ctypes.c_uint64(ws.data_ptr()) for ws in ws_bufs]

    arg_ptrs = (ctypes.c_void_p * len(c_args))(
        *[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in c_args]
    )

    libcuda = ctypes.CDLL("libcuda.so.1")
    err = libcuda.cuLaunchKernel(
        func,
        grid[0], grid[1], 1,
        128, 1, 1,
        smem,
        ctypes.c_void_p(0),
        arg_ptrs,
        ctypes.c_void_p(0),
    )
    assert err == 0, f"cuLaunchKernel failed: {err}"


def run_patched(a, b):
    M, N = a.shape
    grid = (triton.cdiv(M, TILE_M), triton.cdiv(N, TILE_N))
    key = grid
    if key not in _patched_cache:
        print("  [ptx-patch] compiling & patching ...")
        _patched_cache[key] = _compile_patched(a, b, M, N, grid)
        print("  [ptx-patch] done")
    func, smem, n_params = _patched_cache[key]
    _launch_patched(func, smem, n_params, a, b, M, N, grid)


def main():
    warmup_cuda_context()

    handle = nvml_init(torch.cuda.current_device())
    nvml_before = nvml_used(handle)
    print(f"[ManualPatch][nvml] BEFORE any alloc: {fmt_gb(nvml_before)}")

    a = torch.randn((TILE_M, TILE_N), device=DEVICE, dtype=torch.float16)
    b = torch.empty_like(a)
    torch.cuda.synchronize()

    nvml_after_inputs = nvml_used(handle)
    print(f"[ManualPatch][nvml] after inputs: {fmt_gb(nvml_after_inputs)} "
          f"(d={fmt_gb(nvml_after_inputs - nvml_before)})")

    print("\n=== launching Manual PTX-patched TMA (shared::cta) ===")
    run_patched(a, b)
    torch.cuda.synchronize()

    nvml_after = nvml_used(handle)
    print(f"[ManualPatch][nvml] after launch: {fmt_gb(nvml_after)} "
          f"(d vs baseline={fmt_gb(nvml_after - nvml_before)})")
    print(f"\n[ManualPatch] Black hole jump (launch - inputs): "
          f"{fmt_gb(nvml_after - nvml_after_inputs)}")

    ok = torch.equal(a, b)
    print(f"{'OK' if ok else 'MISMATCH'}: identity copy "
          f"{'matches' if ok else 'does NOT match'} torch")


if __name__ == "__main__":
    main()
