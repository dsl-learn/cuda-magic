# SM120 TMA Memory Black Hole Reproducer Suite

This directory contains a collection of scripts to reproduce and analyze a **driver-level memory black hole on SM120 triggered by TMA (Tensor Memory Accelerator) instructions**, where several GiB of device memory are unexpectedly allocated.  
Key finding: when TMA PTX uses the `shared::cluster` scope, the CUDA driver lazily allocates a multi-GiB syscall buffer at **kernel launch time** (size scales with the number of SMs; observed ~3–4 GiB on RTX 5090). This allocation is bound to the CUDA Context and cannot be released by `cuModuleUnload` or `empty_cache`.

---

## Directory Layout

| Directory | Description |
|-----------|-------------|
| `common/` | Shared utilities: NVML memory reading, cubin analysis, CUDA warmup, etc. |
| `cuda_bind/` | Raw CUDA driver API wrappers based on `nvidia-cuda-python` |
| `cutedsl/` | TMA reproducer and PTX patching for NVIDIA CuTe DSL |
| `cutile/` | TMA reproducer and cubin analysis for the cuTile framework |
| `gluon/` | TMA reproducer for the Triton Gluon experimental backend |
| `ptx_dumps_sm100/` | Pre-saved SM100 PTX dump samples |
| `tilelang/` | TMA reproducer for TileLang |
| `triton/` | TMA reproducer, launch-time pinpoint, and manual PTX patches for Triton |

---

## Executable Scripts (Grouped by Directory)

### `triton/` — Triton Reproducers and Patches

#### `triton/tma_copy.py`
**Triton TMA memory black hole reproducer**
- Constructs TMA descriptors with `tl.make_tensor_descriptor` and launches a cluster kernel
- Triton 3.4–3.6 emits `cp.async.bulk.tensor` with `shared::cluster` by default
- **Observation**: NVML memory usage jumps by several GiB (scales with SM count) on first launch
- Run: `python triton/tma_copy.py`

#### `triton/blackhole_at_launch.py`
**Pinpoint the black hole to the launch phase**
- Step-by-step measurement: `warmup (compile only)` → `load_binary` → `cuLaunchKernelEx` → `sync` → `empty_cache`
- **Conclusion**: the memory jump happens at **launch (step C)**, not at compile or module-load time, and cannot be released afterward
- Run: `python triton/blackhole_at_launch.py`

#### `triton/tma_manual_patch.py`
**Manual PTX patch workaround**
- String-replaces `shared::cluster` with `shared::cta` in the generated PTX
- Re-assembles via `ptxas` into a cubin and launches manually through `ctypes + libcuda`
- **Effect**: eliminates the black hole with zero performance loss
- Run: `python triton/tma_manual_patch.py`

#### `triton/tma_triton_runner_patch.py`
**Feed patched PTX back via triton_runner source override**
- Requires `triton_runner` and a call to `configure_jit_backend()`
- No manual `ptxas` / `cuLaunchKernel` needed; Triton driver path handles it
- Run: `python triton/tma_triton_runner_patch.py`

---

### `cuda_bind/` — Raw CUDA Driver API Verification

#### `cuda_bind/driver_blackhole.py`
**Raw CUDA driver API verification**
- Uses raw `cuModuleLoadData` / `cuModuleGetFunction` / `cuModuleUnload` from `cuda_bind/driver.py`
- Proves that even explicit `cuModuleUnload` does **not** release the buffer (Context-bound)
- Run: `python cuda_bind/driver_blackhole.py`

---

### `cutedsl/` — CuTe DSL Reproducers and Patches

#### `cutedsl/tma_copy.py`
**CuTe DSL TMA reproducer**
- Probes whether a CuTe DSL TMA kernel pulls in a multi-GiB driver buffer (scales with SM count)
- **Note**: the compiled kernel livelocks on SM120 (sync never returns); the script reads NVML immediately after launch and skips `synchronize`
- Run: `python cutedsl/tma_copy.py`

#### `cutedsl/tma_manual_patch.py`
**CuTe DSL manual PTX patch**
- Same methodology as the Triton manual patch: compile → locate generated PTX → `cluster→cta` replacement → `ptxas` → raw driver load
- Also scans cubin ELF symbols and SASS to verify no `__cuda_syscall_*` and inline `UTMALDG.2D`
- Run: `python cutedsl/tma_manual_patch.py`

---

### `cutile/` — cuTile Framework Reproducers and Analysis

#### `cutile/cutile_tma_copy.py`
**cuTile TMA reproducer + cubin analysis**
- Uses a reduce+broadcast pattern to force cuTile to stage the tile through shared memory, emitting `UTMALDG.2D`
- **Observation**: cuTile does **not** trigger the black hole on SM120 because its codegen correctly uses `shared::cta` scope
- Dumps and analyzes the generated cubin, listing `UTMALDG` / `UBLKCP` / `CALL.ABS.NOINC` / `LDG.E` / `STG.E` counts
- Run: `python cutile/cutile_tma_copy.py`

#### `cutile/tma_reduce_broadcast.py`
**Simplified cuTile TMA reproducer**
- Similar to `cutile_tma_copy.py` but more concise; only compares memory before/after and checks numerical correctness
- Run: `python cutile/tma_reduce_broadcast.py`

#### `cutile/cutile_sm100_dump.py`
**Force SM100 codegen and dump PTX**
- Monkey-patches `cuda.tile._compile.get_sm_arch` to return `"sm_100"`
- Checks whether `multicast::cluster` and `shared::cluster` appear in PTX when `num_ctas=2` is requested
- Actual launch fails on an SM120 host, but PTX is already on disk before `ptxas`
- Run: `python cutile/cutile_sm100_dump.py`

---

### `tilelang/` — TileLang Reproducer

#### `tilelang/tma_copy.py`
**TileLang TMA reproducer**
- Performs a TMA copy via `T.copy` through `alloc_shared`
- Measures NVML memory before and after launch to check for the black hole
- Run: `python tilelang/tma_copy.py`

---

### `gluon/` — Gluon Experimental Backend Reproducer

#### `gluon/tma_copy.py`
**Gluon (Triton experimental backend) TMA reproducer**
- Uses `tma.async_copy_global_to_shared` / `async_copy_shared_to_global` together with `mbarrier`
- Measures whether the Gluon backend triggers the driver-level memory black hole
- Run: `python gluon/tma_copy.py`

---

## Shared Modules

| File | Purpose |
|------|---------|
| `common/benchmark.py` | Unified black hole measurement harness `run_blackhole_test()` |
| `common/cuda_utils.py` | CUDA warmup, Triton allocator, cubin dump and analysis tools |
| `common/mem_utils.py` | NVML memory reading, formatted output (`fmt_gb`) |
| `common/proc_utils.py` | Process / system helper functions |
| `cuda_bind/driver.py` | Thin wrapper around `nvidia-cuda-python`: `cuModuleLoadData`, `cuModuleGetFunction`, `cuModuleUnload`, `cuMemGetInfo` |

---

## Quick Start

```bash
# 1. Verify you are on an SM120 environment
python -c "import torch; print(torch.cuda.get_device_capability(0))"  # expected (12, 0)

# 2. Run any reproducer (Triton example)
python sm120_tma_mem_blackhole/triton/tma_copy.py

# 3. Pinpoint which step causes the allocation
python sm120_tma_mem_blackhole/triton/blackhole_at_launch.py

# 4. Verify the PTX patch eliminates the black hole
python sm120_tma_mem_blackhole/triton/tma_manual_patch.py
```

---

## Key Findings Summary

1. **Root cause**: Triton 3.4–3.6 and some other compilers emit TMA PTX with `shared::cluster` by default, causing the CUDA driver to allocate a multi-GiB internal syscall buffer at `cuLaunchKernelEx` (size scales with SM count).
2. **Scope matters**: Changing `shared::cluster` to `shared::cta` in the PTX eliminates the black hole with no functional or performance penalty.
3. **Non-releasable**: the buffer is CUDA Context–bound; it cannot be freed through normal APIs (`cuModuleUnload`, `empty_cache`, etc.) before process exit.
4. **Framework differences**: cuTile codegen uses `shared::cta` natively and is immune; Triton, CuTe DSL, TileLang, and others default to `shared::cluster` and trigger the black hole.
