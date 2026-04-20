# CUDA black magic

## Contents

- [Pretending to be CUTLASS for Free Performance](#pretending-to-be-cutlass-for-free-performance)
  - [Triton reproduction](#triton-reproduction)
  - [Minimal example](#minimal-example)
- [RTX 5090 TMA Memory Black Hole](#rtx-5090-tma-memory-black-hole)
- [PTXAS Wrapper](#ptxas-wrapper)

## Pretending to be CUTLASS for Free Performance

### Triton reproduction

find it in [Triton](https://github.com/triton-lang/triton/commit/ade3d49e624ac414cde33a4c982d656fc7e49605).

> fp8 is ~100 tflops faster when the kernel name has "cutlass"
in it.

You can reproduce it by running the command below — no need to build Triton. The only difference between [gluon_attention.ptx](./triton_cache/gluon_attention/attention_kernel.ptx) and [cutlass_gluon_attention.ptx](./triton_cache/cutlass_gluon_attention/attention_kernel.ptx) lies in their function names.

```bash
wget https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.8.93-archive.tar.xz
tar -xf cuda_nvcc-linux-x86_64-12.8.93-archive.tar.xz
```

```bash
git clone https://github.com/OpenMLIR/cuda-magic
cd cuda-magic
cuda_nvcc-linux-x86_64-12.8.93-archive/bin/ptxas -lineinfo -v --gpu-name=sm_100a triton_cache/gluon_attention/attention_kernel.ptx -o gluon_attention.cubin
cuda_nvcc-linux-x86_64-12.8.93-archive/bin/ptxas -lineinfo -v --gpu-name=sm_100a triton_cache/cutlass_gluon_attention/attention_kernel.ptx -o cutlass_gluon_attention.cubin
```

You can use `ls -lh` to check the sizes of different .cubin files.

### Minimal example

The minimal PTX that reproduces the behavior is in [`minimal_example/`](./minimal_example/). The two files are **identical** except for the function name (`cutlass_kernel` vs `plain_kernel`).

The minimum requirements are:
1. Function name contains `cutlass`
2. `--gpu-name=sm_100a` (Blackwell only)
3. `tcgen05.mma.cta_group::1.kind::f8f6f4` instruction **inside a loop**

A single `tcgen05.mma` without a loop does **not** trigger the behavior.

```bash
cuda_nvcc-linux-x86_64-12.8.93-archive/bin/ptxas -v --gpu-name=sm_100a minimal_example/cutlass_kernel.ptx -o minimal_example/cutlass_kernel.cubin
cuda_nvcc-linux-x86_64-12.8.93-archive/bin/ptxas -v --gpu-name=sm_100a minimal_example/plain_kernel.ptx -o minimal_example/plain_kernel.cubin
ls -la minimal_example/*.cubin
```

Result: `cutlass_kernel.cubin` is **56% larger** (12392 vs 7920 bytes). The difference is entirely in the `.nv.capmerc` section (ptxas scheduling metadata): **5476 vs 1134 bytes (4.8×)**.

| Condition | Size ratio | `.nv.capmerc` ratio |
|-----------|-----------|---------------------|
| No GEMM | 1.00× | 1.00× |
| Single `tcgen05.mma`, no loop | 1.02× | 1.00× |
| `brx.idx` state machine, no MMA | 1.02× | 1.00× |
| `tcgen05.mma` **in a loop** | **1.56×** | **4.83×** |
| Original attention kernel | 1.15× | 3.20× |

Notably, removing the warp predicate (`@%p0`) from `tcgen05.mma` causes ptxas itself to **segfault** on the `cutlass_` version while the plain version compiles successfully — further confirming that "cutlass" triggers a completely different code path in ptxas.

## RTX 5090 TMA Memory Black Hole

[`5090_tma_mem_blackhole.py`](./5090_tma_mem_blackhole.py) reproduces another Blackwell-specific issue: on RTX 5090, enabling Triton TMA (`tl.make_tensor_descriptor`) can add **~3.7 GiB** of device memory usage compared with a plain `tl.load` / `tl.store` kernel, even though PyTorch reports the same allocator state in both cases.

Run the two variants in **fresh processes**:

```bash
python 5090_tma_mem_blackhole.py --kernel tma
python 5090_tma_mem_blackhole.py --kernel plain
```

The script measures both `torch.cuda.memory_*` and NVML / `nvidia-smi`, so it can separate PyTorch allocator usage from driver-side memory growth.

Typical observation on RTX 5090:

| Metric | Plain (`tl.load` / `tl.store`) | TMA (`make_tensor_descriptor`) |
|--------|-------------------------------:|-------------------------------:|
| NVML used | 1.062 GiB | 4.703 GiB |
| NVML Δ vs baseline | 0.082 GiB | **3.723 GiB** |
| `torch.cuda.memory_reserved()` | 0.002 GiB | 0.002 GiB |

That means the extra memory is **not** coming from PyTorch's caching allocator. The current working hypothesis is that this is CUDA driver / runtime state triggered by the TMA path, not tensor storage requested by user code.

This was also observed on RTX PRO 6000, but the impact is much easier to notice on 5090 because the card has far less headroom. If your workload is already memory-bound, TMA on 5090 can reduce usable batch size enough to cause unexpected OOMs.


## PTXAS Wrapper

[`ptxas_wrapper.py`](./ptxas_wrapper.py) is a small helper for capturing the PTX that different CUDA frontends eventually hand to `ptxas`. By default it writes dumps to [`ptx_dumps/`](./ptx_dumps/), and you can override that location with `PTX_DUMP_DIR=/your/path`.

It supports two capture modes:

- `python3 ptxas_wrapper.py install [ptxas_path]` replaces the discovered `ptxas` with a thin wrapper and keeps the original binary as `ptxas.real`. This is the mode to use for direct `ptxas` callers such as `nvcc` or `tileiras`.
- `python3 ptxas_wrapper.py triton <script.py>` and `python3 ptxas_wrapper.py cutedsl <script.py>` run the target script with framework-specific dump settings, then collect the generated `.ptx` files into `ptx_dumps/`.

Useful commands:

```bash
python3 ptxas_wrapper.py status
python3 ptxas_wrapper.py install
python3 ptxas_wrapper.py uninstall
```

See [`example/ptx_warpper/`](./example/ptx_warpper/README.md) for end-to-end examples covering cuda.tile, CuteDSL, Triton, and CUDA/CUTLASS C++.
