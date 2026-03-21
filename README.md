# CUDA black magic

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

## Minimal Example

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
