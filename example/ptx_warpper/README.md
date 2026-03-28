# PTX Wrapper Test

Test for intercepting `ptxas` invocations and capturing PTX inputs across multiple CUDA frameworks.

## Setup

```shell
git clone https://github.com/dsl-learn/cuda-magic.git
cd cuda-magic
```

## Install packages

```shell
pip install cuda-tile[tileiras]
# pip install cuda-tile
pip install nvidia-cutlass-dsl
pip install triton
```

## Wrapper

For a brief overview of [`ptxas_wrapper.py`](../../ptxas_wrapper.py), see the [`PTXAS Wrapper`](../../README.md#ptxas-wrapper) section in the repository root README.

In this test directory, the wrapper is used in two ways:

- `triton` and `cutedsl` run the target Python script with framework-specific dump settings and collect the generated `.ptx` files.
- `install` swaps the discovered `ptxas` with a shim, which is useful for flows that call `ptxas` directly, such as `nvcc`.

Captured PTX files are written to `./ptx_dumps/` by default, or to `PTX_DUMP_DIR` when that environment variable is set.

For `nvcc`-based flows, `install` requires the ptxas path explicitly (e.g. `/usr/local/cuda/bin/ptxas`) since nvcc may use a different ptxas than the one auto-detected.

## Usage

**cuda.tile**
```shell
CUDA_TILE_CACHE_DIR=0 python3 test/ptx_warpper/vec_add_cutile.py
```

**CuteDSL (cutlass DSL)**
```shell
python3 ptxas_wrapper.py cutedsl test/ptx_warpper/vec_add_cutedsl.py
```

**Triton**
```shell
python3 ptxas_wrapper.py triton test/ptx_warpper/vec_add_triton.py
```

**CUDA C++ (via nvcc + ptxas wrapper)**
```shell
sudo python3 ptxas_wrapper.py install
nvcc test/ptx_warpper/vec_add_cuda.cu -o /tmp/vec_add_cuda -arch=sm_75
sudo python3 ptxas_wrapper.py uninstall
```

**CUTLASS C++**

Requires the CUTLASS source tree:
```shell
git clone https://github.com/NVIDIA/cutlass.git
```

```shell
sudo python3 ptxas_wrapper.py install
nvcc test/ptx_warpper/vec_add_cutlass.cu -o /tmp/vec_add_cutlass \
    -I./cutlass/include \
    -I./cutlass/tools/util/include
sudo python3 ptxas_wrapper.py uninstall
```
