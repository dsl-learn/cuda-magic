# PTX Wrapper Test

Test for intercepting `ptxas` invocations and capturing PTX inputs across multiple CUDA frameworks.

## Setup

```shell
git clone https://github.com/dsl-learn/cuda-magic.git
cd cuda-magic
git clone https://github.com/NVIDIA/cutlass.git
```

## Install packages

```shell
pip install cuda-tile
pip install cuda-tile[tileiras]
pip install nvidia-cutlass-dsl
pip install triton
```

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
python3 ptxas_wrapper.py install
nvcc test/ptx_warpper/vec_add_cuda.cu -o /tmp/vec_add_cuda -arch=sm_75
```

**CUTLASS C++**
```shell
python3 ptxas_wrapper.py install

nvcc test/ptx_warpper/vec_add_cutlass.cu -o /tmp/vec_add_cutlass \
    -I./cutlass/include \
    -I./cutlass/tools/util/include
```
