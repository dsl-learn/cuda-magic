# PTX Wrapper Test

Test for intercepting `ptxas` invocations triggered by `cuda.tile` and capturing the PTX inputs.

## Usage

```shell
CUDA_TILE_CACHE_DIR=0 python3 test/ptx_warpper/vec_add_cutile.py

python3 ptxas_wrapper.py cutedsl test/ptx_warpper/vec_add_cutedsl.py

python3 ptxas_wrapper.py triton test/ptx_warpper/vec_add_triton.py

nvcc test/ptx_warpper/vec_add_cuda.cu -o /tmp/vec_add_cuda_example
```

## install package
```shell
pip install cuda-tile

pip install cuda-tile[tileiras]

pip install nvidia-cutlass-dsl

pip install triton
```
