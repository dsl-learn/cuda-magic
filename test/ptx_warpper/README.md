# PTX Wrapper Test

Test for intercepting `ptxas` invocations triggered by `cuda.tile` and capturing the PTX inputs.

## Usage

```shell
CUDA_TILE_CACHE_DIR=0 python3 test/ptx_warpper/cutile.py

python3 ptxas_wrapper.py cutedsl test/ptx_warpper/cutedsl.py
```

## install package
```shell
pip install cuda-tile

pip install cuda-tile[tileiras]

pip install nvidia-cutlass-dsl

```
