# PTX Wrapper Test

Test for intercepting `ptxas` invocations triggered by `cuda.tile` and capturing the PTX inputs.

## Usage

```shell
CUDA_TILE_CACHE_DIR=0 python3 test/ptx_warpper/cutile.py
```
