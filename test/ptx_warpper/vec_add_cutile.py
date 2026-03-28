import torch

import cuda.tile as ct
import math

# --- Kernel 1: 1D Tiled Vector Add (Direct Load/Store) ---
@ct.kernel
def vec_add_kernel_1d(a, b, c, TILE: ct.Constant[int]):

    # Get block id along dimension 0
    bid = ct.bid(0)

    # Load a tile of size TILE from global arrays based on bid
    # a_tile[x, ] = array[bid * TILE + x,]  (for all 0<=x<TILE)
    a_tile = ct.load(a, index=(bid,), shape=(TILE,))
    b_tile = ct.load(b, index=(bid,), shape=(TILE,))

    # Element-wise addition of the loaded tiles
    sum_tile = a_tile + b_tile

    # Store the result tile back to global array c based on bid
    ct.store(c, index=(bid,), tile=sum_tile)


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    N = output.numel()
    # Heuristic TILE size: smallest power of 2 >= N, capped at 1024
    TILE = min(1024, 2 ** math.ceil(math.log2(N))) if N > 0 else 1

    # Compute grid dimensions: ceil(N / TILE) blocks to cover the full vector
    grid = (math.ceil(N / TILE), 1, 1)  # (blocks_x, blocks_y, blocks_z)

    # Launch the kernel; last argument is the kernel args tuple
    ct.launch(torch.cuda.current_stream(), grid, vec_add_kernel_1d, (x, y, output, TILE))
    return output


torch.manual_seed(0)
size = 98432
x = torch.rand(size, device="cuda")
y = torch.rand(size, device="cuda")
output_torch = x + y
output_cutile = add(x, y)
print(output_torch)
print(output_cutile)
print(f'The maximum difference between torch and cutile is '
      f'{torch.max(torch.abs(output_torch - output_cutile))}')
