import cutlass.cute as cute

# 这是kernel
@cute.kernel
def vector_add_kernel(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    thread_idx = bidx * bdim + tidx
    if thread_idx < N:
        C[thread_idx] = A[thread_idx] + B[thread_idx]

# 这是host
@cute.jit
def solve(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint32):
    block_dim = 256, 1, 1
    grid_dim = cute.ceil_div((N, 1, 1), block_dim)
    vector_add_kernel(A, B, C, N).launch(grid=grid_dim, block=block_dim)


import torch
from cutlass.cute.runtime import from_dlpack

N = 1005
a = torch.randn(N, device="cuda", dtype=torch.float32)
b = torch.randn(N, device="cuda", dtype=torch.float32)
c = torch.zeros(N, device="cuda", dtype=torch.float32)
a_tensor = from_dlpack(a, assumed_align=16)
b_tensor = from_dlpack(b, assumed_align=16)
c_tensor = from_dlpack(c, assumed_align=16)

solve(a_tensor, b_tensor, c_tensor, N)

# verify correctness
torch.testing.assert_close(c, a + b)
print("pass!")
