#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/tensor_fill.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

constexpr std::uint32_t kNumElements = 1005;
constexpr int kThreadsPerBlock = 256;

#define CHECK_CUDA(expr)                                                       \
    do {                                                                       \
        cudaError_t err__ = (expr);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << ": " << cudaGetErrorString(err__) << std::endl;       \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

__global__ void vector_add_kernel(const float* a,
                                  const float* b,
                                  float* c,
                                  std::uint32_t n) {
    std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void solve(const float* d_a, const float* d_b, float* d_c, std::uint32_t n) {
    dim3 block_dim(kThreadsPerBlock, 1, 1);
    dim3 grid_dim((n + block_dim.x - 1) / block_dim.x, 1, 1);
    vector_add_kernel<<<grid_dim, block_dim>>>(d_a, d_b, d_c, n);
    CHECK_CUDA(cudaGetLastError());
}

}  // namespace

int main() {
    std::vector<float> h_a(kNumElements);
    std::vector<float> h_b(kNumElements);
    std::vector<float> h_c(kNumElements, 0.0f);

    for (std::uint32_t i = 0; i < kNumElements; ++i) {
        h_a[i] = static_cast<float>(i) * 0.001f;
        h_b[i] = static_cast<float>(kNumElements - i) * 0.001f;
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    CHECK_CUDA(cudaMalloc(&d_a, kNumElements * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, kNumElements * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, kNumElements * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), kNumElements * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), kNumElements * sizeof(float), cudaMemcpyHostToDevice));

    solve(d_a, d_b, d_c, kNumElements);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, kNumElements * sizeof(float), cudaMemcpyDeviceToHost));

    for (std::uint32_t i = 0; i < kNumElements; ++i) {
        float expected = h_a[i] + h_b[i];
        if (std::fabs(h_c[i] - expected) > 1e-6f) {
            std::cerr << "Mismatch at index " << i << ": expected " << expected
                      << ", got " << h_c[i] << std::endl;
            CHECK_CUDA(cudaFree(d_a));
            CHECK_CUDA(cudaFree(d_b));
            CHECK_CUDA(cudaFree(d_c));
            return EXIT_FAILURE;
        }
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    std::cout << "pass!" << std::endl;
    return EXIT_SUCCESS;
}
