#include <cuda_runtime.h>

template<int BLOCK_SIZE>
__global__ void sumKernel(float *output, const float *input, int N) {
    __shared__ float warp_sum[BLOCK_SIZE / 32];
    const int tidx = threadIdx.x;
    const int gidx = blockIdx.x * blockDim.x + tidx;
    const int stride = blockDim.x * gridDim.x;

    float acc = 0.0f;
    for (int i = gidx; i < N; i += stride) {
        acc += input[i] * input[i];
    }

    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, off);
    }

    int warp_idx = tidx / 32;
    int lane = tidx % 32;
    if (lane == 0) warp_sum[warp_idx] = acc;
    __syncthreads();

    if (tidx < 32) {
        float val = tidx < BLOCK_SIZE / 32 ? warp_sum[tidx] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, off);
        }
        if (tidx == 0) atomicAdd(output, val);
    }
}

__global__ void RMSKernel(float *output, const float *input, float *rms, float gamma, float beta, float eps, int N) {
    const int tidx = threadIdx.x;
    const int gidx = blockDim.x * blockIdx.x + tidx;
    const int stride = blockDim.x * gridDim.x;
    const float inverse_rms = 1.f / sqrtf(*rms / N + eps);
    for (int i = gidx; i < N; i += stride) {
        output[i] = input[i] * inverse_rms * gamma + beta;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float gamma, float beta, float* output, int N, float eps) {
    constexpr int BLOCK_SIZE = 1024;
    const int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int sms;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    const int grid_rms = min((N + BLOCK_SIZE - 1) / BLOCK_SIZE, sms * 4);

    float *sum_square;
    cudaMalloc(&sum_square, sizeof(float));
    cudaMemset(sum_square, 0x0, sizeof(float));
    sumKernel<BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(sum_square, input, N);
    RMSKernel<<<grid_rms, BLOCK_SIZE>>>(output, input, sum_square, gamma, beta, eps, N);
    cudaFree(sum_square);
}
