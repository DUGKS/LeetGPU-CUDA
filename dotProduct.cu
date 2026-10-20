#include <cuda_runtime.h>

// constexpr int BLOCK_SIZE = 512;

template<int BLOCK_SIZE>
__global__ void dotKernel(float *block_sums, const float *A, const float *B, int N) {
    __shared__ float results[BLOCK_SIZE];
    const int tidx = threadIdx.x;
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    float acc = 0.0f;
    for (int i = idx; i < N; i += stride) {
        acc += A[i] * B[i];
    }
    results[tidx] = acc;
    __syncthreads();

    for (int i = BLOCK_SIZE / 2; i > 16; i >>= 1) {
        if (tidx < i) {
            results[tidx] += results[tidx + i];
        }
        __syncthreads();
    }

    if (tidx < 32) {
        float val = results[tidx];
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (tidx == 0) {
            block_sums[blockIdx.x] = val;
        }
    }

}

template<int BLOCK_SIZE>
__global__ void blockSum(float *output, const float *input, const int N) {
    __shared__ float results[BLOCK_SIZE];
    const int tidx = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tidx;
    const int stride = blockDim.x * gridDim.x;

    float acc = 0.0;
    for (int i = idx; i < N; i += stride) {
        acc += input[i];
    }
    results[tidx] = acc;
    __syncthreads();

    for (int i = BLOCK_SIZE / 2; i > 16; i >>= 1) {
        if (tidx < i) {
            results[tidx] += results[tidx + i];
        }
        __syncthreads();
    }

    if (tidx < 32) {
        float val = results[tidx];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, off);
        }
        if (tidx == 0) {
            atomicAdd(output, val);
        }
    }

}
