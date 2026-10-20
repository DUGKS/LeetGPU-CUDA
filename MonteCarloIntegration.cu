#include <cuda_runtime.h>

template<int BLOCK_SIZE>
__global__ void sumBlockKernel(float *partial, float *input, int N) {
    __shared__ float sum[BLOCK_SIZE];
    const int tidx = threadIdx.x;
    const int gidx = blockDim.x * blockIdx.x + tidx;
    sum[tidx] = gidx < N ? input[gidx] : 0.0f
    __syncthreads();

    for (int i = BLOCK_SIZE / 2; i > 16; i >>= 1) {
        if (tidx < i) {
            sum[tidx] += sum[tidx + i]
        }
        __syncthreads();
    }

    if (tidx < 32) {
        float val = sum[tidx];
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (tidx == 0) {
            partial[blockIdx.x] = val;
        }
    }
}

template<int BLOCK_SIZE>
__global__ void sumTotalKernel(float *result, float *input, int N, float alpha) {
    __shared__ float sum[BLOCK_SIZE];
    const int tidx = threadIdx.x;
    sum[tidx] = tidx < N ? input[tidx] : 0.0f;
    __syncthreads();

    for (int i = BLOCK_SIZE / 2; i > 16; i >>= 1) {
        if (tidx < i) {
            sum[tidx] += sum[tidx + i]
        }
        __syncthreads();
    }

    if (tidx < 32) {
        float val = sum[tidx];
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (tidx == 0) {
            *result = val * alpha;
        }
    }
}