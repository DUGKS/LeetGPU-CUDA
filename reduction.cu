#include <cuda_runtime.h>
#include <format>
#include <iostream>

__global__ void reduceKernel(float *output, const float *input, const int N) {
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int idx = blockDim.x * blockIdx.x + tid;
    const int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int i = idx; i < N; i += stride) {
        sum += input[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int i = blockDim.x / 2; i > 16; i >>= 1) {
        if (tid < i) {
            sdata[tid] += sdata[tid + i];
        }
        __syncthreads();
    }

    if (tid < 32) {
        float val = sdata[tid];
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (tid == 0) {
            atomicAdd(output, val);
        }
    }
}