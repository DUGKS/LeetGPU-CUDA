#include <cuda_runtime.h>

template<int BLOCK_SIZE>
__global__ void subarraySumKernel(int *output, const int *input, int S, int length) {
    __shared__ int sdata[BLOCK_SIZE];
    const int tidx = threadIdx.x;
    const int gidx = blockIdx.x * blockDim.x + tidx;
    const int stride = blockDim.x * gridDim.x;
    int acc = 0;
    for (int i = gidx; i < length; i += stride) {
        acc += input[i + S];
    }
    sdata[tidx] = acc;
    __syncthreads();

    for (int i = BLOCK_SIZE / 2; i > 16; i >>= 1) {
        if (tidx < i) {
            sdata[tidx] += sdata[tidx + i];
        }
        __syncthreads();
    }

    if (tidx < 32) {
        int val = sdata[tidx];
        for (int i = 16; i > 0; i >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, i);
        }
        if (tidx == 0) {
            atomicAdd(output, val);
        }
    }
}

extern "C" void solve(const int* input, int* output, int N, int S, int E) {
    const int BLOCK_SIZE = 1024;
    const int length = E - S + 1;
    const int GRID_SIZE = min(1024, (length + BLOCK_SIZE - 1) / BLOCK_SIZE);
    subarraySumKernel<BLOCK_SIZE><<<GRID_SIZE, BLOCK_SIZE>>>(output, input, S, length);
}