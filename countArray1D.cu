#include <cuda_runtime.h>

template<int BLOCK_SIZE>
__global__ void countEleKernel(int *output, const int *input, int N, int p) {
    __shared__ int sdata[BLOCK_SIZE];
    const int tidx = threadIdx.x;
    const int gidx = tidx + blockDim.x * blockIdx.x;
    const int stride = blockDim.x * gridDim.x;

    int cnt = 0;
    for (int i = gidx; i < N; i += stride)
        cnt += (int)(input[i] == p);

    sdata[tidx] = cnt;
    __syncthreads();

    for (int i = BLOCK_SIZE / 2; i > 16; i >>= 1) {
        if (tidx < i) {
            sdata[tidx] += sdata[tidx + i];
        }
        __syncthreads();
    }

    if (tidx < 32) {
        int val = sdata[tidx];
        for (int off = 16; off > 0; off >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, off);
        }
        if (tidx == 0) {
            atomicAdd(output, val);
        }
    }
}

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int K) {
    const int BLOCK_SIZE = 1024;
    const int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    countEleKernel<BLOCK_SIZE><<<grid_size, BLOCK_SIZE>>>(output, input, N, K);
}
