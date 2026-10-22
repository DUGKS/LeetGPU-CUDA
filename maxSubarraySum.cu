#include <cuda_runtime.h>
#include <climits>

template<int BLOCK_SIZE>
__global__ void prefixSumKernel(int *output, const int *input, int N) {
    __shared__ int sdata[BLOCK_SIZE];
    const int tidx = threadIdx.x;
    const int chunk = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int start = tidx * chunk;
    const int end = min(start + chunk, N);

    int sum = 0;
    for (int i = start; i < end; i++) {
        sum += input[i];
    }
    sdata[tidx] = sum;
    __syncthreads();

    for (int off = 1; off < BLOCK_SIZE; off <<= 1) {
        int val = tidx >= off ? sdata[tidx - off] : 0;
        __syncthreads();
        sdata[tidx] += val;
        __syncthreads();
    }

    int run = sdata[tidx] - sum;

    for (int i = start; i < end; i++) {
        run += input[i];
        output[i] = run;
    }
}

template<int BLOCK_SIZE>
__global__ void maxSubarraySumKernel(int *output, const int *input, int num_in, int num_out, int kernel_size) {
    const int tidx = threadIdx.x;
    const int gidx = blockIdx.x * BLOCK_SIZE + tidx;
    const int stride = blockDim.x * gridDim.x;

    int local = INT_MIN;
    for (int i = gidx; i < num_out; i += stride) {
        int val_start = (i - 1) >= 0 ? input[i - 1] : 0;
        int val_end = input[i + kernel_size - 1];
        local = max(local, val_end - val_start);
    }

    for (int off = 16; off > 0; off >>= 1) {
        local = max(local, __shfl_down_sync(0xffffffff, local, off));
    }

    if ((tidx & 31) == 0) {
        atomicMax(output, local);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int window_size) {
    constexpr int BLOCK_SIZE = 1024;
    const int num_out = N - window_size + 1;
    const int grid = (num_out + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int init = INT_MIN;
    cudaMemcpy(output, &init, sizeof(int), cudaMemcpyHostToDevice);

    int *prefix_sum;
    cudaMalloc(&prefix_sum, sizeof(int) * N);
    prefixSumKernel<BLOCK_SIZE><<<1, BLOCK_SIZE>>>(prefix_sum, input, N);
    maxSubarraySumKernel<BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(output, prefix_sum, N, num_out, window_size);
    cudaFree(prefix_sum);
}