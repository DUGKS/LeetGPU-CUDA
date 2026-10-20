#include <cuda_runtime.h>

__global__ void histogramming(int *output, const int *input, int N, int num_bins) {
    extern __shared__ int sm[];
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        sm[i] = 0;
    }
    __syncthreads();

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < N; i += stride) {
        const int bin = input[i];
        if (bin >= 0 && bin < num_bins) {
            atomicAdd(sm + bin,1);
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        if (sm[i] != 0) {
            atomicAdd(&output[i], sm[i]);
        }
    }
}