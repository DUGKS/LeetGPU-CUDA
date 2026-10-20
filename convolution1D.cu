#include <cuda_runtime.h>
#include <stdio.h>

__global__ void convolutionKernel (float *output, const float *input, const float *kernel, const int input_size, const int kernel_size) {
    extern __shared__ float shared_kernel[];
    for (int i = threadIdx.x; i < kernel_size; i += blockDim.x) {
        shared_kernel[i] = kernel[i];
    }
    __syncthreads();

    const int stride = blockDim.x * gridDim.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = idx; i < input_size - kernel_size + 1; i += stride) {
        float sum = 0.0f;
        for (int k = 0;k < kernel_size; k++) {
            sum += shared_kernel[k] * input[i + k];
        }
        output[i] = sum;
    }
}