#include <cuda_runtime.h>
#include <format>
#include <iostream>

__global__ void RGB2GrayKernel(float *output, float *input, int width, int height) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int idx3 = idx * 3;
    const float alpha = 0.299f;
    const float beta = 0.587f;
    const float gamma = 0.114f;
    if (idx < width * height) {
        const float4 rgb_in = *reinterpret_cast<const float4 *>(input + idx3);
        output[idx] = alpha * rgb_in.x + beta * rgb_in.y+ gamma * rgb_in.z;
    }
}