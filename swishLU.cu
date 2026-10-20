#include <cuda_runtime.h>
#include <stdio.h>

__device__ float silu (float x) {
    return x * 1.0f / (1.0f + __expf(-x));
}

__global__ void swiglu(float *output, const float *input, int N) {
    const int num_half = N / 2;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const float4 *input_A = reinterpret_cast<const float4 *>(input);
    const float4 *input_B = reinterpret_cast<const float4 *>(input + N / 2);
    float4 *output4 = reinterpret_cast<float4 *>(output);
    for (int i = idx; i < num_half / 4; i += stride) {
        float4 a = input_A[i];
        float4 b = input_B[i];
        float4 result;
        result.x = silu(a.x) * b.x;
        result.y = silu(a.y) * b.y;
        result.z = silu(a.z) * b.z;
        result.w = silu(a.w) * b.w;
        output4[i] = result;
    }

    for (int i = num_half / 4 * 4 + idx; i < num_half; i += stride) {
        output[i] = silu(input[i]) * input[i + num_half];
    }
}