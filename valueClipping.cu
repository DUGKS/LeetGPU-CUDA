#include <cuda_runtime.h>
#include <stdio.h>

__device__ float clip(float x, float lo, float hi) {
    return fminf(hi, fmaxf(x, lo));
}

__global__ void valueClippingKernel (float *output, const float *input, float lo, float hi, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int num_float4s = N / 4;
    int stride = blockDim.x * gridDim.x;
    const float4* input_float4 = reinterpret_cast<const float4*> (input);
    float4* output_float4 = reinterpret_cast<float4 *>(output);
    for (int i = idx; i < num_float4s; i += stride) {
        float4 data = input_float4[i];
        data.x = clip(data.x, lo, hi);
        data.y = clip(data.y, lo, hi);
        data.z = clip(data.z, lo, hi);
        data.w = clip(data.w, lo, hi);
        output_float4[i] = data;
    }

    for (int i = num_float4s * 4 + idx; i < N; i += stride) {
        output[i] = clip(input[i], lo, hi);
    }
}