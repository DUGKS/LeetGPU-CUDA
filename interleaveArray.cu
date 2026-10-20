#include <stdio.h>
#include <cuda_runtime.h>

__global__ void interleaveKernel(float *C, const float *A, const float *B, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int idx4 = idx * 4;
    const float4 A4 = *reinterpret_cast<const float4 *> (A + idx4);
    const float4 B4 = *reinterpret_cast<const float4 *> (B + idx4);
    float4 out0 = make_float4(A4.x, B4.x, A4.y, B4.y);
    float4 out1 = make_float4(A4.z, B4.z, A4.w, B4.w);
    *reinterpret_cast<float4 *>(C + idx4 * 2) = out0;
    *reinterpret_cast<float4 *>(C + idx4 * 2 + 1) = out1;
}