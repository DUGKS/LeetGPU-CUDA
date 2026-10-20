#include <cuda_runtime.h>

__global__ void mulMatrixVectorKernel(float *output, const float *A, const float *x, int M, int N, int nnz) {
    const int tidx = threadIdx.x;
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = tidx; i < M; i += stride) {
        float acc = 0.0f;
        for (int k = 0; k < N; k++) {
            acc += A[i * N + k] * x[k];
        }
        output[i] = acc;
    }
}
