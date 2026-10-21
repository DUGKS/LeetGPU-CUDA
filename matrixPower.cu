#include <cuda_runtime.h>

constexpr int TILE = 32;
__global__ void matrixMultiplyKernel(float *C, const float *A, const float *B, int N) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

    const int col = blockIdx.x * TILE + tidx;
    const int row = blockIdx.y * TILE + tidy;
    float acc = 0.0f;

    for (int i = 0; i < N; i += TILE) {
        const int col_A = i + tidx;
        const int row_B = i + tidy;
        sA[tidy][tidx] = (row < N && col_A < N) ? A[row * N + col_A] : 0.0f;
        sB[tidy][tidx] = (row_B < N && col < N) ? B[row_B * N + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            acc += sA[tidy][k] * sB[k][tidx];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = acc;
    }
}

__global__ void identityKernel(float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * N) { C[idx] = (idx / N == idx % N) ? 1.0f : 0.0f; }
}

static inline void matmul(float *C, const float *A, const float *B, int N) {
    dim3 block{TILE, TILE};
    dim3 grid{(N + TILE - 1) / TILE, (N + TILE - 1) / TILE};
    matrixMultiplyKernel<<<grid, block>>>(C, A, B, N);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int P) {
    size_t num_bytes = (size_t) N * N * sizeof(float);
    if (P == 0) {
        int threads = 256;
        int blocks = (N * N + threads - 1) / threads;
        identityKernel<<<blocks, threads>>>(output, N);
        cudaDeviceSynchronize();
        return;
    }
    if (P == 1) {
        cudaMemcpy(output, input, num_bytes, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
        return;
    }

    float *base, *result, *tmp;
    cudaMalloc(&base, num_bytes);
    cudaMalloc(&result, num_bytes);
    cudaMalloc(&tmp, num_bytes);
    cudaMemcpy(base, input, num_bytes, cudaMemcpyDeviceToDevice);

    bool have_result = false;
    int p = P;
    while (p > 0) {
        if (p & 1) {
            if (!have_result) {
                cudaMemcpy(result, base, num_bytes, cudaMemcpyDeviceToDevice);
                have_result = true;
            } else {
                matmul(tmp, result, base, N);
                float *s = result; result = tmp; tmp = s;
            }
        }
        p >>= 1;
        if (p > 0) {
            matmul(tmp, base, base, N);
            float *s = base; base = tmp; tmp = s;
        }

    }

    cudaMemcpy(output, result, num_bytes, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();

    cudaFree(base);
    cudaFree(result);
    cudaFree(tmp);
}
