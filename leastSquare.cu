#include <cuda_runtime.h>

constexpr int TILE = 16;
__global__ void gramKernel(double *C, const float *X, const float *y, int M, int N, int w) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

    const int row = blockIdx.y * TILE + tidy;
    const int col = blockIdx.x * TILE + tidx;

    const int rowLoad = blockIdx.y * TILE + tidx;

    double acc = 0.0;
    for (int k = 0; k < M; k += TILE) {
        const int mk = k + tidy;
        float a = 0.0f, b = 0.0f;

        if (mk < M) {
            if (rowLoad < N) a = X[mk * N + rowLoad];
            if (col < N) b = X[mk * N + col];
            if (col == N) b = y[mk];
        }

        sA[tidy][tidx] = a;
        sB[tidy][tidx] = b;
        __syncthreads();

        for (int i = 0; i < TILE; i++) {
            acc += sA[i][tidy] * sB[i][tidx];
        }
        __syncthreads();
    }
    if (row < N && col < (N + 1)) {
        C[row * (N + 1) + col] = acc;
    }
}

__global__ void GaussJordanEliminateKernel(double *C, int k, int N, int W) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= N || col >= W) return;
    if (row == k || col <= k) return;

    const double alpha = C[row * W + k] / C[k * W + k];
    C[row * W + col] -= alpha * C[k * W + col];
}

__global__ void finalizeKernel(float *beta, const double *C, int N, int W) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        beta[idx] = C[idx * W + W - 1] / C[idx * W + idx];
    }
}

extern "C" void solve(const float* X, const float* y, float* beta, int n_samples, int n_features) {
    const int M = n_samples;
    const int N = n_features;
    const int W = N + 1;

    double *device_C;
    cudaMalloc(&device_C, sizeof(double) * N * W);
    cudaMemset(device_C, 0x0, sizeof(double) * N * W);

    dim3 block(TILE, TILE);
    dim3 grid((W + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    gramKernel<<<grid, block>>>(device_C, X, y, M, N, W);

    dim3 e_block{32, 8};
    dim3 e_grid{(W + 32 - 1) / 32, (N + 8 - 1) / 8};
    for (int k = 0; k < N; k++) {
        GaussJordanEliminateKernel<<<e_grid, e_block>>>(device_C, k, N, W);
    }

    finalizeKernel<<<(N + 256 - 1) / 256, 256>>>(beta, device_C, N, W);
    cudaDeviceSynchronize();
    cudaFree(device_C);
}