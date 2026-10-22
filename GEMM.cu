#include <cuda_fp16.h>
#include <cuda_runtime.h>

template<int TILE>
__global__ void GEMMKernel(half *C, const half *A, const half *B, int M, int K, int N, float alpha, float beta) {
    __shared__ half tile_A[TILE][TILE];
    __shared__ half tile_B[TILE][TILE];

    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

    const int bidx = TILE * blockIdx.x;
    const int bidy = TILE * blockIdx.y;

    const int row = bidy + tidy;
    const int col = bidx + tidx;

    float acc = 0.0;
    const half zero = __float2half(0.0f);

        for (int base = 0; base < K; base += TILE) {
            const int col_A = base + tidx;
            const int row_B = base + tidy;
            tile_A[tidy][tidx] = row < M && col_A < K ? A[row * K + col_A] : zero;
            tile_B[tidy][tidx] = row_B < K && col < N ? B[row_B * N + col] : zero;
            __syncthreads();

            for (int k = 0; k < TILE; k++) {
                acc += __half2float(tile_A[tidy][k]) * __half2float(tile_B[k][tidx]);
            }
            __syncthreads();
        }

    if (row < M && col < N) {
        float result = alpha * acc + beta * __half2float(C[row * N + col]);
        C[row * N + col] = __float2half(result);
    }
}

// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    constexpr int TILE = 16;
    dim3 block_size{TILE, TILE};
    dim3 grid_size{(N + TILE - 1) / TILE, (M + TILE - 1) / TILE};
    GEMMKernel<TILE><<<grid_size, block_size>>>(C, A, B, M, K, N, alpha, beta);
}