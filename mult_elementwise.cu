#include <stdio.h>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <chrono>
#include <iostream>
#include <cassert>

__global__ void warmupKernel() {}

// C = A x B
// A: M x K   B: K x N   C: M x N
__global__ void multKernelPlain(float *C, const float *A, const float *B,
                                const int M, const int K, const int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float sum = 0.0;
        for (int i = 0; i < K;++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// blockDim.x == blockDim.y
// Thread_Tile_M = Block_Tile_M / blockDim.y
// Thread_Tile_N = Block_Tile_N / blockDim.x
// Thread_Tile_K = Block_Tile_K / blockDim.x or blockDim.y
template<int Block_Tile_M, int Block_Tile_N, int Block_Tile_K,
         int Thread_Tile_M, int Thread_Tile_N, int Thread_Tile_K>
__global__ void multKernelTiled (float *C, const float *A, const float *B,
                                 const int M, const int K, const int N) {
    assert(blockDim.x == blockDim.y);

    __shared__ float tile_A[Block_Tile_M][Block_Tile_K];
    __shared__ float tile_B[Block_Tile_K][Block_Tile_N];

    const int row_block = blockIdx.y * Block_Tile_M;
    const int col_block = blockIdx.x * Block_Tile_N;

    const int row_thread_A = threadIdx.y * Thread_Tile_M;
    const int col_thread_A = threadIdx.x * Thread_Tile_K;

    const int row_thread_B = threadIdx.y * Thread_Tile_K;
    const int col_thread_B = threadIdx.x * Thread_Tile_N;

    float sum[Thread_Tile_M][Thread_Tile_N] = {0.0f};

    int num_tiles_K = (K + Block_Tile_K - 1) / Block_Tile_K;

    for (int k = 0;k < num_tiles_K;k++) {
        const int offset_k = k * Block_Tile_K;
        for (int j = 0;j < Thread_Tile_M;j++) {
            const int rowA = row_block + row_thread_A + j;
            for (int i = 0;i < Thread_Tile_K;i++) {
                const int colA = offset_k + col_thread_A + i;
                tile_A[row_thread_A + j][col_thread_A + i] = (rowA < M && colA < K) ? A[rowA * K + colA] : 0.0f;
            }
        }

        for (int j = 0;j < Thread_Tile_K;j++) {
            const int rowB = offset_k + row_thread_B + j;
            for (int i = 0;i < Thread_Tile_N;i++) {
                const int colB = col_block + col_thread_B + i;
                tile_B[row_thread_B + j][col_thread_B + i] = (rowB < K && colB < N) ? B[rowB * N + colB] : 0.0f;
            }
        }
        __syncthreads();

        for (int j = 0;j < Thread_Tile_M;j++) {
            for (int i = 0; i < Thread_Tile_N;i++) {
                for (int n = 0; n < Block_Tile_K;n++) {
                    sum[j][i] += tile_A[row_thread_A + j][n] * tile_B[n][col_thread_B + i];
                }
            }
        }
        __syncthreads();
    }

    for (int j = 0; j < Thread_Tile_M; j++) {
        const int row = row_block + row_thread_A + j;
        for (int i = 0; i < Thread_Tile_N; i++) {
            const int col = col_block + col_thread_B + i;
            if (row < M && col < N) {
                C[row * N + col] = sum[j][i];
            }
        }
    }
}

class multKernel {
public:
    static constexpr int BLOCK_SIZE_X = 8;
    static constexpr int BLOCK_SIZE_Y = 8;
    static constexpr int BLOCK_TILE_M = 32;
    static constexpr int BLOCK_TILE_N = 32;
    // static constexpr int BLOCK_TILE_K = 16;
    // static constexpr int THREAD_TILE_M = 4;
    // static constexpr int THREAD_TILE_N = 4;
    // static constexpr int THREAD_TILE_K = 2;
public:
    int M;
    int K;
    int N;
    float *device_float_A;
    float *device_float_B;
    float *device_float_C;
    float *host_C;
public:
    multKernel(float *host_A, float *host_B, int M, int K, int N): M(M), K(K), N(N) {
        cudaMalloc(&device_float_A, M * K * sizeof(float));
        cudaMalloc(&device_float_B, K * N * sizeof(float));
        cudaMalloc(&device_float_C, M * N * sizeof(float));

        cudaMemcpy(device_float_A, host_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(device_float_B, host_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

        host_C = reinterpret_cast<float*>(malloc(M * N * sizeof(float)));
    }

    void runTiled() {
        auto start = std::chrono::steady_clock::now();
        int num_col = (N + BLOCK_TILE_N - 1) / BLOCK_TILE_N;
        int num_row = (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M;
        dim3 grid_size(num_col, num_row);
        dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);

        multKernelTiled<32,32,16,4,4,2><<<grid_size, block_size>>>(device_float_C, device_float_A, device_float_B, M, K, N);
        cudaMemcpy(host_C, device_float_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        printf("Tiled computing: consumed %ld microseconds\n", ms);
    }

    void run() {
        auto start = std::chrono::steady_clock::now();
        dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
        int num_col = (N + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
        int num_row = (M + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;
        dim3 grid_size(num_col, num_row);
        multKernelPlain<<<grid_size, block_size>>>(device_float_C, device_float_A, device_float_B, M, K, N);
        cudaMemcpy(host_C, device_float_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        printf("Plain computing: consumed %ld microseconds\n", ms);
    }


    ~multKernel() {
        cudaFree(device_float_A);
        cudaFree(device_float_B);
        cudaFree(device_float_C);
        free(host_C);
    }
};

class floatCompare {
    float *a;
    float *b;
    float tol;
    int N;
    int num_errors{0};
public:
    floatCompare(float *A, float *B, int size, float tol): a(A), b(B), N(size), tol(tol) {}

    void operator ()() {
        float max_error = 0.0f;
        for (int n = 0; n < N; n++) {
            float diff = fabsf(a[n] - b[n]);
            if (diff > max_error) {
                max_error = diff;
            }
            if (diff > tol) {
                num_errors++;
                printf("n = %d, a = %f, b = %f\n", n, a[n], b[n]);
            }
        }
        printf("Error num: %d, Max error: %f\n", num_errors, max_error);
    }
};

void gemm_cpu(float *C, const float *A, const float *B, const int M, const int K, const int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}



int main() {
    const int M = 1024;
    const int K = 1024;
    const int N = 1024;

    float *matrixA = reinterpret_cast<float*>(malloc(M * K * sizeof(float)));
    float *matrixB = reinterpret_cast<float*>(malloc(K * N * sizeof(float)));
    float *matrixC = reinterpret_cast<float*>(malloc(M * N * sizeof(float)));

    srand(74);
    for (int i = 0;i < M * K;i++) {
        matrixA[i] = (float)rand() / RAND_MAX;
    }

    for (int i = 0;i < K*N;i++) {
        matrixB[i] = (float)rand() / RAND_MAX;
    }

    warmupKernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    gemm_cpu(matrixC, matrixA, matrixB, M, K ,N);

    const float tol = 1e-6f;

    multKernel kernelPlain(matrixA, matrixB, M, K, N);
    kernelPlain.run();
    // floatCompare(matrixC, kernelPlain.host_C, M * N, tol)();

    multKernel kernelTiled(matrixA, matrixB, M, K, N);
    kernelTiled.runTiled();
    // floatCompare(matrixC, kernelTiled.host_C, M * N, tol)();

    floatCompare(kernelPlain.host_C, kernelTiled.host_C, M * N, tol)();




    free(matrixA);
    free(matrixB);
    free(matrixC);
}