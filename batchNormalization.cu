#include <cuda_runtime.h>

__device__ inline float square(float a, float b) {
    return (a - b) * (a - b);
}

template<int TX, int TY>
__global__ void expectationKernel(float *output, const float *input, int N, int C) {
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int gidx = blockIdx.x * TX + tidx;
    const int gidy = blockIdx.y * TY + tidy;
    __shared__ float data[TY][TX];

    data[tidy][tidx] = (gidx < C && gidy < N) ? input[gidy * C + gidx] : 0.0f;
    __syncthreads();

    for (int i = TY / 2; i > 0; i >>= 1) {
        if (tidy < i) {
            data[tidy][tidx] += data[tidy + i][tidx];
        }
        __syncthreads();
    }

    if (tidy == 0 && gidx < C) {
        atomicAdd(output + gidx, data[tidy][tidx] / N);
    }
}

template<int TX, int TY>
__global__ void varianceKernel(float *output, const float *input, const float* expectation, int N, int C) {
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int gidx = blockIdx.x * TX + tidx;
    const int gidy = blockIdx.y * TY + tidy;
    __shared__ float data[TY][TX];

    data[tidy][tidx] = (gidx < C && gidy < N) ? square(input[gidy * C + gidx], expectation[gidx]) : 0.0f;
    __syncthreads();

    for (int i = TY / 2; i > 0; i >>= 1) {
        if (tidy < i) {
            data[tidy][tidx] += data[tidy + i][tidx];
        }
        __syncthreads();
    }

    if (tidy == 0 && gidx < C) {
        atomicAdd(output + gidx, data[tidy][tidx] / N);
    }
}

template<int TX, int TY>
__global__ void normalization(float *output, const float *input, const float *expectation, const float *variance, const float *gamma, const float *beta, float eps, int N, int C) {
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int gidx = blockIdx.x * TX + tidx;
    const int gidy = blockIdx.y * TY + tidy;

    if (gidx < C && gidy < N) {
        float val = input[gidy * C + gidx];
        val = (val - expectation[gidx]) / sqrt(variance[gidx] + eps) * gamma[gidx] + beta[gidx];
        output[gidy * C + gidx] = val;
    }
}

// input, gamma, beta, output are device pointers
extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output,
                      int N, int C, float eps) {
    float *expectation, *variance;
    cudaMalloc(&expectation, sizeof(float) * C);
    cudaMalloc(&variance, sizeof(float) * C);
    cudaMemset(expectation, 0x0, C * sizeof(float));
    cudaMemset(variance, 0x0, C * sizeof(float));
    const int BLOCK_SIZE_X = 8;
    const int BLOCK_SIZE_Y = 32;
    const int GRID_SIZE_X = (C + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
    const int GRID_SIZE_Y = (N + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;
    dim3 block_size{BLOCK_SIZE_X, BLOCK_SIZE_Y};
    dim3 grid_size{GRID_SIZE_X, GRID_SIZE_Y};
    expectationKernel<BLOCK_SIZE_X, BLOCK_SIZE_Y><<<grid_size, block_size>>>(expectation, input, N, C);
    varianceKernel<BLOCK_SIZE_X, BLOCK_SIZE_Y><<<grid_size, block_size>>>(variance, input, expectation, N, C);
    normalization<BLOCK_SIZE_X, BLOCK_SIZE_Y><<<grid_size, block_size>>>(output, input, expectation,variance, gamma, beta, eps, N, C);
}
