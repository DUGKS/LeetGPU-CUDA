#include <cuda_runtime.h>

template<int BLOCK_SIZE>
__global__ void partialLossKernel(float *output, const float *A, const int *label, int M, int N) {
    __shared__ float block_sums[BLOCK_SIZE];
    const int tidx = threadIdx.x;

    for (int i = blockIdx.x; i < M; i += gridDim.x) {
        float acc = 0.0f;
        for (int j = tidx; j < N; j += blockDim.x) {
            acc += expf(A[i * N + j]);
        }
        block_sums[tidx] = acc;
        __syncthreads();

        for (int k = BLOCK_SIZE / 2; k > 16; k >>= 1) {
            if (tidx < k) {
                block_sums[tidx] += block_sums[tidx + k];
            }
            __syncthreads();
        }

        if (tidx < 32) {
            float val = block_sums[tidx];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                val += __shfl_down_sync(0xffffffff, val, off);
            }
            if (tidx == 0) {
                output[i] = logf(val) - A[i * N + label[i]];
            }
        }
    }
}

template <int BLOCK_SIZE>
__global__ void sum_reduction(float *out, const float *input, int N) {
    __shared__ float sum[BLOCK_SIZE];
    const int tidx = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tidx;
    const int stride = blockDim.x * gridDim.x;

    float acc = 0.0f;
    for (int i = idx; i < N; i += stride) {
        acc += input[i];
    }
    sum[tidx] = acc;
    __syncthreads();

    for (int i = BLOCK_SIZE / 2; i > 16; i >>= 1) {
        if (tidx < i) {
            sum[tidx] += sum[tidx + i];
        }
        __syncthreads();
    }

    if (tidx < 32) {
        float val = sum[tidx];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, off);
        }
        if (tidx == 0) {
            atomicAdd(out, val / N);
        }
    }
}

// logits, true_labels, loss are device pointers
extern "C" void solve(const float* logits, const int* true_labels, float* loss, int N, int C) {
    const int BLOCK_SIZE = 512;
    const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float *partial_sums;
    cudaMalloc(&partial_sums, sizeof(float) * N);
    cudaMemset(loss, 0, sizeof(float));
    partialLossKernel<512><<<GRID_SIZE, BLOCK_SIZE>>>(partial_sums, logits, true_labels, N, C);
    sum_reduction<512><<<GRID_SIZE, BLOCK_SIZE>>>(loss, partial_sums, N);
    cudaFree(partial_sums);
}
