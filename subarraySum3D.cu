#include <cuda_runtime.h>

template<int BLOCK_X, int BLOCK_Y, int BLOCK_Z>
__global__ void subarraySumKernel(int *output, const int *input, int N, int M, int K, int S_DEP, int length_depth, int S_ROW, int length_row, int S_COL, int length_col) {
    constexpr int BLOCK_SIZE = BLOCK_X * BLOCK_Y * BLOCK_Z;
    __shared__ int warp_sum[BLOCK_SIZE / 32];

    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int tidz = threadIdx.z;
    const int tid = (tidz * BLOCK_Y + tidy) * BLOCK_X + tidx;
    const int gidx = blockIdx.x * blockDim.x + tidx;
    const int gidy = blockIdx.y * blockDim.y + tidy;
    const int gidz = blockIdx.z * blockDim.z + tidz;
    const int stride_x = blockDim.x * gridDim.x;
    const int stride_y = blockDim.y * gridDim.y;
    const int stride_z = blockDim.z * gridDim.z;

    int acc = 0;
    for (int k = gidz; k < length_depth; k += stride_z) {
        const int depth = S_DEP + k;
        for (int j = gidy; j < length_row; j += stride_y) {
            const int row = S_ROW + j;
            for (int i = gidx; i < length_col; i += stride_x) {
                const int col = S_COL + i;
                acc += input[(depth * M + row) * K + col];
            }
        }
    }

    #pragma unroll
    for(int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, off);
    }

    const int warp_idx = tid / 32;
    const int lane = tid % 32;
    if (lane == 0)
        warp_sum[warp_idx] = acc;
    __syncthreads();

    if (tid < 32) {
        int val = tid < BLOCK_SIZE / 32 ? warp_sum[tid] : 0;
        #pragma unroll
        for (int i = 16; i > 0; i >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, i);
        }
        if (tid == 0) {
            atomicAdd(output, val);
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int K, int S_DEP, int E_DEP, int S_ROW, int E_ROW, int S_COL, int E_COL) {
    const int length_depth = E_DEP - S_DEP + 1;
    const int length_row = E_ROW - S_ROW + 1;
    const int length_col = E_COL - S_COL + 1;
    constexpr int BLOCK_X = 16;
    constexpr int BLOCK_Y = 8;
    constexpr int BLOCK_Z = 8;
    dim3 block_size{BLOCK_X, BLOCK_Y, BLOCK_Z};
    dim3 grid_size{(length_col + BLOCK_X - 1) / BLOCK_X, (length_row + BLOCK_Y - 1) / BLOCK_Y, (length_depth + BLOCK_Z - 1) / BLOCK_Z};
    subarraySumKernel<BLOCK_X, BLOCK_Y, BLOCK_Z><<<grid_size, block_size>>>(output, input, N, M, K, S_DEP, length_depth, S_ROW, length_row, S_COL, length_col);
}