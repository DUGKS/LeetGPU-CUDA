#include <cuda_runtime.h>

template<int BLOCK_SIZE>
__global__ void countArray(int *output, const int *input, int N, int p, bool first) {
    __shared__ int sdata[BLOCK_SIZE];

    const int tidx = threadIdx.x;
    const int gidx = blockDim.x * blockIdx.x + tidx;
    const int stride = blockDim.x * gridDim.x;

    int cnt = 0;
    for (int i = gidx; i < N; i += stride) {
        cnt += first ? static_cast<int>(input[i] == p) : input[i];
    }
    sdata[tidx] = cnt;
    __syncthreads();

    for (int i = BLOCK_SIZE / 2; i > 16; i >>= 1) {
        if (tidx < i) {
            sdata[tidx] += sdata[tidx + i];
        }
        __syncthreads();
    }

    if (tidx < 32) {
        int val = sdata[tidx];
        for (int off = 16; off > 0; off >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, off);
        }
        if (tidx == 0) {
            output[blockIdx.x] = val;
        }
    }
}

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int K) {
    const int BLOCK_SIZE = 1024;
    const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int *partial_cnt[2] = {nullptr, nullptr};
    cudaMalloc(&partial_cnt[0], sizeof(int) * GRID_SIZE);
    cudaMalloc(&partial_cnt[1], sizeof(int) * GRID_SIZE);
    cudaMemset(partial_cnt[0], 0x0, sizeof(int) * GRID_SIZE);
    cudaMemset(partial_cnt[1], 0x0, sizeof(int) * GRID_SIZE);

    int curr_num = N;
    const int *curr_input = input;
    int which = 0;
    bool first = true;
    while (curr_num > BLOCK_SIZE) {
        const int grid_size = (curr_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
        countArray<BLOCK_SIZE><<<grid_size, BLOCK_SIZE>>>(partial_cnt[which], curr_input, curr_num, K, first);
        first = false;
        curr_num = grid_size;
        curr_input = partial_cnt[which];
        which ^= 1;
    }

    countArray<BLOCK_SIZE><<<1, BLOCK_SIZE>>>(output, curr_input, curr_num, K, first);

    cudaFree(partial_cnt[0]);
    cudaFree(partial_cnt[1]);
}