#include <cuda_runtime.h>

template <int BLOCK_SIZE>
__global__ void meanSquareError(float *output, const float *pre, const float *targets, int N) {
    __shared__ float warp_sums[BLOCK_SIZE / 32];

    const int tidx = threadIdx.x;
    const int gidx = BLOCK_SIZE * blockIdx.x + tidx;
    const int stride = blockDim.x * gridDim.x;

    float acc = 0.0f;
    for (int i = gidx; i < N; i += stride) {
        float val = pre[i] - targets[i];
        acc += val * val;
    }

    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, off);
    }

    int warp_idx =  tidx >> 5;
    int lane = tidx & 31;
    if (lane == 0) warp_sums[warp_idx] = acc;
    __syncthreads();

    if (tidx < 32) {
        float val = lane < BLOCK_SIZE / 32 ? warp_sums[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, off);
        }
        if (lane == 0) { atomicAdd(output, val / N); }
    }
}

// predictions, targets, mse are device pointers
extern "C" void solve(const float* predictions, const float* targets, float* mse, int N) {
    constexpr int BLOCK_SIZE = 1024;
    const int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    meanSquareError<BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(mse, predictions, targets, N);
}
