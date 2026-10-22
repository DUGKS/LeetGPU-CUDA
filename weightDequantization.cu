#include <cuda_runtime.h>

__global__ void weightDequantization(float *output, const float *input, const float *s_ptr, int M, int N, int TILE_SIZE) {
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

    const int gidx = blockIdx.x * TILE_SIZE;
    const int gidy = blockIdx.y * TILE_SIZE;
    float alpha = s_ptr[blockIdx.y * gridDim.x + blockIdx.x];

    for (int r = tidy; r < TILE_SIZE; r += blockDim.y) {
        const int row = r + gidy;
        for (int c = tidx; c < TILE_SIZE; c += blockDim.x) {
            const int col = c + gidx;
            if (col < N && row < M) {
                const int idx = row * N + col;
                output[idx] = input[idx] * alpha;
            }
        }
    }
}

// X, S, Y are device pointers
extern "C" void solve(const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE) {
    const int block_x = 16;
    const int block_y = 16;
    dim3 block_size{block_x, block_y};
    dim3 grid_size{(N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE};
    weightDequantization<<<grid_size, block_size>>>(Y, X, S, M, N, TILE_SIZE);
}