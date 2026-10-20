#include <cuda_runtime.h>

template<int BLOCK_X, int BLOCK_Y>
__global__ void JacobiStencil(float *output, const float *input, int rows, int cols) {
    __shared__ float sdata[BLOCK_Y + 2][BLOCK_X + 2];

    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int gidx = blockIdx.x * BLOCK_X + tidx;
    const int gidy = blockIdx.y * BLOCK_Y + tidy;

    const int base_col = blockIdx.x * BLOCK_X;
    const int base_row = blockIdx.y * BLOCK_Y;


    for (int j = tidy; j < BLOCK_Y + 2; j += blockDim.y) {
        const int row = base_row - 1 + j;
        for (int i = tidx; i < BLOCK_X + 2; i += blockDim.x) {
            const int col = base_col - 1 + i;
            bool in_bound = row >= 0 && row < rows && col >= 0 && col < cols;
            sdata[j][i] = in_bound ? input[row * cols + col] : 0.0;
        }
    }
    __syncthreads();

    if (gidx < cols && gidy < rows) {
    float result;
    bool interior = gidx > 0 && gidx < cols - 1 && gidy > 0 && gidy < rows - 1;
    if (interior) {
        result = 0.25f * (sdata[tidy + 1][tidx] + sdata[tidy + 1][tidx + 2] + sdata[tidy][tidx + 1] + sdata[tidy + 2][tidx + 1]);
    } else {
        result = sdata[tidy + 1][tidx + 1];
    }

    output[gidy * cols + gidx] = result;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    constexpr int BLOCK_X = 64;
    constexpr int BLOCK_Y = 16;

    dim3 block_size{BLOCK_X, BLOCK_Y};
    dim3 grid_size{(cols + BLOCK_X - 1) / BLOCK_X, (rows + BLOCK_Y - 1) / BLOCK_Y};
    JacobiStencil<BLOCK_X, BLOCK_Y><<<grid_size, block_size>>>(output, input, rows, cols);
}
