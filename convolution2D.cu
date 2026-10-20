#include <cuda_runtime.h>

template<int TILE_ROW = 32, int TILE_COL = 32>
__global__ void convolution2DKernel(float * __restrict__ output,
                                    const float * __restrict__ input,
                                    const float * __restrict__ kernel,
                                    int input_row, int input_col,
                                    int kernel_row, int kernel_col) {
    const int base_row = blockIdx.y * TILE_ROW;
    const int base_col = blockIdx.x * TILE_COL;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int num_kernels = kernel_row * kernel_col;

    extern __shared__ float sm[];
    float *shared_kernel = sm;
    float *tile = sm + num_kernels;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int i = tid; i < num_kernels; i += blockDim.x * blockDim.y) {
        shared_kernel[i] = kernel[i];
    }
    __syncthreads();

    const int TILE_Y = TILE_ROW + kernel_row - 1;
    const int TILE_X = TILE_COL + kernel_col - 1;

    for (int j = ty; j < TILE_Y; j += blockDim.y) {
        const int row = base_row + j;
        for (int i = tx; i < TILE_X; i += blockDim.x) {
            const int col = base_col + i;
            tile[j * TILE_X + i] = (row < input_row && col < input_col) ? input[row * input_col + col] : 0.0f;
        }
    }
    __syncthreads();

    const int output_row = input_row - kernel_row + 1;
    const int output_col = input_col - kernel_col + 1;
    for (int j = ty; j < TILE_ROW; j += blockDim.y) {
        const int row = base_row + j;
        for (int i = tx; i < TILE_COL; i += blockDim.x) {
            const int col = base_col + i;
            if (row < output_row && col < output_col) {
                float sum = 0.0f;
                for (int kj = 0; kj < kernel_row; kj++) {
                    for (int ki = 0; ki < kernel_col; ki++) {
                        sum += shared_kernel[kj * kernel_col + ki] * tile[(j + kj) * TILE_X + i + ki];
                    }
                }
                output[row * output_col + col] = sum;
            }
        }
    }
}