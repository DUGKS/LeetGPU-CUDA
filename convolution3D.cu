#include <cuda_runtime.h>

template<int TILE_SIZE = 8>
__global__ void convolution3DKernel(float *output, const float *input, const float *kernel,
                                    int input_depth, int input_rows, int input_cols,
                                    int kernel_depth, int kernel_rows, int kernel_cols) {
    const int output_depth = input_depth - kernel_depth + 1;
    const int output_rows = input_rows - kernel_rows + 1;
    const int output_cols = input_cols - kernel_cols + 1;

    const int base_depth = blockIdx.z * TILE_SIZE;
    const int base_row = blockIdx.y * TILE_SIZE;
    const int base_col = blockIdx.x * TILE_SIZE;
    const int tz = threadIdx.z;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    const int num_kernel = kernel_depth * kernel_rows * kernel_cols;
    const int num_threads = blockDim.z * blockDim.y * blockDim.x;

    extern __shared__ float sm[];
    float *shared_kernel = sm;
    float *tile_input = sm + num_kernel;

    for (int k = tz; k < kernel_depth; k += blockDim.z) {
        for (int j = ty; j < kernel_rows; j += blockDim.y) {
            for (int i = tx; i < kernel_cols; i += blockDim.x) {
                const int tid = k * kernel_rows * kernel_cols + j * kernel_cols + i;
                shared_kernel[tid] = kernel[tid];
            }
        }
    }
    __syncthreads();

    const int tile_input_depth = TILE_SIZE + kernel_depth - 1;
    const int tile_input_rows = TILE_SIZE + kernel_rows - 1;
    const int tile_input_cols = TILE_SIZE + kernel_cols - 1;
    for (int k = tz; k < tile_input_depth; k += blockDim.z) {
        for (int j = ty; j < tile_input_rows; j += blockDim.y) {
            for (int i = tx; i < tile_input_cols; i += blockDim.x) {
                int tile_index = k * tile_input_rows * tile_input_cols +  j * tile_input_cols + i;
                const int depth = base_depth + k;
                const int row = base_row + j;
                const int col = base_col + i;
                const int input_idx = depth * input_rows * input_cols + row * input_cols + col;
                tile_input[tile_index] = (depth < input_depth && row < input_rows && col < input_cols) ?
                input[input_idx] : 0.0f;
            }
        }
    }
    __syncthreads();

    for (int k = tz; k < TILE_SIZE; k += blockDim.z) {
        for (int j = ty; j < TILE_SIZE; j += blockDim.y) {
            for (int i = tx; i < TILE_SIZE; i += blockDim.x) {
                const int depth = base_depth + k;
                const int row = base_row + j;
                const int col = base_col + i;
                const int output_idx = depth * output_rows * output_cols + row * output_cols + col;
                float sum = 0.0f;
                for (int k_kernel = 0; k_kernel < kernel_depth; k_kernel++) {
                    for (int j_kernel = 0; j_kernel < kernel_rows; j_kernel++) {
                        for (int i_kernel = 0; i_kernel < kernel_cols; i_kernel++) {
                            int kernel_idx = k_kernel * kernel_rows * kernel_cols + j_kernel * kernel_cols + i_kernel;
                            int input_idx = (k + k_kernel) * tile_input_rows * tile_input_cols + (j + j_kernel) * tile_input_cols + i + i_kernel;
                            sum += shared_kernel[kernel_idx] * tile_input[input_idx];
                        }
                    }
                }
                output[output_idx] = sum;
            }
        }
    }
}