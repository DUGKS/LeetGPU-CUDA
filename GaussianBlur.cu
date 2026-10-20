#include <cuda_runtime.h>

template<int TILE_SIZE>
__global__ void GaussianBlurKernel(float *output, const float *input, const float *kernel, int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    extern __shared__ float tile[];

    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

    const int halo_rows = kernel_rows / 2;
    const int halo_cols = kernel_cols / 2;

    const int tile_rows = TILE_SIZE + 2 * halo_rows;
    const int tile_cols = TILE_SIZE + 2 * halo_cols;

    const int row_start = blockIdx.y * TILE_SIZE;
    const int col_start = blockIdx.x * TILE_SIZE;

        const int row_low = row_start - halo_rows;
        const int col_low = col_start - halo_cols;

        for (int jy = tidy; jy < tile_rows; jy += blockDim.y) {
            const int row = row_low + jy;
            bool row_valid = (row >= 0) && (row < input_rows);
            for (int ix = tidx; ix < tile_cols; ix += blockDim.x) {
                const int col = col_low + ix;
                bool valid = row_valid && (col >= 0 && col < input_cols);
                float value = valid ? input[row * input_cols + col]: 0.0f;
                tile[jy * tile_cols + ix] = value;
            }
        }
        __syncthreads();

        for (int jy = tidy; jy < TILE_SIZE; jy += blockDim.y) {
            const int row = row_start + jy;
            for (int ix = tidx; ix < TILE_SIZE; ix += blockDim.x) {
                const int col = col_start + ix;
                if (row < input_rows && col < input_cols) {
                    float acc = 0.0f;
                    for (int ky = 0; ky < kernel_rows; ky++) {
                        const int tile_y = jy + ky;
                        for (int kx = 0; kx < kernel_cols; kx++) {
                            const int tile_x = ix + kx;
                            acc += kernel[ky * kernel_cols + kx] * tile[tile_y * tile_cols + tile_x];
                        }
                    }
                    output[row * input_cols + col] = acc;
                }
            }
        }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) {
    const int tile_size = 16;
    dim3 block_size{tile_size, tile_size};
    dim3 grid_size{(input_cols + tile_size - 1) / tile_size, (input_rows + tile_size - 1) / tile_size};
    const int num_shared_mem = (tile_size + kernel_rows / 2 * 2) * (tile_size + kernel_cols / 2 * 2) * sizeof(float);
    GaussianBlurKernel<16><<<grid_size, block_size, num_shared_mem>>>(output, input, kernel, input_rows, input_cols, kernel_rows, kernel_cols);
}