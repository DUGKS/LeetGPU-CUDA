#include <cuda_runtime.h>

template<int BLOCK_SIZE>
__global__ void maxPoolingKernel(float *output, const float *input,
                                 int kernel_size, int padding, int stride,
                                 int N, int C, int H, int W, int H_out, int W_out) {
    const int tidx = threadIdx.x;
    const int gidx = threadIdx.x + blockIdx.x * BLOCK_SIZE;

    if (gidx >= N * C * H_out * W_out) {
        return;
    }

    const int w_idx_out = gidx % W_out;
    const int tmp = gidx / W_out;
    const int h_idx_out = tmp % H_out;
    const int nc = tmp / H_out;

    const int w_idx_start = w_idx_out * stride - padding;
    const int h_idx_start = h_idx_out * stride - padding;

    float acc = -INFINITY;

    for (int kh = 0; kh < kernel_size; kh++) {
        int h_in = h_idx_start + kh;
        for (int kw = 0; kw < kernel_size; kw++) {
            const int w_in = w_idx_start + kw;
            bool valid = h_in >= 0 && h_in < H && w_in >= 0 && w_in < W;
            float val = valid ? input[(nc * H + h_in) * W + w_in] : -INFINITY;
            acc = acc < val ? val : acc;
        }
    }
        output[gidx] = acc;
}
// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N, int C, int H, int W,
                      int kernel_size, int stride, int padding) {
    const int BLOCK_SIZE = 1024;
    const int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    const int W_out = (W + 2 * padding - kernel_size) / stride + 1;
    const int grid_size = (N * C * H_out * W_out + BLOCK_SIZE - 1) / BLOCK_SIZE;
    maxPoolingKernel<BLOCK_SIZE><<<grid_size, BLOCK_SIZE>>>(output, input, kernel_size, padding, stride,
    N, C, H, W, H_out, W_out);
                      }