#include <cuda_runtime.h>

__device__ __forceinline__ float blockScanInclusive(float x, float *warp_sum, float &total) {
    const int tidx = threadIdx.x;
    int warp_idx = tidx >> 5;
    int warp_num = blockDim.x >> 5;
    int lane = tidx & 31;

    #pragma unroll
    for (int i = 1; i < 32; i <<= 1) {
        float o = __shfl_up_sync(0xffffffff, x, i);
        if (lane >= i) x += o;
    }

    if (lane == 31) {
        warp_sum[warp_idx] = x;
    }
    __syncthreads();

    if (warp_idx == 0) {
        float val = lane < warp_num ? warp_sum[lane] : 0.0f;
        for (int i = 1;i < 32;i <<= 1) {
            float val_ = __shfl_up_sync(0xffffffff, val, i);
            if (lane >= i) val += val_;
        }
        if (lane < warp_num) warp_sum[lane] = val;
    }
    __syncthreads();

    float res = warp_idx > 0 ? x + warp_sum[warp_idx - 1] : x;
    total = warp_sum[warp_num - 1];
    __syncthreads();
    return res;
}

__global__ void rowScanKernel(float *output, const float *input, int H, int W) {
    __shared__ float warp_sum[32];
    const int tidx = threadIdx.x;

    int row = blockIdx.x;
    int base = row * W;
    float carry = 0.0f;

    for (int start = 0; start < W; start += blockDim.x) {
        int col = start + tidx;
        float x = col < W ? input[base + col] : 0.0f;
        float total;
        float s = blockScanInclusive(x, warp_sum, total);
        if (col < W) output[base + col] = s + carry;
        carry +=  total;
    }
}

__global__ void colScanKernel(float *output, int H, int W) {
    const int tidx = threadIdx.x;
    int c = blockDim.x * blockIdx.x + tidx;
    if (c >= W) return;
    float acc = 0.0f;
    for (int r = 0; r < H; ++r) {
        acc += output[r * W + c];
        output[r * W + c] = acc;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int H, int W) {
    constexpr int block_size = 32;
    const int grid = (W + block_size - 1) / block_size;
    rowScanKernel<<<H, 1024>>>(output, input, H, W);
    colScanKernel<<<grid, block_size>>>(output, H, W);
    cudaDeviceSynchronize();
}
