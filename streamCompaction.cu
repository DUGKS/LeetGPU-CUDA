#include <cuda_runtime.h>

template<int BLOCK_SIZE>
__global__ void countKernel(int *block_counts, const float *A, int N) {
    __shared__ int warp_cnt[BLOCK_SIZE / 32];
    const int tidx = threadIdx.x;
    const int gidx = blockIdx.x * BLOCK_SIZE + tidx;
    int flag = ((gidx < N) && (A[gidx] > 0.0f));

    unsigned ballot = __ballot_sync(0xffffffff, flag);
    const int lane = tidx & 31;
    const int warp_idx = tidx >> 5;
    if (lane == 0) warp_cnt[warp_idx] = __popc(ballot);
    __syncthreads();

    if (tidx == 0) {
        int sum = 0;
        for (int i = 0;i < BLOCK_SIZE / 32; i++) {
            sum += warp_cnt[i];
        }
        block_counts[blockIdx.x] = sum;
    }
}

__global__ void prefixBlockSum(int *block_counts, int num_blocks) {
    __shared__ int sdata[1024];
    const int tidx = threadIdx.x;
    const int chunk = (num_blocks + 1023) / 1024;
    const int start = tidx * chunk;
    const int end = min(start + chunk, num_blocks);

    int acc = 0;
    for (int i = start; i < end; i++) {
        acc += block_counts[i];
    }
    sdata[tidx] = acc;
    __syncthreads();

    for (int i = 1; i < 1024; i <<= 1) {
        int val = tidx >= i ? sdata[tidx - i] : 0;
        __syncthreads();
        sdata[tidx] += val;
        __syncthreads();
    }

    int run = sdata[tidx] - acc;
    for (int i = start; i < end; i++) {
        int c = block_counts[i];
        block_counts[i] = run;
        run += c;
    }
}

template<int BLOCK_SIZE>
__global__ void scatterKernel(float *out, const float *A, const int *block_offset, int N) {
    __shared__ int warp_start[BLOCK_SIZE / 32];

    const int tidx = threadIdx.x;
    const int gidx = BLOCK_SIZE * blockIdx.x + tidx;
    float v = gidx < N ? A[gidx] : 0.0f;
    int flag = (gidx < N) && (v > 0.0f);

    unsigned ballot = __ballot_sync(0xffffffff, flag);
    const int lane = tidx & 31;
    const int warp_idx = tidx >> 5;
    const int rank = __popc(ballot & ((1u << lane) - 1));
    if (lane == 0) warp_start[warp_idx] = __popc(ballot);
    __syncthreads();

    if (tidx == 0) {
        int run = 0;
        for (int w = 0; w < BLOCK_SIZE / 32; w++) {
            int c = warp_start[w];
            warp_start[w] = run;
            run += c;
        }
    }
    __syncthreads();

    int out_idx = block_offset[blockIdx.x] + warp_start[warp_idx] + rank;
    if (flag) {
        out[out_idx] = v;
    }
}

// A, out are device pointers
extern "C" void solve(const float* A, int N, float* out) {
    constexpr int BLOCK_SIZE = 1024;
    const int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int *block_counts;
    cudaMalloc(&block_counts, sizeof(int) * grid);
    countKernel<BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(block_counts, A, N);
    prefixBlockSum<<<1, BLOCK_SIZE>>>(block_counts, grid);
    scatterKernel<BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(out, A, block_counts, N);
}
