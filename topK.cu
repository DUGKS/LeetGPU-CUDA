#include <cuda_runtime.h>
#include <cassert>

__device__ inline void swap(float &a, float &b) {
    float tmp = a;
    a = b;
    b = tmp;
}

template<int BLOCK_SIZE>
__global__ void topKBlock(float * __restrict__ intermediate_vals,
                          const float * __restrict__ input, int N, int K) {
    __shared__ float data[BLOCK_SIZE];
    const int tidx = threadIdx.x;
    const int gidx = blockIdx.x * BLOCK_SIZE + tidx;

    data[tidx] = gidx < N ? input[gidx] : -1E30f;
    __syncthreads();

    for (int size = 2; size <= BLOCK_SIZE; size <<= 1) {
        for (int stride = size / 2; stride > 0; stride >>= 1) {
            int pos = tidx ^ stride;
            if (pos > tidx) {
                bool ascending = ((tidx & size) == 0);
                if ((ascending && data[tidx] > data[pos]) ||
                    (!ascending && data[tidx] < data[pos])) {
                        swap(data[tidx], data[pos]);
                    }
            }
            __syncthreads();
        }
    }

    if (tidx < K) {
        intermediate_vals[blockIdx.x * K + tidx] = data[BLOCK_SIZE - tidx - 1];
    }
}

template<int BLOCK_SIZE>
__global__ void topK(float * __restrict__ output, const float *input, int N, int K) {
    assert(N <= BLOCK_SIZE);
    __shared__ float data[BLOCK_SIZE];
    const int tidx = threadIdx.x;
    data[tidx] = tidx < N ? input[tidx] : -1E30f;
    __syncthreads();

    for (int size = 2; size <= BLOCK_SIZE; size <<= 1) {
        for (int stride = size / 2; stride > 0; stride >>= 1) {
            int pos = tidx ^ stride;
            if (pos > tidx) {
                bool ascending = ((tidx & size) == 0);
                if ((ascending && data[tidx] > data[pos]) ||
                    (!ascending && data[tidx] < data[pos])) {
                        swap(data[tidx], data[pos]);
                    }
            }
            __syncthreads();
        }
    }

    if (tidx < K) {
        output[tidx] = data[BLOCK_SIZE - 1 - tidx];
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int k) {
    const int BLOCK_SIZE = 1024;

    int curr_N = N;
    const float *curr_input = input;
    float *buffers[2] = {nullptr, nullptr};

    if (curr_N > BLOCK_SIZE) {
        int GRID_SIZE = (curr_N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        cudaMalloc(&buffers[0], GRID_SIZE * k * sizeof(float));
        cudaMalloc(&buffers[1], GRID_SIZE * k * sizeof(float));
    }
    int which = 0;

    while (curr_N > BLOCK_SIZE) {
        int GRID_SIZE = (curr_N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        topKBlock<BLOCK_SIZE><<<GRID_SIZE, BLOCK_SIZE>>>(buffers[which], curr_input, N, k);
        curr_input = buffers[which];
        curr_N = GRID_SIZE * k;
        which ^= 1;
    }

    topK<BLOCK_SIZE><<<1, BLOCK_SIZE>>>(output, curr_input, curr_N, k);
    cudaFree(&buffers[0]);
    cudaFree(&buffers[1]);
}