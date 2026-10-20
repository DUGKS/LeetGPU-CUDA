#include <cuda_runtime.h>
#include <cassert>
#include <cfloat>

__device__ float dist(const float3& a, const float3 &b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    const float dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

template<int BLOCK_SIZE>
__global__ void nearestNeighborKernel(int *output, const float3 *input, int N) {
    assert(BLOCK_SIZE == blockDim.x);

    const int tidx = threadIdx.x;
    const int gidx = blockDim.x * blockIdx.x + tidx;

    __shared__ float3 data[BLOCK_SIZE];

    float3 p = make_float3(0.0f, 0.0f, 0.0f);
    if (gidx < N) p = input[gidx];

    float best = FLT_MAX;
    int best_idx = -1;

    for (int base = 0; base < N; base += blockDim.x) {
        const int input_idx = base + tidx;
        if (input_idx < N) data[tidx] = input[input_idx];
        __syncthreads();

        if (gidx < N) {
            int count = min(BLOCK_SIZE, N - base);
            for (int k = 0; k < count;k++) {
                float dist_val = dist(p, data[k]);
                if ((k + base) == gidx) continue;
                if (dist_val < best) {
                    best = dist_val;
                    best_idx = k + base;
                }
            }
        }
        __syncthreads();
    }
    if (gidx < N) output[gidx] = best_idx;
}
// points and indices are device pointers
extern "C" void solve(const float* points, int* indices, int N) {
    const float3 *input = reinterpret_cast<const float3 *>(points);
    constexpr int BLOCK_SIZE = 1024;
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    nearestNeighborKernel<BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(indices, input, N);
}
