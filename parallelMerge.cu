#include <cuda_runtime.h>

__device__ __forceinline__ int mergePath(const float* A, int M, const float* B, int N, long long k) {
    int lo = (int)max(0LL, k - N);
    int hi = (int)min(k, (long long)M);
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (A[mid] <= B[k - 1 - mid]) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

template <int ITEMS>
__global__ void mergeKernel(float *C, const float *A, int M, const float *B, int N) {
    const long long total = M + N;
    const long long k0 = (blockIdx.x * blockDim.x + threadIdx.x) * ITEMS;
    if (k0 >= total) return;
    const long long k1 = min(k0 + ITEMS, total);

    int i = mergePath(A, M, B, N, k0);
    int j = (int)(k0 - i);


    for (long long k = k0; k < k1; k++) {
        float a = (i < M) ? A[i] : 0.0f;
        float b = (j < N) ? B[j] : 0.0f;
        bool takeA = (j >= N) || (i < M && a <= b);
        if (takeA) {
            C[k] = a;
            ++i;
        } else {
            C[k] = b;
            ++j;
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
    constexpr int ITEMS = 16;
    constexpr int block_size = 1024;
    const int grid_size = ((M + N) + block_size * ITEMS - 1) / (block_size * ITEMS);
    mergeKernel<ITEMS><<<grid_size, block_size>>>(C, A, M, B, N);
}