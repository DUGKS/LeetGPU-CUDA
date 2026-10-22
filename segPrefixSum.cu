#include <cuda_runtime.h>

constexpr int BLOCK = 256;
constexpr int ITEMS = 4;
constexpr int TILE = BLOCK * ITEMS;

struct Seg {int flag; float value;};

__device__ __forceinline__ Seg comb(Seg a, Seg b) {
    Seg r;
    r.flag = a.flag | b.flag;
    r.value = b.flag ? b.value : a.value + b.value;
    return r;
}

__device__ __forceinline__ Seg __shfl_up_seg(Seg x, int d) {
    Seg r;
    r.flag = __shfl_up_sync(0xffffffff, x.flag, d);
    r.value = __shfl_up_sync(0xffffffff, x.value, d);
    return r;
}

__device__ Seg blockScanExclusive(Seg x, Seg *warp_total, Seg &total) {
    const Seg zero = {0, 0.0f};
    int lane = threadIdx.x & 31;
    int warp_idx = threadIdx.x >> 5;
    int warp_num = blockDim.x >> 5;

    Seg incl_value = x;
    #pragma unroll
    for (int i = 1; i < 32; i <<= 1) {
        Seg v = __shfl_up_seg(incl_value, i);
        if (lane >= i) {
            incl_value = comb(v, incl_value);
        }
    }
    Seg excl_value = __shfl_up_seg(incl_value, 1);
    if (lane == 0) {excl_value = zero; }

    if (lane == 31) {
        warp_total[warp_idx] = incl_value;
    }
    __syncthreads();

    if (warp_idx == 0) {
        Seg v = lane < warp_num ? warp_total[lane] : zero;
        for (int i = 1; i < 32; i <<= 1) {
            Seg tmp = __shfl_up_seg(v, i);
            if (lane >= i) v = comb(tmp, v);
        }
        if (lane < warp_num) {
            warp_total[lane] = v;
        }
    }
    __syncthreads();
    total = warp_total[warp_num - 1];
    Seg val = warp_idx > 0 ? warp_total[warp_idx - 1] : zero;
    return comb(val ,excl_value);
}

__device__ __forceinline__ void load4(const float *vals, const int *flags, int N,
                                      long long base, float v[4], int f[4]) {
    if (base + 3 < N) {
        float4 a = *reinterpret_cast<const float4*>(vals + base);
        int4 b = *reinterpret_cast<const int4*>(flags + base);
        v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
        f[0] = b.x; f[1] = b.y; f[2] = b.z; f[3] = b.w;
    } else {
        for (int k = 0; k < 4; k++) {
            bool ok = base + k < N;
            v[k] = ok ? vals[base + k] : 0.0f;
            f[k] = ok ? flags[base + k] : 0;
        }
    }
}

__device__ __forceinline__ Seg threadAggregate(const float v[4], const int f[4]) {
    Seg a = {0, 0.0f};
    for (int i = 0; i < 4; i++) {
        Seg tmp = {f[i], v[i]};
        a = comb(a, tmp);
    }
    return a;
}

__global__ void blockAggregateKernel(Seg* agg, const float *vals, const int *flags, int N) {
    __shared__ Seg warp_total[32];
    float v[4]; int f[4];
    int base = blockIdx.x * TILE + threadIdx.x * ITEMS;
    load4(vals, flags, N, base, v, f);
    Seg total;
    blockScanExclusive(threadAggregate(v, f), warp_total, total);
    if (threadIdx.x == 0) { agg[blockIdx.x] = total; }
}

__global__ void carryKernel(float *carry, const Seg* agg, int N) {
    __shared__ Seg warp_total[32];
    int items = (N + blockDim.x - 1) / blockDim.x;
    int start = threadIdx.x * items;
    int end = min(start + items, N);
    Seg a = {0, 0.0f};
    for (int i = start; i < end; i++) {
        a = comb(a, agg[i]);
    }
    Seg total;
    Seg run = blockScanExclusive(a, warp_total, total);
    for (int i = start; i < end; i++) {
        carry[i] = run.value;
        run = comb(run, agg[i]);
    }
}

__global__ void offsetKernel(float *out, const float *vals, const int *flags, int N, const float* carry) {
    __shared__ Seg warp_total[32];
    int base = blockIdx.x * TILE + threadIdx.x * ITEMS;
    float v[4];
    int f[4];
    load4(vals, flags, N, base, v, f);
    Seg total;
    Seg pre_threads = blockScanExclusive(threadAggregate(v, f), warp_total, total);
    Seg pre_blocks = {0, carry[blockIdx.x]};
    float run = comb(pre_blocks, pre_threads).value;

    float o[4];
    for (int k = 0; k < 4;k++) {
        o[k] = f[k] ? 0.0f : run;
        run = f[k] ? v[k]: run + v[k];
    }

    if (base + 3 < N) {
        *reinterpret_cast<float4 *>(out + base) = make_float4(o[0], o[1], o[2], o[3]);
    } else {
        for (int i = base; i < N; i++) {
            out[i] = o[i - base];
        }
    }
}
// values, flags, output are device pointers
extern "C" void solve(const float* values, const int* flags, float* output, int N) {
    int block_nums = (N + TILE - 1) / TILE;
    Seg *agg;
    cudaMalloc(&agg, sizeof(Seg) * block_nums);
    float *carry;
    cudaMalloc(&carry, sizeof(float) * block_nums);

    blockAggregateKernel<<<block_nums, BLOCK>>>(agg, values, flags, N);
    carryKernel<<<1, 1024>>>(carry, agg, block_nums);
    offsetKernel<<<block_nums, BLOCK>>>(output, values, flags, N, carry);

    cudaDeviceSynchronize();
    cudaFree(agg);
    cudaFree(carry);
}