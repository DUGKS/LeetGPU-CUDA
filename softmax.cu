#include <cuda_runtime.h>

__device__ inline void atomicMaxFloat(float *address, float val) {
    int *address_as_int = reinterpret_cast<int*>(address);
    int old = __float_as_int(*address);
    int assumed;
    do {
        assumed = old;
        int new_value = __float_as_int(fmaxf(__int_as_float(assumed), val));
        old = atomicCAS(address_as_int, assumed, new_value);

    } while (assumed != old);
}

__global__ void findMaxKernel(float *max, const float *input, int N) {
    extern  __shared__ float sdata[];

    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int tid = threadIdx.x;

    float local_max = -INFINITY;
    for (int i = idx; i < N; i += stride) {
        local_max = fmaxf(local_max, input[i]);
    }
    sdata[tid] = local_max;
    __syncthreads();

    for (int i = blockDim.x / 2; i >= 32; i >>= 1) {
        if (tid < i) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + i]);
        }
        __syncthreads();
    }

    if (tid < 32) {
        float val = sdata[tid];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        }
        if (tid == 0) {
            atomicMaxFloat(max, val);
        }
    }
}

__global__ void expSumKernel(float *sum, const float *input, int N, float *max_val_ptr) {
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    float max_val = *max_val_ptr;

    float local_sum = 0.0f;
    for (int i = idx; i < N; i += stride) {
        local_sum += expf(input[i] - max_val);
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (int i = blockDim.x / 2; i >= 32; i >>= 1) {
        if (tid < i) {
            sdata[tid] += sdata[tid + i];
        }
        __syncthreads();
    }

    if (tid < 32) {
        float val = sdata[tid];
        #pragma unroll
        for (int i = 16; i > 0; i >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, i);
        }
        if (tid == 0) {
            atomicAdd(sum, val);
        }
    }
}

__global__ void fractionKernel(float *out, const float *input, int N, float *max_val_ptr, float *sum_ptr) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    float max_val = *max_val_ptr;
    float inverse_sum = 1.0f / (*sum_ptr);
    for (int i = idx; i < N; i += stride) {
        out[i] = expf(input[i] - max_val) * inverse_sum;
    }
}


// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    int sharedMemSize = threadsPerBlock * sizeof(float);

    float host_max = -INFINITY;
    float host_sum = 0.0f;
    float *device_max, *device_sum;
    cudaMalloc(&device_max, sizeof(float));
    cudaMalloc(&device_sum, sizeof(float));
    cudaMemcpy(device_max, &host_max, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_sum, &host_sum, sizeof(float), cudaMemcpyHostToDevice);

    findMaxKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(device_max, input, N);
    expSumKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(device_sum, input, N, device_max);
    fractionKernel<<<blocksPerGrid, threadsPerBlock>>>(output, input, N, device_max, device_sum);

    cudaDeviceSynchronize();
    cudaFree(device_max);
    cudaFree(device_sum);
}
