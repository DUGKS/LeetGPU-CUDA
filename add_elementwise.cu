#include <stdio.h>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <chrono>
#include <iostream>

__host__ __device__ inline float4 operator+(const float4 &a, const float4 &b) {
    return make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}

__global__ void element_add_f32_kernel(float *a, float *b, float*c, int num_floats) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_floats) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void element_add_f32x4_kernel(float4 *a, float4 *b, float4 *c, int num_float4s) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_float4s) {
        c[idx].x = a[idx].x + b[idx].x;
        c[idx].y = a[idx].y + b[idx].y;
        c[idx].z = a[idx].z + b[idx].z;
        c[idx].w = a[idx].w + b[idx].w;
    }
}

__global__ void element_add_f32x4_kernel(float *a, float *b, float *c, int num_floats) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx4 = idx * 4;
    if (idx4 + 3 < num_floats) {
        float4 a4 = reinterpret_cast<float4*>(a + idx4)[0];
        float4 b4 = reinterpret_cast<float4*>(b + idx4)[0];
        float4 c4 = a4 + b4;
        reinterpret_cast<float4*>(c + idx4)[0] = c4;

    } else {
        if (idx == 0) {
            for (int i = 0; i + idx4 < num_floats; i++) {
                c[idx4 + i] = a[idx4 + i] + b[idx4 + i];
            }
        }
    }
}

__global__ void element_add_f16_kernel(half *a, half *b, half *c, int num_halfs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_halfs) {
        c[idx] = __hadd(a[idx], b[idx]);
    }
    if (blockIdx.x == 0 and threadIdx.x == 0) {
        printf("kernel name: element_add_f16_kernel, threadIdx.x: %d, blockIdx.x: %d, blockDim.x: %d, idx: %d\n", threadIdx.x, blockIdx.x, blockDim.x, idx);
    }
}

__global__ void element_add_f16x2_kernel(half2 *a, half2 *b, half2 *c, int num_half2s) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_half2s) {
        c[idx] = __hadd2(a[idx], b[idx]);
    }
}

__global__ void element_float_to_half2_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                              half2* __restrict__ half2_a, half2* __restrict__ half2_b,
                                              int num_floats) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idx2 = idx * 2;
    if (idx2 + 1 < num_floats) {
        half a_lo = __float2half(a[idx2]);
        half a_hi = __float2half(a[idx2+1]);
        half2_a[idx] = __halves2half2(a_lo, a_hi);

        half b_lo = __float2half(b[idx2]);
        half b_hi = __float2half(b[idx2+1]);
        half2_b[idx] = __halves2half2(b_lo, b_hi);
    }
}

__global__ void element_half2_to_float_kernel(const half2* __restrict__ half2_c, float* __restrict__ c, int num_floats) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idx2 = idx * 2;
    if (idx2 + 1 < num_floats) {
        float2 pair = __half22float2(half2_c[idx]);
        c[idx2] = pair.x;
        c[idx2+1] = pair.y;
    }
}

class KernelFloat {
public:
    float *host_a{nullptr};
    float *host_b{nullptr};
    float *host_c{nullptr};
    float *device_float_a;
    float *device_float_b;
    float *device_float_c;
    int num_floats{0};
    int num_threads{256};
public:
    KernelFloat(float *a, float *b, float *c, int num_floats) : host_a(a), host_b(b), host_c(c), num_floats(num_floats) {
        deviceResources();
    }

public:
    virtual void run() = 0;
    virtual ~KernelFloat() {
        cudaFree(device_float_a);
        cudaFree(device_float_b);
        cudaFree(device_float_c);
    }
private:
    void deviceResources() {
        cudaMalloc(&device_float_a, num_floats * sizeof(float));
        cudaMalloc(&device_float_b, num_floats * sizeof(float));
        cudaMalloc(&device_float_c, num_floats * sizeof(float));
    }
};

class KernelFloat4: public KernelFloat {
public:
    KernelFloat4(float *a, float *b, float *c, int num_floats): KernelFloat(a, b, c, num_floats) {
        cudaMemcpy(device_float_a, host_a, num_floats * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(device_float_b, host_b, num_floats * sizeof(float), cudaMemcpyHostToDevice);
    }

    void run() override {
        computeGPU();
        cudaMemcpy(host_c, device_float_c, num_floats * sizeof(float), cudaMemcpyDeviceToHost);
        for (int n = num_floats - 1; n > num_floats - 20; n--) {
            printf("function_name: %s, %f + %f = %f\n",__func__, host_a[n], host_b[n], host_c[n]);
        }
        printf("Current context: %s\n", __PRETTY_FUNCTION__);
    }

    ~KernelFloat4() override {
    }
private:
    void computeGPU() {
        int num_float4s = (num_floats + 3) / 4;
        int num_blocks = (num_float4s + num_threads - 1) / num_threads;
        element_add_f32x4_kernel<<<num_blocks, num_threads>>>(device_float_a, device_float_b, device_float_c, num_floats);
    }
};

class KernelHalf: public KernelFloat {
public:
    half* device_half_a;
    half* device_half_b;
    half* device_half_c;
public:
    KernelHalf(float *a, float *b, float *c, int num_floats): KernelFloat(a, b, c, num_floats) {
        deviceResources();
        std::vector<half> host_half_a(num_floats);
        std::vector<half> host_half_b(num_floats);
        for (int i = 0;i < num_floats;++i) {
            host_half_a[i] = __float2half(host_a[i]);
            host_half_b[i] = __float2half(host_b[i]);
        }
        cudaMemcpy(device_half_a, host_half_a.data(), num_floats*sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(device_half_b, host_half_b.data(), num_floats*sizeof(half), cudaMemcpyHostToDevice);
    }

    void run() override {
        computeGPU();
        std::vector<half> host_half_c(num_floats);
        cudaMemcpy(host_half_c.data(), device_half_c, num_floats * sizeof(half), cudaMemcpyDeviceToHost);
        for (int i = 0;i < num_floats;++i) {
            host_c[i] = __half2float(host_half_c[i]);
        }
        for (int n = num_floats - 1; n > num_floats - 20; n--) {
            printf("function_name: %s, %f + %f = %f\n",__func__, host_a[n], host_b[n], host_c[n]);
        }
        printf("Current context: %s\n", __PRETTY_FUNCTION__);
    }

    virtual ~KernelHalf() {
        cudaFree(device_half_a);
        cudaFree(device_half_b);
        cudaFree(device_half_c);
    }
private:
    void computeGPU() {
        int num_blocks = (num_floats + num_threads - 1) / num_threads;
        element_add_f16_kernel<<<num_blocks, num_threads>>>(device_half_a, device_half_b, device_half_c, num_floats);
    }
    void deviceResources() {
        cudaMalloc(&device_half_a, num_floats * sizeof(half));
        cudaMalloc(&device_half_b, num_floats * sizeof(half));
        cudaMalloc(&device_half_c, num_floats * sizeof(half));
    }
};

class KernelHalf2 : public KernelFloat {
public:
    half2 *device_half2_a;
    half2 *device_half2_b;
    half2 *device_half2_c;
    int num_elements;
public:
    KernelHalf2(float *a, float *b, float *c, int num_floats): KernelFloat(a,b,c,num_floats) {
        num_elements = num_floats/2;
        deviceResources();
    }

    void runCPU() {
        auto start = std::chrono::high_resolution_clock::now();
        int num_blocks = (num_elements + num_threads - 1) / num_threads;
        convertCPU();
        element_add_f16x2_kernel<<<num_blocks, num_threads>>>(device_half2_a, device_half2_b, device_half2_c, num_elements);
        std::vector<half2> host_half2_c(num_elements);
        cudaMemcpy(host_half2_c.data(), device_half2_c, num_elements * sizeof(half2), cudaMemcpyDeviceToHost);
        for (size_t i = 0;i < num_elements;++i) {
            float2 float_pair = __half22float2(host_half2_c[i]);
            host_c[i*2] = float_pair.x;
            host_c[i*2+1] = float_pair.y;
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "function name: " << __func__ <<"    "
                  << "elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count()
                  << std::endl;

        // for (int n = numFloats - 1; n > numFloats - 20; n--) {
        //     printf("function_name: %s, %f + %f = %f\n",__func__, host_a[n], host_b[n], host_c[n]);
        // }
        printf("Current context: %s\n", __PRETTY_FUNCTION__);
    }

    void runGPU() {
        auto start = std::chrono::high_resolution_clock::now();
        int num_blocks = (num_elements + num_threads - 1) / num_threads;
        convertGPU();
        element_add_f16x2_kernel<<<num_blocks, num_threads>>>(device_half2_a, device_half2_b, device_half2_c, num_elements);
        element_half2_to_float_kernel<<<num_blocks, num_threads>>>(device_half2_c, device_float_c, num_floats);
        cudaMemcpy(host_c, device_float_c, num_floats * sizeof(float), cudaMemcpyDeviceToHost);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "function name: " << __func__ <<"    "
                  << "elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count()
                  << std::endl;

        // for (int n = numFloats - 1; n > numFloats - 20; n--) {
        //     printf("function_name: %s, %f + %f = %f\n",__func__, host_a[n], host_b[n], host_c[n]);
        // }
        printf("Current context: %s\n", __PRETTY_FUNCTION__);
    }

    void run() override {
        runCPU();
        runGPU();
        runCPU();
    }

    ~KernelHalf2 () override {
        cudaFree(device_half2_a);
        cudaFree(device_half2_b);
        cudaFree(device_half2_c);
    }
private:
    void deviceResources() {
        cudaMalloc(&device_half2_a, num_elements * sizeof(half2));
        cudaMalloc(&device_half2_b, num_elements * sizeof(half2));
        cudaMalloc(&device_half2_c, num_elements * sizeof(half2));
    }

    void convertCPU() {
        std::vector<half2> host_half2_a(num_elements);
        std::vector<half2> host_half2_b(num_elements);
        for(size_t i = 0;i < num_elements; ++i) {
            half lo = __float2half(host_a[2*i]);
            half hi = __float2half(host_a[2*i+1]);
            host_half2_a[i] = __halves2half2(lo, hi);
        }
        for(size_t i = 0;i < num_elements; ++i) {
            half lo = __float2half(host_b[2*i]);
            half hi = __float2half(host_b[2*i+1]);
            host_half2_b[i] = __halves2half2(lo, hi);
        }

        cudaMemcpy(device_half2_a, host_half2_a.data(), num_elements * sizeof(half2), cudaMemcpyHostToDevice);
        cudaMemcpy(device_half2_b, host_half2_b.data(), num_elements * sizeof(half2), cudaMemcpyHostToDevice);
    }

    void convertGPU() {
        cudaMemcpy(device_float_a, host_a, num_floats * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(device_float_b, host_b, num_floats * sizeof(float), cudaMemcpyHostToDevice);
        int num_blocks = (num_elements + num_threads - 1) / num_threads;
        element_float_to_half2_kernel<<<num_blocks, num_threads>>>(device_float_a, device_float_b, device_half2_a, device_half2_b, num_floats);
    }

};


int main() {
    cudaFree(0);
    int NumFloats = 4098;
    int PaddedNumFloats = (NumFloats + 3) / 4 * 4;
    size_t size = PaddedNumFloats * sizeof(float);
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);
    memset(h_a, 0, size);
    memset(h_b, 0, size);
    memset(h_c, 0, size);
    for (int i = 0; i < NumFloats;i++) {
        h_a[i] = i * 0.001;
        h_b[i] = ((i << 1) + 1.0) * 0.001;
    }

    // elementwise addition
    // KernelTestF16(h_a, h_b, h_c, PaddedNumFloats).run();
    // KernelFloat4(h_a, h_b, h_c, PaddedNumFloats).run();
    // KernelHalf(h_a, h_b, h_c, PaddedNumFloats).run();
    KernelHalf2(h_a, h_b, h_c, PaddedNumFloats).run();

    free(h_a);
    free(h_b);
    free(h_c);
}

