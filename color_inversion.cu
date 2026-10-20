#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

__global__ void inversionKernel (unsigned char* image, const int width, const int height) {

    const int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixel_x < width && pixel_y < height) {
        const int ele =  pixel_y * width + pixel_x;

        uchar4 *ptr = reinterpret_cast<uchar4 *> (image + ele * 4);
        uchar4 pixel = *ptr;
        pixel.x = 255 - pixel.x;
        pixel.y = 255 - pixel.y;
        pixel.z = 255 - pixel.z;
        *ptr = pixel;
    }
}