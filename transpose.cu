#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
__global__ void matrixTransposeKernel(float *output, const float *input, int num_row, int num_col) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < num_row && col < num_col) {
        output[col * num_row + row] = input[row * num_col + col];
    }
}

template <int TILE_SIZE>
__global__ void matrixTransposeTiledKernel(float *output, const float *input, int num_row, int num_col) {
    __shared__ float input_block[TILE_SIZE][TILE_SIZE + 1];

    int row_in = TILE_SIZE * blockIdx.y + threadIdx.y;
    int col_in = TILE_SIZE * blockIdx.x + threadIdx.x;

    input_block[threadIdx.y][threadIdx.x] = (row_in < num_row && col_in < num_col) ? input[row_in * num_col + col_in] : 0.0;

    __syncthreads();

    int row_out = TILE_SIZE * blockIdx.x + threadIdx.y;
    int col_out = TILE_SIZE * blockIdx.y + threadIdx.x;

    if (col_out < num_row && row_out < num_col) {
        output[row_out * num_row + col_out] = input_block[threadIdx.x][threadIdx.y];
    }
}

class matrixTranspose {
public:
    int M;
    int N;
    float *device_matrix_input;
    float *device_matrix_output;
    float *host_matrix_output;
    dim3 block_size{16, 16};
public:
    matrixTranspose(float *host_matrix_input, int M, int N): M(M), N(N) {
        cudaMalloc(&device_matrix_input, M * N * sizeof(float));
        cudaMalloc(&device_matrix_output, M * N * sizeof(float));
        host_matrix_output = (float*) malloc(M * N * sizeof(float));
        cudaMemcpy(device_matrix_input, host_matrix_input, M * N * sizeof(float), cudaMemcpyHostToDevice);
    }

    void run() {
       dim3 grid_size{(M + block_size.y - 1 / block_size.y), (N + block_size.x - 1) / block_size.x};
       matrixTransposeTiledKernel<16><<<grid_size, block_size>>>(device_matrix_output, device_matrix_input, M, N);
       cudaMemcpy(host_matrix_output, device_matrix_output, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    }
    ~matrixTranspose() {
        cudaFree(device_matrix_input);
        cudaFree(device_matrix_output);
        cudaFree(host_matrix_output);
    }
};

void transpose_cpu(float *output, const float *input, int M, int N) {
    for(int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            output[j * M + i] = input[i * N + j];
        }
    }
}

void compareMatrix(float *host, float *device, int M, int N) {
    int num_errors = 0;
    for(int i = 0;i < N;i++) {
        for (int j = 0;j < M;j++) {
            float host_value = host[i * M + j];
            float device_value = device[i * M + j];
            float diff = host_value - device_value;
            if (fabs(diff) > 1E-6) {
                num_errors++;
                printf("row: %d, col: %d, host_value: %f, device_value: %f, diff: %f\n", i, j, host_value, device_value, diff);
            }
        }
    }
    if (num_errors == 0) {
        printf("Checking done. No different value found.\n");
    }
}

int main() {
    const int M = 128;
    const int N = 128;
    float *input_matrix = (float*) malloc(M * N * sizeof(float));
    float *output_matrix = (float*) malloc(M * N * sizeof(float));

    for (int i = 0;i < M; i++) {
        for (int j = 0; j < N; j++) {
            input_matrix[i * N + j] = (2 * i + j) * 0.001;
        }
    }

    transpose_cpu(output_matrix, input_matrix, M, N);

    matrixTranspose kernel(input_matrix, M, N);
    kernel.run();

    compareMatrix(output_matrix, kernel.host_matrix_output, M, N);

    free(input_matrix);
    free(output_matrix);
}