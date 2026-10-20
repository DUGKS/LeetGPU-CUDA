constexpr int BLOCK_SIZE = 512;
constexpr int MAX_BLOCKS = BLOCK_SIZE;

__device__ void blockScan(float *data, int tidx, int len) {
    for (int i = 1; i < len; i <<= 1) {
        float v = i <= tidx ? data[tidx - i] : 0.0f;
        __syncthreads();
        data[tidx] += v;
        __syncthreads();
    }
}

__global__ void blockPrefixSum(float *output, float *blockSums, const float *input, int N, int K) {
    __shared__ float thread_sum[BLOCK_SIZE];
    const int tidx = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tidx;
    const int low_idx = idx * K;
    const int high_idx = min(low_idx + K, N);
    double acc = 0.0;
    for (int i = low_idx; i < high_idx; i++) {
        acc += input[i];
        output[i] = (float)acc;
    }
    thread_sum[tidx] = acc;
    __syncthreads();

    blockScan(thread_sum, tidx, blockDim.x);

    const float offset_sum = (tidx == 0) ? 0.0f : thread_sum[tidx - 1];
    for (int i = low_idx; i < high_idx; i++) {
        output[i] += offset_sum;
    }

    if (tidx == blockDim.x - 1) {
        blockSums[blockIdx.x] = thread_sum[BLOCK_SIZE - 1];
    }
}

__global__ void scanBlockSums(float *blockSums, int numBlocks) {
    __shared__ float shared_block_sum[BLOCK_SIZE];

    const int tidx = threadIdx.x;
    shared_block_sum[tidx] = tidx < numBlocks ? blockSums[tidx] : 0.0f;
    __syncthreads();

    blockScan(shared_block_sum, threadIdx.x, BLOCK_SIZE);
    if (tidx < numBlocks) {
        blockSums[tidx] = shared_block_sum[tidx];
    }
}

__global__ void addBlockSums(float *output, const float *blockSums, int N, int K) {
    const int tidx = threadIdx.x;
    const int idx = blockDim.x * blockIdx.x + tidx;
    const int low_idx = idx * K;
    const int high_idx = min(low_idx + K, N);
    const float offset_sum = blockIdx.x == 0 ? 0.0f : blockSums[blockIdx.x - 1];
    for (int i = low_idx; i < high_idx; i++) {
        output[i] += offset_sum;
    }
}