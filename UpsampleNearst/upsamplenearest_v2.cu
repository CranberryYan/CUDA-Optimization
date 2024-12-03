// shared memory
// baseline(pytorch): Compute (SM) Throughput [%]	59.52
// baseline(pytorch): Memory Throughput [Gbyte/s]	334.98
// v1: Memory Throughput [%]	39.43
// v1: Memory Throughput [Gbyte/second]	289.31
// v1: L1/TEX Hit Rate [%]	16.67
// v2: Memory Throughput [%]	35.82
// v2: Memory Throughput [Gbyte/second]	262.15
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

#define N 1 * 1024 * 1024
#define BLOCK_SIZE 32
#define SCALE 2

void upsampleNearst_CPU(const float* inputImage, float* outputImage, 
    int inputWidth, int inputHeight) {

    int outputWidth = inputWidth * SCALE;
    int outputHeight = inputHeight * SCALE;
    for (int y_out = 0; y_out < outputHeight; ++y_out) {
        for (int x_out = 0; x_out < outputWidth; ++x_out) {
            
            // 计算输入图像中对应的最近邻像素
            int x_in = static_cast<int>(x_out * inputWidth / outputWidth);
            int y_in = static_cast<int>(y_out * inputHeight / outputHeight);

            int inIdx = y_in * inputWidth + x_in;
            int outIdx = y_out * outputWidth + x_out;

            outputImage[outIdx] = inputImage[inIdx];
        }
    }
}

// __global__ void upsampleNearest(const float* input, float* output, int input_width, int input_height) {

//     int out_x = blockIdx.x * blockDim.x + threadIdx.x;
//     int out_y = blockIdx.y * blockDim.y + threadIdx.y;
//     int output_width = SCALE * input_width;
//     int output_height = SCALE * input_height;
//     if (out_x >= output_width || out_y >= output_height) {
//         return;
//     }
//     // 优化思路: 现在每次需要访问两次global memory
//     int in_x = min((int)((float)out_x / output_width * input_width), input_width - 1);
//     int in_y = min((int)((float)out_y / output_height * input_height), input_height - 1);
//     output[out_y * output_width + out_x] = input[in_y * input_width + in_x];
// }

__global__ void upsampleNearest(const float* input, float* output, int input_width, int input_height) {

    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int output_width = SCALE * input_width;
    int output_height = SCALE * input_height;
    if (out_x >= output_width || out_y >= output_height) {
        return;
    }

    __shared__ float smem_in[BLOCK_SIZE * BLOCK_SIZE]; // 注意: BlockSize: [32, 32]

    int in_x = min((int)((float)out_x / output_width * input_width), input_width - 1);
    int in_y = min((int)((float)out_y / output_height * input_height), input_height - 1);

    // 没有意义: 
    //  如果不使用shared memory: global memory -> global memory
    //  使用shared memory: global memory -> shared memory -> gloabl memory
    //  凭空多了一步, 如果在传输之间需要进行计算
    //  如果需要进行局部的运算, 使用shared memory可以避免在global memory计算, 有意义
    smem_in[tid_y * blockDim.x + tid_x] = input[in_y * input_width + in_x];
    __syncthreads();

    output[out_y * output_width + out_x] = smem_in[tid_y * blockDim.x + tid_x];
}

int main() {
    float *input_host, *output_host;
    float *input_d, *output_d, *output_h;
    input_host = (float*)malloc(sizeof(float) * N);
    output_host = (float*)malloc(sizeof(float) * N * std::pow(SCALE, 2));
    output_h = (float*)malloc(sizeof(float) * N * std::pow(SCALE, 2));
    cudaMalloc((void**)&input_d, sizeof(float) * N);
    cudaMalloc((void**)&output_d, sizeof(float) * N * std::pow(SCALE, 2));

    for (int i = 0; i < N; ++i) {
        input_host[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    cudaMemcpy(input_d, input_host, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemset(output_d, 0, sizeof(float) * N * std::pow(SCALE, 2));

    int blockSize = BLOCK_SIZE;
    int blockNum = (1024 * SCALE + blockSize - 1) / blockSize;
    dim3 grid(blockNum, blockNum);
    dim3 block(blockSize, blockSize);
    upsampleNearest<<<grid, block>>>(input_d, output_d, 1024, 1024);
    cudaMemcpy(output_h, output_d, sizeof(float) * N * std::pow(SCALE, 2), cudaMemcpyDeviceToHost);

    upsampleNearst_CPU(input_host, output_host, 1024, 1024);
    for (int i = 0; i < N * std::pow(SCALE, 2); ++i) {
        if (std::abs(output_h[i] - output_host[i]) > 0.001) {
            printf("FAILED\n");
            std::cout << "output_host[" << i << "]: " << output_host[i] << std::endl;
            std::cout << "output_h[" << i << "]:    " << output_h[i] << std::endl;
            break;
        }
    }
    printf("PASSED\n");
#if 1
    for (int i = 0; i < 20; ++i) {
        std::cout << "output_host[" << i << "]: " << output_host[i] << std::endl;
        std::cout << "output_h[" << i << "]:    " << output_h[i] << std::endl;
    }
#endif
    free(input_host);
    free(output_host);
    free(output_h);
    cudaFree(input_d);
    cudaFree(output_d);

    return 0;
}