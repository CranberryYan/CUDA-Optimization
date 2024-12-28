// 内存对齐, 解决v3中的高利用率低吞吐率
// baseline(pytorch): Compute (SM) Throughput [%]	59.52
// baseline(pytorch): Memory Throughput [Gbyte/s]	334.98
// v1: Memory Throughput [%]	39.43
// v1: Memory Throughput [Gbyte/second]	289.31
// v1: L1/TEX Hit Rate [%]	16.67
// v2: Memory Throughput [%]	35.82
// v2: Memory Throughput [Gbyte/second]	262.15
// v3: Memory Throughput [%]	71.02
// v3: Memory Throughput [Gbyte/second]	281.60
// v3: L1/TEX Hit Rate [%]	77.25
// v3: L2 Hit Rate [%]	93.93
// v4: Memory Throughput [%]	84.89
// v4: Memory Throughput [Gbyte/second]	626.22
// v4: L1/TEX Hit Rate [%]	0
// v4: L2 Hit Rate [%]	80.04

#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <algorithm>
#include <iostream>
#include <cuda_fp16.h>

#define N 1 * 1024 * 1024
#define BLOCK_SIZE 256
#define pack_size 2
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

            int32_t inIdx = y_in * inputWidth + x_in;
            int32_t outIdx = y_out * outputWidth + x_out;

            outputImage[outIdx] = inputImage[inIdx];
        }
    }
}

template<typename T>
struct alignas(pack_size * sizeof(T)) Pack {
    T x;
    T y;
};
template<typename T>
__global__ void upsampleNearest(const T* input, T* output,
    const int32_t input_height, const int32_t input_width) {
    int32_t input_size = input_height * input_width;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = gid; i < N; i += blockDim.x * gridDim.x) {
        T in = input[i];
        int h = i / input_width;    // 第n行
        int w = i % input_width;    // 第m列
        Pack<T> out{in, in};
        Pack<T> *output_ = reinterpret_cast<Pack<T>*>(output);
        output_[(h * SCALE + 0) * input_width + w] = out; // 此时类似于向量化, 两个值, 宽增加一倍
        output_[(h * SCALE + 1) * input_width + w] = out;
    }
}

int main() {
    float *input_host, *output_host;
    float *input_d, *output_d, *output_h;
    input_host = (float*)malloc(sizeof(float) * N);
    output_host = (float*)malloc(sizeof(float) * N * std::pow(SCALE, 2));
    output_h = (float*)malloc(sizeof(float) * N * std::pow(SCALE, 2));
    cudaMalloc((void**)&input_d, sizeof(float) * N);
    cudaMalloc((void**)&output_d, sizeof(float) * N * std::pow(SCALE, 2));

    for (int32_t i = 0; i < N; ++i) {
        input_host[i] = static_cast<float>(rand()) / RAND_MAX;
        // input_host[i] = 1;
    }
    cudaMemcpy(input_d, input_host, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemset(output_d, 0, sizeof(float) * N * std::pow(SCALE, 2));

    int blockSize = BLOCK_SIZE;
    int blockNum = (N + blockSize - 1) / blockSize;
    dim3 grid(blockNum);
    dim3 block(blockSize);
    upsampleNearest<float><<<grid, block>>>(input_d, output_d, 1024, 1024);
    cudaMemcpy(output_h, output_d, sizeof(float) * N * std::pow(SCALE, 2), cudaMemcpyDeviceToHost);

    upsampleNearst_CPU(input_host, output_host, 1024, 1024);

    float sum = 0;
    for (int32_t i = 0; i < N * std::pow(SCALE, 2); ++i) {
        sum += output_h[i];
        if (std::abs(output_h[i] - output_host[i]) > 0.001) {
            printf("FAILED\n");
            std::cout << "output_host[" << i << "]: " << output_host[i] << std::endl;
            std::cout << "output_h[" << i << "]:    " << output_h[i] << std::endl;
        }
    }
    printf("sum: %f\n", sum);
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