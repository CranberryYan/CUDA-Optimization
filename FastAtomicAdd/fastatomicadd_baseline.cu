// baseline 几乎没有并行性
// v0: Memory Throughput [%]	1.68
// v0: Memory Throughput [Gbyte/second]	3.65

#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

#define N 8 * 1024 * 1024
#define BLOCK_SIZE 512

// 向量乘法
__global__ void dot(float* a, float* b, float* c){
    const int step = gridDim.x * blockDim.x;
    float temp = 0.0;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = gid; i < N; i += step) {
        temp += a[i] * b[i];
    }
    atomicAdd(c, temp);
}

int main() {

    float *output_h;
    float *input1_d;
    float *input2_d;
    float *output_d;
    float *input1_host;
    float *input2_host;
    float output_host = 0;

    input1_host = (float*)malloc(sizeof(float) * N);
    input2_host = (float*)malloc(sizeof(float) * N);
    output_h = (float*)malloc(sizeof(float));
    cudaMalloc((void**)&input1_d, sizeof(float) * N);
    cudaMalloc((void**)&input2_d, sizeof(float) * N);
    cudaMalloc((void**)&output_d, sizeof(float));

    for (int i = 0; i < N; ++i) {
        input1_host[i] = static_cast<float>(rand()) / RAND_MAX;
        input2_host[i] = static_cast<float>(rand()) / RAND_MAX;
        output_host += input1_host[i] * input2_host[i];
    }

    cudaMemcpy(input1_d, input1_host, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(input2_d, input2_host, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemset(output_d, 0, sizeof(float));

    int blockSize = BLOCK_SIZE;
    int blockNum = (N + blockSize - 1) / blockSize;
    dim3 grid(blockNum);
    dim3 block(blockSize);
    dot<<<grid, block>>>(input1_d, input2_d, output_d);
    cudaMemcpy(output_h, output_d, sizeof(float), cudaMemcpyDeviceToHost);

    if (std::abs(output_h[0] - output_host) > output_host / 100) {
        std::cout << "output_h:     " << output_h[0] << std::endl;
        std::cout << "output_host:  " << output_host << std::endl;
        std::cout << "FAILED!" << std::endl;
    } else {
        std::cout << "output_h:     " << output_h[0] << std::endl;
        std::cout << "output_host:  " << output_host << std::endl;
        std::cout << "PASSED!" << std::endl;
    }

    free(input1_host);
    free(input2_host);
    free(output_h);
    cudaFree(input1_d);
    cudaFree(input2_d);


    return 0;
}