// 向量化
// v0: 580.19us
// v0: Memory Throughput [%]	95.31
// v0: Memory Throughput [Gbyte/second]	702.51

#include <time.h>
#include <iostream>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 32 * 1024 * 1024
#define BLOCK_SIZE 512
#define DEVICE_FUNCTION __device__ __host__ __forceinline__

template<typename T>
void checkout(T *output_h, T *output_host) {
    for (int i = 0; i < N; ++i) {
        if (std::abs(output_h[i] - output_host[i]) > 0.001) {
            std::cout << "output_d[" << i << "]: " << output_h[i] << std::endl;
            std::cout << "output_host[" << i << "]: " << output_host[i] << std::endl;
            std::cout << "FAILED" << std::endl;
        }
    }
    for (int i = 0; i < 10; ++i) {
        std::cout << "output_d[" << i << "]: " << output_h[i] << std::endl;
        std::cout << "output_host[" << i << "]: " << output_host[i] << std::endl;
    }
    std::cout << "PASSED" << std::endl;
}

template<typename T>
struct MultiplyFunctor {
    DEVICE_FUNCTION T operator()(T x, T y) {
        return x * y;
    }
    DEVICE_FUNCTION T operator()(T x, T y, T z) {
        return x * y * z;
    }
};

template<typename T>
void dotmul_CPU(T *input1, T *input2, T *input3, T *output) {
    MultiplyFunctor<T> dotmul;
    for (int i = 0; i < N; ++i) {
        output[i] = dotmul(input1[i], input2[i], input3[i]);
    }
}

// 每个thread负责一个数据
template<typename T>
__global__ void dotmul(T *input1, T *input2, T *input3, T *output) {
    MultiplyFunctor<T> dotmul;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid * 4 > N) {
        return;
    }
    float4 in1 = reinterpret_cast<float4*>(input1)[gid];
    float4 in2 = reinterpret_cast<float4*>(input2)[gid];
    float4 in3 = reinterpret_cast<float4*>(input3)[gid];

    output[gid * 4 + 0] = dotmul(in1.x, in2.x, in3.x);
    output[gid * 4 + 1] = dotmul(in1.y, in2.y, in3.y);
    output[gid * 4 + 2] = dotmul(in1.z, in2.z, in3.z);
    output[gid * 4 + 3] = dotmul(in1.w, in2.w, in3.w);
}

int main() {
    float *input1_host;
    float *input2_host;
    float *input3_host;
    float *output_host;
    float *input1_d;
    float *input2_d;
    float *input3_d;
    float *output_d;

    input1_host = (float*)malloc(sizeof(float) * N);
    input2_host = (float*)malloc(sizeof(float) * N);
    input3_host = (float*)malloc(sizeof(float) * N);
    output_host = (float*)malloc(sizeof(float) * N);

    cudaMalloc((void**)&input1_d, sizeof(float) * N);
    cudaMalloc((void**)&input2_d, sizeof(float) * N);
    cudaMalloc((void**)&input3_d, sizeof(float) * N);
    cudaMalloc((void**)&output_d, sizeof(float) * N);

    for (int i = 0; i < N; ++i) {
        input1_host[i] = 1.0;
        input2_host[i] = 2.0;
        input3_host[i] = 3.0;
    }

    cudaMemcpy(input1_d, input1_host, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(input2_d, input2_host, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(input3_d, input3_host, sizeof(float) * N, cudaMemcpyHostToDevice);

    int block_size = BLOCK_SIZE;
    int block_num = (N + block_size - 1) / block_size;
    dim3 block(block_size);
    dim3 grid(block_num);
    
    std::cout << "=========== call kernel ===========" << std::endl;
    dotmul<float><<<grid, block>>>(input1_d, input2_d, input3_d, output_d);
    cudaDeviceSynchronize();
    std::cout << "=========== call kernel done ===========" << std::endl;

    std::cout << "=========== verify ===========" << std::endl;
    cudaMemcpy(output_host, output_d, sizeof(float) * N, cudaMemcpyDeviceToHost);
    dotmul_CPU<float>(input1_host, input2_host, input3_host, output_host);
    checkout<float>(output_host, output_host);

    free(input1_host);
    free(input2_host);
    free(output_host);
    cudaFree(input1_d);
    cudaFree(input2_d);
    cudaFree(output_d);

    return 0;
}