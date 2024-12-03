// v1: 内存对齐 + unrolling
// v0: Memory Throughput [%]	1.68
// v0: Memory Throughput [Gbyte/second]	3.65
// v1: Memory Throughput [%]	1.69
// v1: Memory Throughput [Gbyte/second]	3.65

#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

#define N 32 * 1024 * 1024
#define BLOCK_SIZE 512

// alignas: 确保每个 Pack 对象的内存地址按照 sizeof(T) * pack_size 的倍数对齐
template<typename T, int pack_size>
struct alignas(sizeof(T) * pack_size) Pack {
    T elem[pack_size];
};

template<typename T, int pack_size>
__device__ __inline__ void AtomicAdd(Pack<T, pack_size> *addr, T val) {

#pragma unroll // 编译命令
    for (int i = 0; i < pack_size; ++i) {
        atomicAdd(reinterpret_cast<T*>(addr) + i, static_cast<T>(val));
    }
}

// 向量乘法
template<typename T, int pack_size>
__global__ void dot(Pack<T, pack_size> *a, Pack<T, pack_size> *b, Pack<T, pack_size> *c){
    int step = blockDim.x * gridDim.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    T tmp = 0.0;
    for (int i = gid; i * pack_size < N; i += step) {
#pragma unroll
        for (int j = 0; j < pack_size; ++j) {
            tmp += a[i].elem[j] * b[i].elem[j];
        }
    }

    AtomicAdd<T, pack_size>(c, tmp);
}

int main(){

    const int pack_size = 4;
    float *x_host, *y_host;
    float *x_d, *y_d;
    double output_host = 0;
    float *output_h, *output_d;
    x_host = (float*)malloc(sizeof(float) * N);
    y_host = (float*)malloc(sizeof(float) * N);
    output_h = (float*)malloc(sizeof(float) * pack_size);
    cudaMalloc((void**)&x_d, sizeof(float) * N);
    cudaMalloc((void**)&y_d, sizeof(float) * N);
    cudaMalloc((void**)&output_d, sizeof(float) * pack_size);

    for (int i = 0; i < N; ++i) {
        x_host[i] = static_cast<float>(rand()) / RAND_MAX;
        y_host[i] = static_cast<float>(rand()) / RAND_MAX;
        output_host += x_host[i] * y_host[i];
    }

    cudaMemcpy(x_d, x_host, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y_host, sizeof(float) * N, cudaMemcpyHostToDevice);
    Pack<float, pack_size> *x_pack = reinterpret_cast<Pack<float, pack_size>*>(x_d);
    Pack<float, pack_size> *y_pack = reinterpret_cast<Pack<float, pack_size>*>(y_d);
    Pack<float, pack_size> *out_pack = reinterpret_cast<Pack<float, pack_size>*>(output_d);

    int blockNum = (N + BLOCK_SIZE - 1) / BLOCK_SIZE / pack_size;
    dim3 grid(blockNum);
    dim3 block(BLOCK_SIZE);
    dot<float, pack_size><<<grid, block>>>(x_pack, y_pack, out_pack);
    cudaMemcpy(output_h, output_d, sizeof(float) * pack_size, cudaMemcpyDeviceToHost);

    std::cout << "output_h:      " << output_h[0] << std::endl;
    std::cout << "output_host:   " << output_host << std::endl;
    if (std::abs(output_h[0] - output_host) > output_host / 100) {
        std::cout << "FAILED" << std::endl;
    } else {
        std::cout << "PASSED" << std::endl;
    }
}