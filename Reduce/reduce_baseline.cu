// v0运行时间: 933.44us
// v0带宽利用率: 62.67%
// v0内存吞吐量: 144.33GB/s

#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <vector>
#include <iostream>

#define N 32 * 1024 * 1024 // 32MB
#define BLOCK_SIZE 256

void CPU_reduce(std::vector<float> &input_, double &output_) {
    for (auto x : input_) {
        output_ += x;
    }
}

bool checkout(float output_, float output_host) {
    if (std::abs(output_ - output_host) > 0.0001) {
        return false;
    } else {
        return true;
    }
}

__global__ void reduce_v0(float *g_idata, float *g_odata) {
    
    // 256 * 32/8 = 1024Byte -> 1KB
    // 3080: 单个SM的L1 cache 128KB
    __shared__ float smem[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // SM, 每个block独享
    // <<<N / BLOCK_SIZE, BLOCK_SIZE(向上取整)>>>
    smem[tid] = g_idata[gid];
    __syncthreads(); // 使用smem, 同步

    // 归约
    // eg: [1, 2, 3, 4, 5, 6, 7, 8]
    // [1 + 2, 2, 3 + 4, 4, 5 + 6, 6, 7 + 8, 8]
    // [1 + 2 + 3 + 4, 2, 3 + 4 + 5 + 6, 4, 5 + 6 + 7 + 8, 6, 7 + 8, 8]
    // [1 + 2 + 3 + 4 + 5 + 6 + 7 + 8, 2, 3 + 4 + 5 + 6 + 7 + 8 4, 5 + 6 + 7 + 8, 6, 7 + 8, 8]
    for (unsigned int i = 1; i < blockDim.x; i *= 2) {
        if (tid % (2 * i) == 0) {
            smem[tid] += smem[tid + i];
        }
        __syncthreads(); // 使用smem, 同步
    }

    if (tid == 0) {
        // 写回每个block的sum
        g_odata[blockIdx.x] = smem[0];
    }
}

int main() {
    float *input_device;
    float *output_device;
    float *input_host = (float*)malloc(N * sizeof(float));
    float *output_host = (float*)malloc(N / BLOCK_SIZE * sizeof(float));
    cudaMalloc((void**)&input_device, N * sizeof(float));
    cudaMalloc((void**)&output_device, (N / BLOCK_SIZE) * sizeof(float));
    for (int i = 0; i < N; ++i) {
        input_host[i] = 1.0;
    }
    cudaMemcpy(input_device, input_host, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(N / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    int block_num = (N + BLOCK_SIZE - 1) / BLOCK_SIZE; // 向上取整
    reduce_v0<<<grid, block>>>(input_device, output_device);
    cudaMemcpy(output_host, output_device, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 1; i < N / BLOCK_SIZE; ++i) {
        output_host[0] += output_host[i];
    }

    bool res;
    std::vector<float> input_(N, 1.0);
    double output_ = 0;
    CPU_reduce(input_, output_);
    res = checkout(output_, output_host[0]);
    if (res) {
        std::cout << "PASSED!" << std::endl;
    } else {
        std::cout << "FAILED!" << std::endl;
    }


    return 0;
}