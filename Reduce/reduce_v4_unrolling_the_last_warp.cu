// 展开循环
// v0运行时间: 933.44us
// v0带宽利用率: 62.67%
// v0内存吞吐量: 144.33GB/s
// v1运行时间: 675.90us
// v1带宽利用率: 86.46%
// v1内存吞吐量: 199.35GB/s
// v2运行时间: 649.50us
// v2带宽利用率: 89.95%
// v2内存吞吐量: 207.42GB/s
// v3运行时间: 337.25us
// v3带宽利用率: 89.86%
// v3内存吞吐量: 398.79GB/s
// v4运行时间: 199.33us
// v4带宽利用率: 92.26%
// v4内存吞吐量: 674.85GB/s(比较接近760GB/s的理论值)
// v4: L1/TEX Cache Throughput [%]	68.56
// v4: L1/TEX Hit Rate [%]	0.27

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

__device__ void warpReduce(volatile float *smem, int tid) {
    smem[tid] += smem[tid + 32];
    smem[tid] += smem[tid + 16];
    smem[tid] += smem[tid + 8];
    smem[tid] += smem[tid + 4];
    smem[tid] += smem[tid + 2];
    smem[tid] += smem[tid + 1];
}

__global__ void reduce_v4(float *g_idata, float *g_odata) {
    
    // 256 * 32/8 = 1024Byte -> 1KB
    // 3080: 单个SM的L1 cache 128KB
    __shared__ float smem[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // SM, 每个block独享
    // <<<N / BLOCK_SIZE, BLOCK_SIZE(向上取整)>>>
    // 每次取数据之后再额外做一次加法
    // 只用原来一半的block, 因此在读数据的时候就做一次规约
    // 将后一半的数据全部加到前一半
    // 少分配了一半的block, 提高了利用率
    smem[tid] = g_idata[gid] + g_idata[gid + blockDim.x];
    __syncthreads(); // 使用smem, 同步

    // for (unsigned int i = blockDim.x / 2; i > 0; i >>= 1)
    //  当 i = 32以后, 每次的add都是发生在第0号warp中, 因此可以展开
    for (unsigned int i = blockDim.x / 2; i > 32; i >>= 1) {
        if (tid < i) {
            smem[tid] += smem[tid + i];
        }
        __syncthreads();
    }

    if (tid < 32) {
        warpReduce(smem, tid);
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

    int block_num = (N + BLOCK_SIZE - 1) / BLOCK_SIZE / 2; // 向上取整
    dim3 grid(block_num);
    dim3 block(BLOCK_SIZE);
    reduce_v4<<<grid, block>>>(input_device, output_device);
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
        std::cout << "CPU: " << output_ << std::endl;
        std::cout << "GPU: " << output_host[0] << std::endl;
    } else {
        std::cout << "FAILED!" << std::endl;
        std::cout << "CPU: " << output_ << std::endl;
        std::cout << "GPU: " << output_host[0] << std::endl;
    }

    free(input_host);
    free(output_host);
    cudaFree(input_device);
    cudaFree(output_device);

    return 0;
}