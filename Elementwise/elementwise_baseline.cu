#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>


void checkout (float *output_h, float *output_host, int n) {
    for (int i = 0; i < n; ++i) {
        if (std::abs(output_h[i] - output_host[i] > 0.0001)) {
            std::cout << "output_h[" << i << "]:    " << output_h[i] << std::endl;
            std::cout << "output_host[" << i << "]: " << output_host[i] << std::endl;
            std::cout << "FAILED" << std::endl;
        }
    }
    std::cout << "PASSED" << std::endl;
}

void relu_CPU(float *input, float *output, int n) {
    for (int i = 0; i < n; ++i) {
        output[i] = input[i] > 0 ? input[i] : 0;
    }
}

__global__ void relu(float *input, float *output) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    output[gid] = input[gid] > 0 ? input[gid] : 0;
}

int main() {
    float *input_h;
    float *output_h;
    float *input_d;
    float *output_d;
    float *input_host;
    float *output_host;
    int32_t elem_cnt = 32 * 1024 * 1024; // 32M
    input_h = (float*)malloc(elem_cnt * sizeof(float));
    output_h = (float*)malloc(elem_cnt * sizeof(float));
    input_host = (float*)malloc(elem_cnt * sizeof(float));
    output_host = (float*)malloc(elem_cnt * sizeof(float));
    cudaMalloc((void**)&input_d, sizeof(float) * elem_cnt);
    cudaMalloc((void**)&output_d, sizeof(float) * elem_cnt);
    for (int i = 0; i < elem_cnt; ++i) {
        input_host[i] = i - 10000;
    }
    cudaMemcpy(input_d, input_host, sizeof(float) * elem_cnt, cudaMemcpyHostToDevice);

    int blockSize = 1024;
    int blockNum = (elem_cnt + blockSize - 1) / blockSize;
    dim3 grid(blockNum);
    dim3 block(blockSize);
    std::cout << "=========== call kernel ===========" << std::endl;
    relu<<<grid, block>>>(input_d, output_d);
    std::cout << "=========== call kernel done ===========" << std::endl;
    cudaMemcpy(output_h, output_d, sizeof(float) * elem_cnt, cudaMemcpyDeviceToHost);

    std::cout << "=========== verfiy ===========" << std::endl;
    relu_CPU(input_host, output_host, elem_cnt);
    checkout(output_h, output_host, elem_cnt);
    

    return 0;
}
