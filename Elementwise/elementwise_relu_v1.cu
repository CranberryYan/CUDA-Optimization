// 向量化读取
// v0运行时间: 431.14us
// v0带宽利用率: 83.56%
// v0内存吞吐量: 616.56GB/s
// v1运行时间: 393.63us
// v1带宽利用率: 92.28%
// v1内存吞吐量: 675.51GB/s
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 32 * 1024 * 1024

void checkout (float *output_h, float *output_host, int n) {
    for (int i = 0; i < n; ++i) {
        if (std::abs(output_h[i] - output_host[i] > 0.0001)) {
            std::cout << "output_h[" << i << "]:    " << output_h[i] << std::endl;
            std::cout << "output_host[" << i << "]: " << output_host[i] << std::endl;
            std::cout << "FAILED" << std::endl;
        }
    }
    for (int i = 10000; i < 10010; ++i) {
        std::cout << "output_h[" << i << "]:    " << output_h[i] << std::endl;
        std::cout << "output_host[" << i << "]: " << output_host[i] << std::endl;
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
    if (gid * 4 >= N) {
        return;
    }
    float4 in = reinterpret_cast<float4*>(input)[gid];
    output[gid * 4 + 0] = in.x > 0 ? in.x : 0;
    output[gid * 4 + 1] = in.y > 0 ? in.y : 0;
    output[gid * 4 + 2] = in.z > 0 ? in.z : 0;
    output[gid * 4 + 3] = in.w > 0 ? in.w : 0;
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
    int blockNum = (elem_cnt + blockSize - 1) / blockSize / 4;
    dim3 grid(blockNum);
    dim3 block(blockSize);
    std::cout << "=========== call kernel ===========" << std::endl;
    relu<<<grid, block>>>(input_d, output_d);
    std::cout << "=========== call kernel done ===========" << std::endl;
    cudaMemcpy(output_h, output_d, sizeof(float) * elem_cnt, cudaMemcpyDeviceToHost);

    std::cout << "=========== verfiy ===========" << std::endl;
    relu_CPU(input_host, output_host, elem_cnt);
    checkout(output_h, output_host, elem_cnt);

    free(input_h);
    free(output_h);
    free(input_host);
    free(output_host);
    cudaFree(input_d);
    cudaFree(output_d);


    return 0;
}
