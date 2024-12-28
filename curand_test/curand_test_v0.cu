#include <curand_kernel.h>
#include <iostream>

__global__ void generate_randoms(float* x, uint64_t counter_offset) {
    int i = threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(123, i, counter_offset, &state);  // 使用不同的 counter_offset
    x[i] = curand_uniform(&state);
}

int main() {
    float *d_x1, *d_x2;
    float h_x1[4], h_x2[4];
    size_t size = 4 * sizeof(float);

    // 分配设备内存
    cudaMalloc(&d_x1, size);
    cudaMalloc(&d_x2, size);

    // 启动内核，使用不同的 counter_offset
    generate_randoms<<<1, 4>>>(d_x1, 0);  // 使用 counter_offset 为 0
    generate_randoms<<<1, 4>>>(d_x2, 1);  // 使用 counter_offset 为 4

    // 同步设备
    cudaDeviceSynchronize();

    // 将结果从设备复制到主机
    cudaMemcpy(h_x1, d_x1, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_x2, d_x2, size, cudaMemcpyDeviceToHost);

    // 打印结果
    std::cout << "Random numbers from sequence 1 (counter_offset = 0):" << std::endl;
    for (int i = 0; i < 4; i++) {
        std::cout << h_x1[i] << std::endl;
    }

    std::cout << "Random numbers from sequence 2 (counter_offset = 4):" << std::endl;
    for (int i = 0; i < 4; i++) {
        std::cout << h_x2[i] << std::endl;
    }

    // 释放内存
    cudaFree(d_x1);
    cudaFree(d_x2);

    return 0;
}