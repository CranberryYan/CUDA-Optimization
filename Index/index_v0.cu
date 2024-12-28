#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <vector>
#include <random>
#include <iostream>

#define ELE_NUM 32 * 1024 * 1024
#define INDEX_SIZE 15


void index_cpu(int *tensor_host, int *tensor_res, int *index) {
    for (int i = 0; i < INDEX_SIZE; ++i) {
        tensor_res[i] = tensor_host[index[i]];
    }
}

__global__ void index_gpu(int *tensor_d, int *tensor_d_res, int *index, int n) {
    int tid = threadIdx.x;
    int gid = blockIdx.x  * blockDim.x + threadIdx.x;
    if (gid >= n) {
        return;
    }

    tensor_d_res[gid] = tensor_d[index[gid]];
}

int main() {
    int *tensor_d, *tensor_d_res, *index_d;
    int *tensor_res = (int*)malloc(sizeof(int) * INDEX_SIZE);
    int *tensor_h = (int*)malloc(sizeof(int) * INDEX_SIZE);
    int *tensor_host = (int*)malloc(sizeof(int) * ELE_NUM);
    int *index = (int*)malloc(sizeof(int) * INDEX_SIZE);
    for (int i = 0; i < ELE_NUM; ++i) {
        tensor_host[i] = i % 9;
    }

    std::random_device rd;  // 设置随机数引擎和分布
    std::mt19937 gen(rd()); // 随机数生成器
    std::uniform_int_distribution<int> dis(0, ELE_NUM - 1); // 区间 [0, MAX_VAL)

    // 使用生成器为index赋值
    for (int i = 0; i < INDEX_SIZE; ++i) {
        index[i] = dis(gen);
    }

    cudaMalloc((void**)&index_d, sizeof(int) * INDEX_SIZE);
    cudaMalloc((void**)&tensor_d, sizeof(int) * ELE_NUM);
    cudaMalloc((void**)&tensor_d_res, sizeof(int) * INDEX_SIZE);
    cudaMemcpy(index_d, index, sizeof(int) * INDEX_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(tensor_d, tensor_host, sizeof(int) * ELE_NUM, cudaMemcpyHostToDevice);
    int blockSize = INDEX_SIZE > 1024 ? 1024 : INDEX_SIZE;
    int blockNum = (INDEX_SIZE + blockSize - 1) / blockSize;
    dim3 grid(blockNum);
    dim3 block(blockSize);
    index_gpu<<<grid, block>>>(tensor_d, tensor_d_res, index_d, INDEX_SIZE);
    cudaMemcpy(tensor_h, tensor_d_res, sizeof(int) * INDEX_SIZE, cudaMemcpyDeviceToHost);

    index_cpu(tensor_host, tensor_res, index);
    for (int i = 0; i < INDEX_SIZE; ++i) {
        if (std::abs(tensor_res[i] - tensor_h[i]) > 0.001) {
            std::cout << "tensor_res[" << i << "]:  " << tensor_res[i] << std::endl;
            std::cout << "tensor_h[" << i << "]:    " << tensor_h[i] << std::endl;
            printf("FAILED\n");
            break;
        }
    }
    printf("PASSED!");


    free(index);
    free(tensor_h);
    free(tensor_res);
    free(tensor_host);
    cudaFree(index_d);
    cudaFree(tensor_d);
    cudaFree(tensor_d_res);

    return 0;
}