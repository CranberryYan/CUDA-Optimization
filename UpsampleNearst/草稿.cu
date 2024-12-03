// 内存对齐, 解决v3中的高利用率低吞吐率
// v1: Memory Throughput [%]	39.43
// v1: Memory Throughput [Gbyte/second]	289.31
// v1: L1/TEX Hit Rate [%]	16.67
// v2: Memory Throughput [%]	35.82
// v2: Memory Throughput [Gbyte/second]	262.15
// v3: Memory Throughput [%]	71.02
// v3: Memory Throughput [Gbyte/second]	281.60
// v3: L1/TEX Hit Rate [%]	77.25
// v3: L2 Hit Rate [%]	93.93

#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <algorithm>
#include <iostream>
#include <cuda_fp16.h>
using namespace std;
#define N 1 * 1024 * 1024
#define kBlockSize 256

// CUDA: grid stride looping
#define CUDA_1D_KERNEL_LOOP(i, n)                                                                 \
  for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x, step = blockDim.x * gridDim.x; i < (n); \
        i += step)

// Upsample Nearest2D Kernel is copyed from https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/user/kernels/upsample_nearest_kernel.cu#L78
template<typename T>
struct alignas(2 * sizeof(T)) Pack2X {
    T x;
    T y;
};

template<typename T>
__global__ void UpsampleNearest2D2XForward(const int32_t in_elem_cnt, const T* in_dptr,
                                           const int32_t in_height, const int32_t in_width,
                                           T* out_dptr) {
    const int32_t in_hw_size = in_width * in_height;
    CUDA_1D_KERNEL_LOOP(index, in_elem_cnt) {
        const T in_value = in_dptr[index];
        const int32_t nc_idx = index / in_hw_size;
        const int32_t hw_off = index - nc_idx * in_hw_size;
        const int32_t h = hw_off / in_width;
        const int32_t w = hw_off - h * in_width;
        Pack2X<T> out_value{in_value, in_value};
        Pack2X<T>* out_pack_dptr = reinterpret_cast<Pack2X<T>*>(out_dptr);
        out_pack_dptr[nc_idx * in_hw_size * 2 + h * 2 * in_width + w] = out_value;
        out_pack_dptr[nc_idx * in_hw_size * 2 + (h * 2 + 1) * in_width + w] = out_value;
    }
}

int main(){
    float *input_host = (float*)malloc(N*sizeof(float));
    float *input_device;
    cudaMalloc((void **)&input_device, N*sizeof(float));
    for (int i = 0; i < N; i++) input_host[i] = 1.0;
    cudaMemcpy(input_device, input_host, N*sizeof(float), cudaMemcpyHostToDevice);

    float *output_host = (float*)malloc(N * 4 * sizeof(float));
    float *output_device;
    cudaMalloc((void **)&output_device, N * 4 * sizeof(float));
    
    dim3 grid(N / kBlockSize, 1);
    dim3 block(kBlockSize, 1);
    UpsampleNearest2D2XForward<<<grid, block>>>(N, input_device, 1024, 1024, output_device);
    cudaMemcpy(output_host, output_device, N * 4 * sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 50; i++) {
        printf("%.5f\n", output_host[i]);
    }

    return 0;
}