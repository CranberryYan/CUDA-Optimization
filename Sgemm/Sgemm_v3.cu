// v0:
//  1.49ms
//	Compute (SM) Throughput [%]	92.10
//	Memory Throughput [Gbyte/second]	8.43
// v1:
//	使用shared_memory
//	性能分析意义不大, 为了强行适配该版本的改动
//	shape做出调整
//	Memory Throughput [Gbyte/second]	12.33
// v2:
//  使用shared_memory + 分块计算
//  Memory Throughput [Gbyte/second]	11.34
// v2_2:
//	重新分配BLOCK_SIZE, 适应shared_memory和block中最大thread限制
//	Memory Throughput [Gbyte/second]	12.31
// v2_3:
//  无性能优化, 根据自己的理解更改索引方式
// v3:
//  增加每个thread的工作量, 减少block的数量
//  297.82us
//  Memory Throughput [Gbyte/second]	14.77
#include <iostream>
#define STRIDE 2
#define BLOCK_SIZE 16

void random_matrix(int m, int n, float *mat) {
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			mat[i * n + j] = 2.0 * static_cast<float>(drand48()) - 1.0;
		}
	}
}

// A: [M, K]   B: [K, N]
void sgemm_CPU(float *A, float *B, float *C,
	const int M, const int N, const int K) {
	for (int m = 0; m < M; ++m) {
		for (int n = 0; n < N; ++n) {
			float temp = 0.0f;
			for (int k = 0; k < K; ++k) {
				temp += A[m*K + k] * B[k*N + n];
			}
			C[m*N + n] = temp;
			// printf("C[%d]: %f", m*N + n, temp);
		}
	}
}

bool checkout(float *C_buf_host_cpu, float *C_buf_host_gpu,
	const int M, const int N) {
	for (int m = 0;  m < M; ++m) {
		for (int n = 0; n < N; ++n) {
			if (std::abs(C_buf_host_cpu[m*N+n] -
				C_buf_host_gpu[m*N+n]) > 1e-3) {
				std::cout << "FAILED!" << std::endl;
				std::cout << "C_buf_host_cpu[" << m*N+n << "]: "
					<< C_buf_host_cpu[m*N+n] << std::endl
					<< "C_buf_host_gpu[" << m*N+n << "]: "
					<< C_buf_host_gpu[m*N+n] << std::endl;
				return false;
			}
		}
	}
	std::cout << "PASSED!" << std::endl;
	return true;
}

// before: 一个thread对应一个元素
// now: 一个thread会处理4个元素
template<unsigned int threadNum, unsigned int stride>
__global__ void sgemm_v3(float *A, float *B, float *C,
	const int M, const int N, const int K) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
	// A矩阵先偏移到所在的行
	float *A_ptr = A + (blockIdx.y * STRIDE * blockDim.y + threadIdx.y) * K;
	// B矩阵先偏移到所在的列
	float *B_ptr = B + blockIdx.x * STRIDE * blockDim.x + threadIdx.x;

  // 一个thread处理4个元素
  float temp[threadNum*stride][threadNum*stride];
  __shared__ float a_shared[threadNum*stride][threadNum*stride];
  __shared__ float b_shared[threadNum*stride][threadNum*stride];

  for (int k = 0; k < K; k += blockDim.x * stride) {
    for (int i = 0; i < STRIDE; ++i) {
      for (int j = 0; j < STRIDE; ++j) {
        a_shared[threadIdx.y + i * threadNum][threadIdx.x + j * threadNum] =
          A_ptr[i * threadNum * K + j * threadNum + k + threadIdx.x];
        b_shared[threadIdx.y + i * threadNum][threadIdx.x + j * threadNum] =
          B_ptr[(i * threadNum + threadIdx.y) * N + j * threadNum + k];
      }
    }

    // // UNROLL
    // a_shared[threadIdx.y + 0 * threadNum][threadIdx.x + 0 * threadNum] =
    //   A_ptr[0 * threadNum * K + 0 * threadNum + k + threadIdx.x];
    // a_shared[threadIdx.y + 0 * threadNum][threadIdx.x + 1 * threadNum] =
    //   A_ptr[0 * threadNum * K + 1 * threadNum + k + threadIdx.x];
    // b_shared[threadIdx.y + 0 * threadNum][threadIdx.x + 0 * threadNum] =
    //   B_ptr[(0 * threadNum + threadIdx.y) * N + 0 * threadNum + k];
    // b_shared[threadIdx.y + 0 * threadNum][threadIdx.x + 1 * threadNum] =
    //   B_ptr[(0 * threadNum + threadIdx.y) * N + 1 * threadNum + k];
    __syncthreads();

    for (int i = 0; i < STRIDE; ++i) {
      for (int j = 0; j < STRIDE; ++j) {
        for (int t = 0; t < threadNum; ++t) {
          temp[i][j] += a_shared[i][t] * b_shared[t][j];
        }
      }
    }
    __syncthreads();

    for (int i = 0; i < STRIDE; ++i) {
      for (int j = 0; j < STRIDE; ++j) {
        C[(y + i * threadNum) * N + (x + j + threadNum)] = temp[i][j];
      }
    }
  }
}

int main() {
	// lhs: [M, K]
	// rhs: [K, N]
	const unsigned int m = 1024;
	const unsigned int n = 1024;
	const unsigned int k = 1024;

	// host
	std::cout << " ============= host ============= " << std::endl;
	float *A_buf_host = (float*)malloc(m * k *sizeof(float));
	float *B_buf_host = (float*)malloc(k * n *sizeof(float));
	float *C_buf_host_cpu = (float*)malloc(m * n *sizeof(float));
	float *C_buf_host_gpu = (float*)malloc(m * n *sizeof(float));
	random_matrix(m, k, A_buf_host);
	random_matrix(k, n, B_buf_host);
	memset(C_buf_host_cpu, 0, m * n *sizeof(float));
	memset(C_buf_host_gpu, 0, m * n *sizeof(float));

	// device
	std::cout << " ============= device ============= " << std::endl;
	float *A_buf_device, *B_buf_device, *C_buf_device;
	cudaMalloc((void**)&A_buf_device, m * k *sizeof(float));
	cudaMalloc((void**)&B_buf_device, m * k *sizeof(float));
	cudaMalloc((void**)&C_buf_device, m * k *sizeof(float));
	cudaMemcpy(A_buf_device, A_buf_host, m * k *sizeof(float),
		cudaMemcpyHostToDevice);
	cudaMemcpy(B_buf_device, B_buf_host, k * n *sizeof(float),
		cudaMemcpyHostToDevice);

	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	cudaFuncCache cacheConfig;
	cudaDeviceProp deviceProp;
	cudaDeviceGetCacheConfig(&cacheConfig);
	cudaGetDeviceProperties(&deviceProp, 0);
	switch (cacheConfig) {
			case cudaFuncCachePreferNone:
					std::cout << "Current cache config: PreferNone" << std::endl;
					break;
			case cudaFuncCachePreferShared:
					std::cout << "Current cache config: PreferShared" << std::endl;
					break;
			case cudaFuncCachePreferL1:
					std::cout << "Current cache config: PreferL1" << std::endl;
					break;
			case cudaFuncCachePreferEqual:
					std::cout << "Current cache config: PreferEqual" << std::endl;
					break;
	}
	std::cout << "Max Shared Memory per Block: "
		<< deviceProp.sharedMemPerBlock << " bytes" << std::endl;
	std::cout << "Max Shared Memory per SM: "
		<< deviceProp.sharedMemPerMultiprocessor << " bytes" << std::endl;

	// CPU_segmm
	std::cout << " ============= CPU_segmm ============= " << std::endl;
	sgemm_CPU(A_buf_host, B_buf_host, C_buf_host_cpu,
		m, n, k);

	// GPU_segmm
  // 增加每个thread的工作量
  //  -> 减少block的数量
  // girdDim: [64, 64] -> [32, 32]
  // blockDim: [16, 16] -> 一个block有256个thread
	std::cout << " ============= GPU_segmm ============= " << std::endl;
	dim3 grid((m + BLOCK_SIZE - 1) / BLOCK_SIZE / STRIDE,
		(m + BLOCK_SIZE - 1) / BLOCK_SIZE / STRIDE);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	sgemm_v3<BLOCK_SIZE, STRIDE><<<grid, block>>>(
		A_buf_device, B_buf_device, C_buf_device,
		m, n, k);

	// verify
	std::cout << " ============= VERIFY ============= " << std::endl;
	cudaMemcpy(C_buf_host_gpu, C_buf_device, m * k *sizeof(float),
		cudaMemcpyDeviceToHost);
	bool res = checkout(C_buf_host_cpu, C_buf_host_gpu,
		m, n);

	// free
	free(A_buf_host);
	free(B_buf_host);
	free(C_buf_host_cpu);
	free(C_buf_host_gpu);
	cudaFree(A_buf_device);
	cudaFree(B_buf_device);
	cudaFree(C_buf_device);

	return 0;
}