// v0:
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
#include <iostream>
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
				std::cout << "C_buf_host_cpu[" << m*N+n << "]: " << C_buf_host_cpu[m*N+n] << std::endl
					<< "C_buf_host_gpu[" << m*N+n << "]: " << C_buf_host_gpu[m*N+n] << std::endl;
				return false;
			}
		}
	}

	std::cout << "PASSED!" << std::endl;
	return true;
}

// BLOCK_SIZE: 16
// 每个block有16个thread, 每个thread负责一个元素
// 	每个block负责16*16个元素, 一共64*64个block
//	-> 1024*1024个元素
template<unsigned int BLOCK_DIM>
__global__ void sgemm_v3(float *A, float *B, float *C,
	const int M, const int N, const int K) {
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	// block_level
	// blockDim: [16, 16]
	// blockIdx: [0, 63]
	// A矩阵先偏移到所在的行
	float *A_ptr = A + (blockIdx.y * blockDim.y + threadIdx.y) * K;
	// B矩阵先偏移到所在的列
	float *B_ptr = B + blockIdx.x * blockDim.x + threadIdx.x;

	// 注: smem的大小需要在编译期确定
	//	-> template
	// 3080: L1 cache + smem = 8704KB
	//	单个SM: 100KB(max)
	//	单个block: 48KB(max)
	float temp = 0.0f;
	__shared__ float a_shared[BLOCK_DIM][BLOCK_DIM];
	__shared__ float b_shared[BLOCK_DIM][BLOCK_DIM];
	for (int k = 0; k < K; k += blockDim.x) {
		a_shared[threadIdx.y][threadIdx.x] =
			A_ptr[threadIdx.x + k];
		b_shared[threadIdx.y][threadIdx.x] =
			B_ptr[(threadIdx.y + k) * N];
    __syncthreads();
    for (int i = 0; i < BLOCK_DIM; ++i) {
      temp += a_shared[threadIdx.y][i] * b_shared[i][threadIdx.x];
    }
    __syncthreads();
	}

  C[y * N + x] = temp;
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
	std::cout << "Max Shared Memory per Block: " << deviceProp.sharedMemPerBlock << " bytes" << std::endl;
	std::cout << "Max Shared Memory per SM: " << deviceProp.sharedMemPerMultiprocessor << " bytes" << std::endl;

	// CPU_segmm
	std::cout << " ============= CPU_segmm ============= " << std::endl;
	sgemm_CPU(A_buf_host, B_buf_host, C_buf_host_cpu,
		m, n, k);

	// GPU_segmm
	// sgemm: 二维 -> grid和block都是二维
	std::cout << " ============= GPU_segmm ============= " << std::endl;
	dim3 grid((m + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	sgemm_v3<BLOCK_SIZE><<<grid, block>>>(
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