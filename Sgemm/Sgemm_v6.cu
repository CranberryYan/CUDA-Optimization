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
// v4:
//	向量化
//	Memory Throughput [Gbyte/second]	37.36
// v5:
//	内积 -> 外积
// 	Memory Throughput [Gbyte/second]	54.32
// v6:
//	内积 -> 外积 + 向量化 + 增加每个thread的工作量
// 	Memory Throughput [Gbyte/second]	76.28
#include <iostream>
#define STRIDE	 					2
#define BLOCK_SIZE 				16
#define NUM_PER_REG  			4
#define NUM_PER_THREAD  	16
#define M_NUM_PER_BLOCK 	64
#define M_NUM_PER_THREAD 	4
#define N_NUM_PER_BLOCK 	64
#define N_NUM_PER_THREAD 	4
#define K_NUM_PER_BLOCK 	64
#define K_NUM_PER_THREAD 	4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

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

template<unsigned int M_NUM_PER_BLOCK_, unsigned int N_NUM_PER_BLOCK_,
  unsigned int K_NUM_PER_BLOCK_, unsigned int NUM_PER_THREAD_,
	unsigned int NUM_PER_REG_>
__global__ void sgemm_v6(float *A, float *B, float *C,
	const int M, const int N, const int K) {
	float temp[NUM_PER_REG_][NUM_PER_REG_] = {0.0f};
	__shared__ float a_shared[K_NUM_PER_BLOCK_][M_NUM_PER_BLOCK_]; // 转置
	__shared__ float b_shared[K_NUM_PER_BLOCK_][N_NUM_PER_BLOCK_];
	float a_reg[NUM_PER_REG_] = {0.0f};
	float b_reg[NUM_PER_REG_] = {0.0f};

	// A矩阵偏移到所在的行
	float *A_ptr = A + (blockIdx.y * M_NUM_PER_BLOCK_) * K;
	// B矩阵偏移到所在的列
	float *B_ptr = B + blockIdx.x * N_NUM_PER_BLOCK_;
	// C矩阵偏移到所在的行+列
	float *C_ptr = C + (blockIdx.y * M_NUM_PER_BLOCK_) * K +
		blockIdx.x * N_NUM_PER_BLOCK_;

	// 遍历小块
	int ty = threadIdx.y;
	int tx = threadIdx.x;
	for (int k = 0; k < K; k += K_NUM_PER_BLOCK_) {
		// Shared赋值
		for (int i = 0; i < M_NUM_PER_THREAD; ++i) {
			FETCH_FLOAT4(a_shared[ty * M_NUM_PER_THREAD + i]
				[tx * K_NUM_PER_THREAD]) =
			FETCH_FLOAT4(A_ptr[(ty * M_NUM_PER_THREAD + i) * K + 
				k + tx * K_NUM_PER_THREAD]);
			FETCH_FLOAT4(b_shared[ty * N_NUM_PER_THREAD + i]
				[tx * K_NUM_PER_THREAD]) =
			FETCH_FLOAT4(B_ptr[(ty * N_NUM_PER_THREAD + k + i) * N +
				tx * K_NUM_PER_THREAD]);
		}
		__syncthreads();

		// 小块求解, 外积 -> A矩阵竖着读值, 无法向量化, B矩阵可以向量化
		for (int k_ = 0; k_ < K_NUM_PER_BLOCK_; ++k_) {
			a_reg[0] = a_shared[ty * M_NUM_PER_THREAD + 0][k_];
			a_reg[1] = a_shared[ty * M_NUM_PER_THREAD + 1][k_];
			a_reg[2] = a_shared[ty * M_NUM_PER_THREAD + 2][k_];
			a_reg[3] = a_shared[ty * M_NUM_PER_THREAD + 3][k_];
			FETCH_FLOAT4(b_reg[0]) =
				FETCH_FLOAT4(b_shared[k_][tx * N_NUM_PER_THREAD]);
			for (int m = 0; m < NUM_PER_REG_; ++m) {
				for (int n = 0; n < NUM_PER_REG_; ++n) {
					temp[m][n] += a_reg[m] * b_reg[n];
				}
			}
		}
		__syncthreads();
	}

	for (int m = 0; m < NUM_PER_REG_; ++m) {
		for (int n = 0; n < NUM_PER_REG_; ++n) {
			C_ptr[(ty * NUM_PER_REG_ + m) * N + tx * NUM_PER_REG_ + n] =
				temp[m][n];
		}
	}
}

int main() {
	// lhs: [M, K]
	// rhs: [K, N]
	const unsigned int m = 2048;
	const unsigned int n = 2048;
	const unsigned int k = 2048;

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
  //  girdDim: [16, 16]
  //  blockDim: [16, 16] -> 一个block有256个thread, 一个thread处理16个元素
	std::cout << " ============= GPU_segmm ============= " << std::endl;
	dim3 grid((n + N_NUM_PER_BLOCK - 1) / N_NUM_PER_BLOCK,
    (m + M_NUM_PER_BLOCK - 1) / M_NUM_PER_BLOCK);
	dim3 block(16, 16);
	sgemm_v6<M_NUM_PER_BLOCK, N_NUM_PER_BLOCK,
		K_NUM_PER_BLOCK, NUM_PER_THREAD, NUM_PER_REG>
    <<<grid, block>>>( A_buf_device, B_buf_device, C_buf_device, m, n, k);

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