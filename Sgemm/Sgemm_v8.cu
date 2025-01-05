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
// 1024 * 1024
// 	Memory Throughput [Gbyte/second]	76.28
// 2048 * 2048
//	1.61ms
// 	Memory Throughput [Gbyte/second]	170.60
// 	Compute (SM) Throughput [%]	49.90
// v7:
//	转置 -> A矩阵也可以向量化
// 	Memory Throughput [Gbyte/second]	59.19
// v8:
//	乒乓(double buffer)
// 1024 * 1024
// 	Memory Throughput [Gbyte/second]	47.13
// 2048 * 2048
//	1.30ms
//	Memory Throughput [Gbyte/second]	75.41
// 	Compute (SM) Throughput [%]	59.82
#include <iostream>
#define NUMS_PER_Y_THREAD 8
#define NUMS_PER_X_THREAD	8
#define M_NUM_PER_BLOCK 	128
#define N_NUM_PER_BLOCK 	128
#define K_NUM_PER_BLOCK 	8
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
  unsigned int K_NUM_PER_BLOCK_, unsigned int NUMS_PER_Y_THREAD_,
	unsigned int NUMS_PER_X_THREAD_>
__global__ void sgemm_v8(float *A, float *B, float *C,
	const int M, const int N, const int K) {
	int ty = threadIdx.y;
	int tx = threadIdx.x;
	int tid = ty * blockDim.x + tx;
	float temp[NUMS_PER_Y_THREAD_][NUMS_PER_X_THREAD_] = {0.0f};
	float a_reg[NUMS_PER_Y_THREAD_] = {0.0f};
	float b_reg[NUMS_PER_X_THREAD_] = {0.0f};
	float a_load_reg[4] = {0.0f}; // 协助转置, 向量化读取 -> 4
	__shared__ float a_shared[2][K_NUM_PER_BLOCK_][M_NUM_PER_BLOCK_];
	__shared__ float b_shared[2][K_NUM_PER_BLOCK_][N_NUM_PER_BLOCK_];

	// A矩阵偏移到所在的行
	float *A_ptr = A + (blockIdx.y * M_NUM_PER_BLOCK_) * K;
	// B矩阵偏移到所在的列
	float *B_ptr = B + blockIdx.x * N_NUM_PER_BLOCK_;
	// C矩阵偏移到所在的行+列
	float *C_ptr = C + (blockIdx.y * M_NUM_PER_BLOCK_) * K +
		blockIdx.x * N_NUM_PER_BLOCK_;

	const int A_tile_thread_per_row = K_NUM_PER_BLOCK_ / 4; // 2
	const int B_tile_thread_per_row = N_NUM_PER_BLOCK_ / 4; // 32

	const int A_tile_tid_x = tid % A_tile_thread_per_row;
	const int A_tile_tid_y = tid / A_tile_thread_per_row;
	const int B_tile_tid_x = tid % B_tile_thread_per_row;
	const int B_tile_tid_y = tid / B_tile_thread_per_row;

	// 第一次乒
	FETCH_FLOAT4(a_load_reg[0]) =
		FETCH_FLOAT4(A_ptr[A_tile_tid_y * K + A_tile_tid_x * 4]);
	// A矩阵转置
	a_shared[0][A_tile_tid_x * 4 + 0][A_tile_tid_y] = a_load_reg[0];
	a_shared[0][A_tile_tid_x * 4 + 1][A_tile_tid_y] = a_load_reg[1];
	a_shared[0][A_tile_tid_x * 4 + 2][A_tile_tid_y] = a_load_reg[2];
	a_shared[0][A_tile_tid_x * 4 + 3][A_tile_tid_y] = a_load_reg[3];
	// B矩阵直接赋值
	FETCH_FLOAT4(b_shared[0][B_tile_tid_y][B_tile_tid_x * 4]) =
		FETCH_FLOAT4(B_ptr[B_tile_tid_y * N + B_tile_tid_x * 4]);
	__syncthreads();

	// 此时已经取了第一组shared的值, k初始值不再是0
	int pingpong = 1;
	for (int k = K_NUM_PER_BLOCK_; k < K; k += K_NUM_PER_BLOCK_) {
		FETCH_FLOAT4(a_load_reg[0]) =
			FETCH_FLOAT4(A_ptr[A_tile_tid_y * K + k + A_tile_tid_x * 4]);
		// A矩阵转置
		a_shared[pingpong][A_tile_tid_x * 4 + 0][A_tile_tid_y] = a_load_reg[0];
		a_shared[pingpong][A_tile_tid_x * 4 + 1][A_tile_tid_y] = a_load_reg[1];
		a_shared[pingpong][A_tile_tid_x * 4 + 2][A_tile_tid_y] = a_load_reg[2];
		a_shared[pingpong][A_tile_tid_x * 4 + 3][A_tile_tid_y] = a_load_reg[3];
		// B矩阵直接赋值
		FETCH_FLOAT4(b_shared[pingpong][B_tile_tid_y][B_tile_tid_x * 4]) =
			FETCH_FLOAT4(B_ptr[(k + B_tile_tid_y) * N + B_tile_tid_x * 4]);
		pingpong = pingpong ^ 1;
		// 小块求解
		for (int k_ = 0; k_ < K_NUM_PER_BLOCK_; ++k_) {
			// 此时使用ty/x, 因为此时在操作Shared, 是方形的
			FETCH_FLOAT4(a_reg[0]) =
				FETCH_FLOAT4(a_shared[pingpong][k_][ty * NUMS_PER_Y_THREAD_ + 0]);
			FETCH_FLOAT4(a_reg[4]) =
				FETCH_FLOAT4(a_shared[pingpong][k_][ty * NUMS_PER_Y_THREAD_ + 4]);
			FETCH_FLOAT4(b_reg[0]) =
				FETCH_FLOAT4(b_shared[pingpong][k_][tx * NUMS_PER_X_THREAD_ + 0]);
			FETCH_FLOAT4(b_reg[4]) =
				FETCH_FLOAT4(b_shared[pingpong][k_][tx * NUMS_PER_X_THREAD_ + 4]);
			for (int y = 0; y < NUMS_PER_Y_THREAD_; ++y) {
				for (int x = 0; x < NUMS_PER_X_THREAD_; ++x) {
					temp[y][x] += a_reg[y] * b_reg[x];
				}
			}
		}
		__syncthreads();
	}
	pingpong = pingpong ^ 1;
	for (int k_ = 0; k_ < K_NUM_PER_BLOCK_; ++k_) {
		// 此时使用ty/x, 因为此时在操作Shared, 是方形的
		FETCH_FLOAT4(a_reg[0]) =
			FETCH_FLOAT4(a_shared[pingpong][k_][ty * NUMS_PER_Y_THREAD_ + 0]);
		FETCH_FLOAT4(a_reg[4]) =
			FETCH_FLOAT4(a_shared[pingpong][k_][ty * NUMS_PER_Y_THREAD_ + 4]);
		FETCH_FLOAT4(b_reg[0]) =
			FETCH_FLOAT4(b_shared[pingpong][k_][tx * NUMS_PER_X_THREAD_ + 0]);
		FETCH_FLOAT4(b_reg[4]) =
			FETCH_FLOAT4(b_shared[pingpong][k_][tx * NUMS_PER_X_THREAD_ + 4]);
		for (int y = 0; y < NUMS_PER_Y_THREAD_; ++y) {
			for (int x = 0; x < NUMS_PER_X_THREAD_; ++x) {
				temp[y][x] += a_reg[y] * b_reg[x];
			}
		}
	}

	for (int y = 0; y < NUMS_PER_Y_THREAD_; ++y) {
		FETCH_FLOAT4(C_ptr[(ty * NUMS_PER_Y_THREAD_ + y) * N + tx * NUMS_PER_X_THREAD_ + 0]) =
			FETCH_FLOAT4(temp[y][0]);
		FETCH_FLOAT4(C_ptr[(ty * NUMS_PER_Y_THREAD_ + y) * N + tx * NUMS_PER_X_THREAD_ + 4]) =
			FETCH_FLOAT4(temp[y][4]);
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
	dim3 block(N_NUM_PER_BLOCK / NUMS_PER_X_THREAD,
		M_NUM_PER_BLOCK / NUMS_PER_Y_THREAD);
	sgemm_v8<M_NUM_PER_BLOCK, N_NUM_PER_BLOCK,
		K_NUM_PER_BLOCK, NUMS_PER_Y_THREAD, NUMS_PER_X_THREAD>
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
