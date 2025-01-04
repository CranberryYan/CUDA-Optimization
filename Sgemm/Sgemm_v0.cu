// v0: 
//	Compute (SM) Throughput [%]	92.10
//	Memory Throughput [Gbyte/second]	8.43
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
__global__ void sgemm_v0(float *A, float *B, float *C,
	const int M, const int N, const int K) {
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	// block_level
	// blockDim: [16, 16]
	// blockIdx: [0, 63]
	//	offset_stride: 16
	// 行的偏移(y轴)
	//	接下来通过偏移k, 遍历这一行
	int offset_row = blockIdx.y * blockDim.y * K;
	// 列的偏移(x轴)
	//	接下来通过偏移k*N, 遍历这一列
	int offset_col = blockIdx.x * blockDim.x;
	float *A_ptr = A + offset_row;
	float *B_ptr = B + offset_col;

	// thread_level
	float temp = 0.0f;
	for (int k = 0; k < K; ++k) {
		temp += A_ptr[threadIdx.y * K + k] *
			B_ptr[k * N + threadIdx.x];
	}
	int offset_C = y * N + x;
	C[offset_C] = temp;
}

int main() {
	// lhs: [M, K]
	// rhs: [K, N]
	printf("gemm_baseline\n");
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

	// CPU_segmm
	std::cout << " ============= CPU_segmm ============= " << std::endl;
	sgemm_CPU(A_buf_host, B_buf_host, C_buf_host_cpu,
		m, n, k);

	// GPU_segmm
	// sgemm: 二维 -> grid和block都是二维
	std::cout << " ============= GPU_segmm ============= " << std::endl;
	dim3 grid((m + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	sgemm_v0<<<grid, block>>>(A_buf_device, B_buf_device, C_buf_device,
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