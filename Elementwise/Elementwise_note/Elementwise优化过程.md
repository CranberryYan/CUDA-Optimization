1. Baseline
		无任何优化手段，每个thread处理一个数据
	compute
	1.GPU Speed of Ligth Throught
	![[Pasted image 20241127222502.png]]
	Memory Throughput：还可以
	2.Memory Workload Analysis
	![[Pasted image 20241127222637.png]]
	带宽也还可以
	
2. 向量化+shared memory
		![[Pasted image 20241202134911.png]]
		性能大幅度下降，不能将向量化读取的数据直接给shared memory，冲突
3. shared memory
	性能提升不明显

4. 向量化
	性能提升明显，带宽利用率: 92.28%，v1内存吞吐量: 675.51GB/s