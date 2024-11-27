# CUDA-Optimization
一、Compute基本使用方法及参数意义
triton文件生成并使用ui界面分析 .ncu-rep 流程:
![[Pasted image 20241122152717.png]]
源码:
![[Pasted image 20241122152806.png]]
Night Compute分析:
1. GPU Speed of Light Throughtput
![[Pasted image 20241122152919.png]]
	可以看出, L1 和 L2 利用率不高, 并且Memory > Compute, 说明是访存密集型算子
	eg: x: [H, W]
		计算次数: H * W   访存次数: H * W * 2

2. Memory Workload Analysis
![[Pasted image 20241122153243.png]]
![[Pasted image 20241122153356.png]]
	L1 和 L2 Hit Rate 不高, Max BandWidth很低(3080： 790GB/s)
	eg: x: [1823, 781] FP32
		数据量: 1823 * 781 * 32 / 8 / 1024 / 1024 = 5.43MB

3. Source Counters
![[Pasted image 20241122153633.png]]
	分支指令:
		一共7292个分支, 分支效率和平均发散分支都是0 -> 分支预测做得很好(因为确实没有分支)
	未合并的全局访问:
		有未合并的全局访问, 导致78384个多余的sector
	L2 未有效利用:
		问题出在24行
			tl.store(output_ptrs, square_output, mask=col_offsets < n_cols)
	Warp停滞
		问题处在17和24行(一个读一个存)
			row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
			tl.store(output_ptrs, square_output, mask=col_offsets < n_cols)
			![[Pasted image 20241122154828.png]]
1. Compute Workland Analysis
![[Pasted image 20241122154446.png]]
	计算pipeline没有被充分利用(因为是访存密集)

5. Launch Statistics
![[Pasted image 20241122154619.png]]

二、Reduce算子实现及其优化
1. Baseline
		![[Pasted image 20241124192231.png]]
		将以树状图的方式执行数据累加，由于没有Global级别的同步操作，将计算分成多个阶段的方式来避免Global级别的同步操作，代码如下：
		![[Pasted image 20241124202248.png]]
	Compute
	1.GPU Speed of Light Throughput
	![[Pasted image 20241124203031.png]]
	计算和访存效率还可以
	
	2.Memory Workload Analysis
	![[Pasted image 20241124203427.png]]
	带宽：144.33GB/S

	3.Sourcee Counters
	![[Pasted image 20241124203708.png]]
	一共 10616832个分支指令，分支效率是100%
	虽然有if-else分支，但是此时的if分支是满足的进入代码块，不满足的什么也不做，而不是进入其他代码块，因此不会造成太大的性能影响，但是还是会造成性能浪费，因为有很多空闲线程，所以还是要避免
	
改进措施：
	v1. 避免warp divergent，不使用%
	v2. 减少bank conflict
	v3. 减少Idel thread，提高利用率
	v4. 展开最后一个warp的for循环
	v5. 展开所有warp的for循环
	v6. 优化block数量
		v6.0 block_size: 1024
		v6.1 block_size: 2048
		v6.2 block_size: 512(效果最佳)
		分析：
			同一个任务，相同的数据量，grid大，block小和grid小，block大，有什么区别？
				Block大，grid小：
					优点：每个block有更多的thread，每个block共享一个shared memory，这些thread用shared memory进行通信，blcok内使用__syncthreads实现同步，开销小
					缺点：bank conflict
				Block小，grid大：
					优点：更多的block，更多SM
					缺点：smem利用率低，同步代价大
			同一个任务，相同的数据量，为什么不是thread越多越好？
				3080一共68个SM，一个block只能位于一个SM，一个SM可以有多个block，但是如果有太多的block(65535)，那会通过调度器来进行调度，但是本任务其实不需要太多的block，那调度开销就会影响性能。