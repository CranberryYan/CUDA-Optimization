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
