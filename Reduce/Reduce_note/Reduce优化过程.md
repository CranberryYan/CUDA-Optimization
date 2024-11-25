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