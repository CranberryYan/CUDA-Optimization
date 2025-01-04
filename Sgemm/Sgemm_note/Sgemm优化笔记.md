参考资料：cuda入门的正确姿势：how to optimize gemm
https://zhuanlan.zhihu.com/p/478846788
![[Pasted image 20241228151906.png]]

V0:
	![[Pasted image 20241228161307.png]]
	![[Pasted image 20241228162044.png]]
	**重复读取：**
	每次计算一个结果, 需要循环k次, 且这k次需要的元素全部要去global memory上读取, 而且在算C矩阵的[0, 0]和[0, 1]时, 其实需要的是A矩阵的同一行, 但是还是会重新取global memory上读取；
	read: 2 * M * N * K
	write: M * N
V1:
![[Pasted image 20241229131529.png]]
	使用shared_memory, 优化重复读取
	eg：
		把A矩阵的第0行存入shared_memory, 在计算C矩阵的[0, x]时, 全部可以直接去shared_memory寻找所需要的A矩阵的元素
		把B矩阵的第0列存入shared_memory, 在计算C矩阵的[x, 0]时, 全部可以直接去shared_memory寻找所需要的B矩阵的元素
	read: KNM * (1/bn + 1/bm)
	write: M * N
v2:
![alt text](image.png)
	v1的问题, 强行的将整个行/列全部塞入smem, 但是smem的大小是有限的(48KB/100KB),
	极大的限制了矩阵shape, 没有应用价值
	因此, 使用分块计算
v3:
![alt text](image-1.png)
	before: 不管是从Global还是Shared, 核心循环中的三条指令,
	两条load, 一条FMA -> 计算指令占比低(仅占1/3) -> 增加计算指令
	after: 四条load, 四条FMA -> 提高计算指令占比(占1/2)
v4:
![alt text](image-2.png)
	v3需要读取四个元素 -> 向量化
	v3的四个元素是离散的(stride: threadNum), 但是向量化读取要求是连续的2/4个元素
	v3的四个元素, 其实还是四条load指令, 从Global拿四次数据 ->
	向量化, 一次拿四个数(内存连续) -> 只有一条load指令
	TODO: 对B矩阵进行Transpose