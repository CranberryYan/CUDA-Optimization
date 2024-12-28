参考资料：cuda入门的正确姿势：how to optimize gemm
https://zhuanlan.zhihu.com/p/478846788
![[Pasted image 20241228151906.png]]

V0:
	![[Pasted image 20241228161307.png]]
	![[Pasted image 20241228162044.png]]
	**重复读取：**
	每次计算一个结果，需要循环k次，且这k次需要的元素全部要去global memory上读取，而且在算C矩阵的[0, 0]和[0, 1]时，其实需要的是A矩阵的同一行，但是还是会重新取global memory上读取；
	