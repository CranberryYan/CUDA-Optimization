import torch


if __name__ == "__main__":
    
    # Memory Throughput [Mbyte/s]	654.21
    """
    for i in range(len(t)):
        x[index[i]] += t[i]
    """
    x = torch.randn(32*1024*1024).to("cuda")
    t = torch.randn(15).to("cuda")
    index = torch.randint(0, 1024, (15,)).to("cuda") # [0, 1023]取15个索引
    x.index_add_(0, index, t)
    torch.cuda.synchronize()

    # # x: [32768, 1024]
    # # t: [15, 1024]
    # # index: [15,](1 -> 1023)
    # # x的某一行 + t的某一行(15行)
    # #   x[index[i]] += t[i]
    # x = torch.randn(32*1024, 1024).to("cuda")
    # t = torch.randn(15, 1024).to("cuda")
    # index = torch.randint(0, 32*1024, (15,)).to("cuda")
    # x.index_add_(0, index, t) # dim0: 32K
    # torch.cuda.synchronize()

    # # x: [32K, 1024]
    # # t: [1024, 1024]
    # # index: [1024](0 -> 32K)
    # # x中的某一行 + t的某一行(1024行)
    # x = torch.randn(32*1024, 1024).to("cuda")
    # t = torch.randn(1024, 1024).to("cuda")
    # index = torch.randint(0, 32*1024, (1024,)).to("cuda")
    # x.index_add_(0, index, t)
    # torch.cuda.synchronize()

    # # x: [32, 1024, 1024]
    # # t: [15, 1024, 1024]
    # # index: [15](0 -> 31)
    # x = torch.randn(32, 1024, 1024).to("cuda")
    # t = torch.randn(15, 1024, 1024).to("cuda")
    # index = torch.randint(0, 32, (15,)).to("cuda")
    # x.index_add_(0, index, t)
    # torch.cuda.synchronize()

    # # x: [32M,]
    # # t: [1024,]
    # # index: [1024](0 -> 32M)
    # # x中的某一个数(一共有1024个)加上t中的某一个数
    # #   x[index[i]] += t[i]
    # x = torch.randn(32*1024*1024).to("cuda")
    # t = torch.randn(1024).to("cuda")
    # index = torch.randint(0, 32*1024*1024, (1024,)).to("cuda")
    # x.index_add_(0, index, t)
    # torch.cuda.synchronize()
