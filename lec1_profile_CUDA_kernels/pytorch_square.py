import torch


# CUDA is ASYNC, so cant use python time module
# func: 函数指针
# input: 函数指针所接受的参数
def time_pytorch_function(func, input):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # warmup
    for _ in range(5):
        func(input)
    
    start.record()
    func(input)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end)

def square_2(a):
    return a ** 2

def square_3(a):
    return a ** 3


if __name__ == "__main__":

    # b: [10000, 10000]
    b = torch.randn(10000, 10000).cuda()
    time_b_torch = time_pytorch_function(torch.square, b)
    time_b_square_2 = time_pytorch_function(square_2, b)
    time_b_square_3 = time_pytorch_function(square_3, b)
    print(f"time_b_torch: {time_b_torch}")
    print(f"time_b_square_2: {time_b_square_2}")
    print(f"time_b_square_3: {time_b_square_3}")

    print("=============")
    print("Profiling square_2")
    print("=============")
    with torch.profiler.profile() as prof:
        square_2(b)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print("=============")
    print("Profiling square_3")
    print("=============")
    with torch.profiler.profile() as prof:
        square_3(b)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print("==============")
    print("Profiling torch.square")
    print("==============")
    with torch.profiler.profile() as prof:
        torch.square(b)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

