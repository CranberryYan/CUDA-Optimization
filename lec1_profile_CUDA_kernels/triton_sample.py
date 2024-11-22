from sympy import true
import torch
import triton
import triton.language as tl

# 装饰器: 使用triton的jit编译器来编译这个函数
# BLCOK_SIZE: thread数量, 用于处理的元素数量, 使用 tl.constexpr 指定为编译时常量
@triton.jit
def square_kernel(output_ptr, input_ptr, output_row_stride, input_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)  # 获取当前程序id, block_id
    row_start_ptr = input_ptr + row_idx * input_row_stride;
    col_offsets = tl.arange(0, BLOCK_SIZE)  # thread_id
    input_ptrs = row_start_ptr + col_offsets
    
    # 从GPU中加载数据
    # mask: BLOCK_SIZE = triton.next_power_of_2(n_cols) 可能大于 n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    
    # 计算加载元素的平方
    square_output = row * row
    
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, square_output, mask=col_offsets < n_cols)

def square(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols) # 找到最接近的大于所给参数的2^n eg: n_cols: 20 -> 返回32
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    
    y = torch.empty_like(x)
    
    # []: grid_size
    # (n_rows, ): 一维grid
    square_kernel[(n_rows, )](y, x, y.stride(0), x.stride(0), n_cols, BLOCK_SIZE)
    
    return y


if __name__ == "__main__":
    x = torch.randn(1823, 781, device='cuda')
    y_triton = square(x)
    y_torch = torch.square(x)
    
    res = torch.allclose(y_triton, y_torch)
    if (res == true):
        print("passed")
    else:
        print("mismatch")
        for i in range(10):
            print(y_torch[i])
            print(y_triton[i])
