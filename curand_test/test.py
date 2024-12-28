import torch

def generate_uniform_tensor(device, size, seed):
    # 设置随机数种子
    torch.manual_seed(seed)
    # 在指定设备上生成均匀分布的张量
    tensor = torch.empty(size, device=device).uniform_(0, 1)
    return tensor

def compare_tensors(tensor1, tensor2, tolerance=1e-6):
    # 检查张量大小是否相同
    if tensor1.size() != tensor2.size():
        return False
    
    # 检查张量的元素是否在容差范围内相等
    return torch.allclose(tensor1, tensor2, atol=tolerance)

# 张量大小
size = (1000, 1000)
# 随机数种子
seed = 123

# 在CPU上生成均匀分布的张量
cpu_tensor = generate_uniform_tensor('cpu', size, seed)
# 在GPU上生成均匀分布的张量
gpu_tensor = generate_uniform_tensor('cuda', size, seed)
gpu_tensor2 = generate_uniform_tensor('cuda', size, seed)

# 将GPU张量复制回CPU以进行比较
gpu_tensor_cpu = gpu_tensor.cpu()
gpu_tensor_cpu2 = gpu_tensor2.cpu()

# 比较张量是否接近
tensors_are_close = compare_tensors(cpu_tensor, gpu_tensor_cpu)

print(f"Tensors are {'close' if tensors_are_close else 'not close'}")

# 如果需要，可以打印一些张量值进行可视化检查
print("CPU Tensor:", cpu_tensor[:5, :5])
print("GPU Tensor:", gpu_tensor_cpu[:5, :5])
print("GPU Tensor:", gpu_tensor_cpu2[:5, :5])
