import torch

if "__main__" == __name__:
    x = torch.randn(1, 1, 1024, 1024, device="cuda", dtype=torch.float32)

    m = torch.nn.Upsample(scale_factor=2.0, mode="nearest")

    y = m(x)