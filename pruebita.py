import torch

# Attempt to allocate a tensor on the GPU
try:
    x = torch.rand(3, 3).cuda()
    print("CUDA is working!")
except Exception as e:
    print(f"CUDA Error: {e}")
