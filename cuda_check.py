import torch

print("Torch:", torch.__version__)
print("CUDA Build:", torch.version.cuda)
print("CUDA Available:", torch.cuda.is_available())