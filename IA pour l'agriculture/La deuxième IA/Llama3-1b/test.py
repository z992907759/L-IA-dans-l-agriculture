import torch
print("MPS available:", torch.backends.mps.is_available())
print("MPS built-in:", torch.backends.mps.is_built())