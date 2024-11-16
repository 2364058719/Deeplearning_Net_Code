import torch

x = torch.arange(12)
print(x)
print(x.shape)
print(x.numel())

X = x.reshape(3, 4)
print(X)

print(torch.zeros(2, 3, 4))
print(torch.ones(2, 3, 4))
