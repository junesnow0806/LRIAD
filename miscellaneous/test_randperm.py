import torch

x = torch.rand(2708)
permutation = torch.randperm(x.shape[0])
idx = permutation[:int(2708 * 0.5)]
x[idx] = 0
idx = permutation[int(2708 * 0.5):int(2708 * 0.5) + int(2708 * 0.1)]
x[idx] = 1
idx = permutation[int(2708 * 0.5) + int(2708 * 0.1):]
x[idx] = 2
print(x)
print(torch.sum(torch.eq(x, 0)))
print(torch.sum(torch.eq(x, 1)))
print(torch.sum(torch.eq(x, 2)))
print(torch.sum(torch.eq(x, 0)) + torch.sum(torch.eq(x, 1)) + torch.sum(torch.eq(x, 2)))
