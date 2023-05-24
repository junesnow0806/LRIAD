import torch

x = torch.rand(3, requires_grad=True)
y = x ** 3 + torch.sin(x)
dy = 3 * x ** 2 + torch.cos(x)
d2y = 6 * x - torch.sin(x)

dydx = torch.autograd.grad(y, x, grad_outputs=torch.ones(x.shape), create_graph=True, retain_graph=True)[0]
print(dydx)
print(dy)

d2ydx2 = torch.autograd.grad(dydx, x, grad_outputs=torch.ones(x.shape))[0]
print(d2ydx2)
print(d2y)
