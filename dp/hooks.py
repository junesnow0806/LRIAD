import torch
from torch_geometric.nn import GCNConv
from opt_einsum import contract

from .utils import calculate_propagated_x


def capture_input_Linear(module: torch.nn.Linear, input: tuple, output: torch.Tensor):
    if len(input) != 1:
        raise ValueError("input must be a tuple of length 1")
    module.x = input[0].detach().clone()

def capture_gradoutput_Linear(module: torch.nn.Linear, grad_input: tuple, grad_output: tuple):
    if len(grad_output) != 1:
        raise ValueError("grad_output must be a tuple of length 1")
    module.grad_output = grad_output[0].detach().clone()
    module.per_sample_grad_weight = contract("n...i,n...j->nij", module.grad_output, module.x)
    if module.bias is not None:
        module.per_sample_grad_bias = module.grad_output.clone()

def capture_input_GCNConv(module: GCNConv, input: tuple, output: torch.Tensor):
    if len(input) != 2:
        raise ValueError("input must be a tuple of length 2")
    x = input[0].detach().clone()
    edge_index = input[1].detach().clone()
    module.propagated_x = calculate_propagated_x(x, edge_index)

def capture_gradoutput_GCNConv(module: GCNConv, grad_input: tuple, grad_output: tuple):
    if len(grad_output) != 1:
        raise ValueError("grad_output must be a tuple of length 1")
    module.grad_output = grad_output[0].detach().clone()
    module.per_sample_grad_weight = contract("n...i,n...j->nij", module.grad_output, module.propagated_x)
    if module.bias is not None:
        module.per_sample_grad_bias = module.grad_output.clone()
