import math
import torch


def calculate_noise_multiplier(epsilon, delta):
    return math.sqrt(2 * math.log(1.25 / delta)) / epsilon

def clip_and_aggregate(per_sample_grad, max_grad_norm):
    #  Clip per sample gradients
    for i in range(per_sample_grad.size(0)):
        grad_norm = torch.norm(per_sample_grad[i])
        if grad_norm > max_grad_norm:
            per_sample_grad[i] = per_sample_grad[i] * (max_grad_norm / grad_norm)
    return torch.sum(per_sample_grad, dim=0)

def add_noise(aggregated, noise_multiplier, max_grad_norm):
    noise = torch.normal(0, noise_multiplier * max_grad_norm, aggregated.shape, device=aggregated.device)
    return aggregated + noise

def calculate_propagated_x(x, edge_index):
    device = x.device
    num_nodes = x.shape[0]
    ones = torch.ones(edge_index.shape[1], device=device)
    sparse = torch.sparse_coo_tensor(edge_index, ones, torch.Size([num_nodes, num_nodes]))
    A = sparse.to_dense()
    I = torch.eye(num_nodes, device=device)
    A_hat = A + I
    D_hat = torch.diag(torch.sum(A_hat, dim=1) ** (-0.5))
    P = D_hat @ A_hat @ D_hat
    return P @ x

def compute_gamma(orig_auc, dp_auc, orig_acc, dp_acc):
    return ((orig_auc - dp_auc) / orig_auc) / ((orig_acc - dp_acc) / orig_acc)
