import torch
import torch.nn.functional as F
from torchmetrics import AUROC

def compute_loss(top_model, v_top, v_tar, v_adv):
    return torch.norm(v_top - top_model(torch.cat([v_tar, v_adv])), p=2)

def latent_representation_approximation(top_model, v_top, v_adv, tar_latent_features, epsilon, lr):
    """
    Args:
    -----
        - top_model: the top GNN model
        - v_top: the latent representation of the top GNN model, shape: (num_nodes, num_classes)
        - v_adv: the latent representation of the adversary bottom model, shape: (num_nodes, adv_latent_features)
        - tar_latent_features(int): the shape of the target latent representation of a sample/node
        - epsilon: the threshold of the loss

    Returns:
    --------
        approximated_v_tar: the approximated target latent representation, shape: (num_nodes, tar_latent_features)
    """
    v_tar = []
    v_top_no_grad = v_top.detach()
    v_adv_no_grad = v_adv.detach()
    num_nodes = v_top.shape[0]
    print("device: ", v_top.device)
    print(f"{num_nodes} samples to be approximated")
    for sample_idx in range(num_nodes):
        v_tar_element = torch.zeros(tar_latent_features, dtype=v_top.dtype, device=v_top.device, requires_grad=True)
        loss = compute_loss(top_model, v_top_no_grad[sample_idx], v_tar_element, v_adv_no_grad[sample_idx])

        iter_count = 0
        while True:
            iter_count += 1
            grad = torch.autograd.grad(loss, v_tar_element, create_graph=True, retain_graph=True)[0]
            # hessian = torch.autograd.grad(grad, v_tar_element, grad_outputs=torch.ones_like(v_tar_element))[0]
            # v_tar_element = v_tar_element - grad / hessian
            v_tar_element = v_tar_element - grad * lr
            old_loss = loss
            loss = compute_loss(top_model, v_top_no_grad[sample_idx], v_tar_element, v_adv_no_grad[sample_idx])
            if torch.abs(loss - old_loss) <= epsilon or iter_count >= 100:
                # print(iter_count)
                break

        v_tar.append(v_tar_element)
        if len(v_tar) % 100 == 0:
            print(f"{len(v_tar)} samples approximated")

    return torch.stack(v_tar).detach()

def lri_attack(v: torch.Tensor, delta: float):
    """Use the latent representation to attack the target model, i.e., predict the existence of relations.

    Args:
    ----
        v(torch.Tensor): the latent representation used
        delta(float): the threshold of the distance between two nodes to be considered as a relation

    Returns:
    -------
        Q(torch.Tensor): the predicted existence of relations, i.e., the adjacency matrix
    """
    v_normalized = F.normalize(v, p=2, dim=1)
    Q = 1 - torch.matmul(v_normalized, v_normalized.t())
    mask = torch.le(Q, delta)
    Q[mask] = 1
    Q[~mask] = 0
    # set the diagonal elements to 0
    Q[torch.eye(Q.shape[0], dtype=torch.bool)] = 0
    return Q

def remove_isolated(adj: torch.Tensor, Q: torch.Tensor = None):
    degrees = adj.sum(dim=1)
    isolated = torch.where(degrees == 0)[0].tolist()
    if isolated:
        keep = [adj[i] for i in range(adj.shape[0]) if i not in isolated]
        adj = torch.stack(keep).t()
        keep = [adj[i] for i in range(adj.shape[0]) if i not in isolated]
        adj = torch.stack(keep).t()
        if Q is not None:
            keep = [Q[i] for i in range(Q.shape[0]) if i not in isolated]
            Q = torch.stack(keep).t()
            keep = [Q[i] for i in range(Q.shape[0]) if i not in isolated]
            Q = torch.stack(keep).t()
    if Q is not None:
        return adj, Q
    else:
        return adj

def compute_auc(v: torch.Tensor, edge_index: torch.Tensor):
    """Compute the AUC of the predicted existence of relations.

    Args:
    ----
        v(torch.Tensor): nodes' final latent representations, i.e., the output of the top model
        edge_index(torch.Tensor): the ground truth of the existence of relations, i.e., the adjacency matrix

    Returns:
    -------
        auc(float): the AUC score
    """
    v_normalized = F.normalize(v, p=2, dim=1)
    D = 1 - torch.matmul(v_normalized, v_normalized.t())  # the distance matrix
    # the smaller distance means the higher probability of the existence of a relation
    # so we need to reverse the distance matrix
    Q = 2 - D  # the possible max value of distance is 2
    ones = torch.ones(edge_index.shape[1], device=edge_index.device)
    sparse = torch.sparse_coo_tensor(edge_index, ones, Q.shape)
    adj = sparse.to_dense()
    adj, Q = remove_isolated(adj.cpu(), Q.cpu())
    auroc = AUROC(task="binary")
    return auroc(Q, adj).item()
