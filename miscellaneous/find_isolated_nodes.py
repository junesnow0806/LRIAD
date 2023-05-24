import torch
from torch_geometric.datasets import Planetoid, Amazon


# dataset = Planetoid(root='./data', name='Cora')
# dataset = Planetoid(root='./data', name='CiteSeer')
# dataset = Amazon(root='./data', name='Computers')
dataset = Amazon(root='./data', name='Photo')
data = dataset[0]
print("dataset info: ", data)

edge_index = data.edge_index
ones = torch.ones(edge_index.shape[1])
sparse = torch.sparse_coo_tensor(edge_index, ones, size=[data.num_nodes, data.num_nodes])
Adj = sparse.to_dense()
degrees = Adj.sum(dim=1)
# isolated = torch.nonzero(degrees == 0).flatten()
isolated = torch.where(degrees == 0)
print(isolated)
# isolated = isolated.tolist()
# keep = []
# for i in range(Adj.shape[0]):
#     if i not in isolated:
#         keep.append(Adj[i])
# new_adj = torch.stack(keep)
# print(new_adj.shape)
# new_adj = new_adj.t()
# keep = []
# for i in range(new_adj.shape[0]):
#     if i not in isolated:
#         keep.append(new_adj[i])
# new_adj = torch.stack(keep)
# print(new_adj.shape)
# new_adj = new_adj.t()
# degrees = new_adj.sum(dim=1)
# isolated = torch.nonzero(degrees == 0).flatten()
# print(isolated, isolated.shape)
