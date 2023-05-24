import torch
from torch_geometric.data import Data


def party_partition(data, host_ratio, train_ratio=0.6):
    num_nodes = data.x.shape[0]
    num_features = data.x.shape[1]
    hc = int(num_features * host_ratio)
    hx = data.x[:, :hc]
    gx = data.x[:, hc:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:int(num_nodes * train_ratio)] = True
    test_mask[int(num_nodes * train_ratio):] = True

    host_data = Data(
        x=hx,
        edge_index=data.edge_index.clone(),
        edge_attr=data.edge_attr.clone() if data.edge_attr is not None else None,
        pos=data.pos.clone() if data.pos is not None else None,
        y=data.y.clone(),
        train_mask=train_mask,
        test_mask=test_mask,
    )

    return host_data, gx
