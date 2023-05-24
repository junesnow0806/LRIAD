import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_features=256, out_features=16):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, out_features)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class DNN(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features, hidden_features)
        self.linear2 = torch.nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.linear2(x)
        return x
