import torch
from torch_geometric.datasets import Planetoid
from vfl.data_utils import teacher_partition, party_partition


data = Planetoid(root='./data', name='Cora')[0]
host_data, gx = party_partition(data, 0.9)
print(teacher_partition(host_data, 2, 0))
