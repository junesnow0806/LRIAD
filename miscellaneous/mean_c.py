import torch
from torch_geometric.datasets import Planetoid

from vfl.data_utils import party_partition
from vfl.models import DNN, GCN

# load data
dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]
print("Cora info: ", data)

hr = 0.5
print("host feature ratio: ", hr)
host_data, gx = party_partition(data, hr)

host_bottom_dim = 32
guest_bottom_dim = 32
host_top_dim = host_bottom_dim + guest_bottom_dim

max_grad_norm_sum = 0

for i in range(10):
    host_bottom = GCN(host_data.num_node_features, hidden_features=128, out_features=host_bottom_dim)
    guest_bottom = DNN(gx.shape[1], 128, guest_bottom_dim)
    host_top = DNN(host_top_dim, 32, dataset.num_classes)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    host_top = host_top.to(device)
    host_bottom = host_bottom.to(device)
    guest_bottom = guest_bottom.to(device)
    host_data = host_data.to(device)
    gx = gx.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    host_top_optimizer = torch.optim.Adam(host_top.parameters(), lr=0.001)
    host_bottom_optimizer = torch.optim.Adam(host_bottom.parameters(), lr=0.001)
    guest_bottom_optimizer = torch.optim.Adam(guest_bottom.parameters(), lr=0.001)

    host_top.train()
    host_bottom.train()
    guest_bottom.train()
    meet_max_grad_norm = 0
    for epoch in range(300):
        host_top_optimizer.zero_grad()
        host_bottom_optimizer.zero_grad()
        guest_bottom_optimizer.zero_grad()
        host_out = host_bottom(host_data)

        guest_out = guest_bottom(gx)
        out = torch.cat([host_out, guest_out], dim=1)
        out = host_top(out)
        loss = criterion(out[host_data.train_mask], host_data.y[host_data.train_mask])
        loss.backward()

        for param in host_top.parameters():
            if torch.norm(param.grad) > meet_max_grad_norm:
                meet_max_grad_norm = torch.norm(param.grad)

        host_top_optimizer.step()
        host_bottom_optimizer.step()
        guest_bottom_optimizer.step()

        # if (epoch+1) % 50 == 0:
        #     print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}')

    print("meet_max_grad_norm: ", meet_max_grad_norm)
    max_grad_norm_sum += meet_max_grad_norm

print("average meet_max_grad_norm: ", max_grad_norm_sum / 10)
