import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from vfl.data_utils import party_partition
from vfl.models import GCN, DNN

dataset = Planetoid(root='./data', name='Cora')
host_data, _ = party_partition(dataset[0], 0.1)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = GCN(host_data.num_node_features, out_features=64).to(device)
top = DNN(64, 16, dataset.num_classes).to(device)
host_data = host_data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
optimizer_top = torch.optim.Adam(top.parameters(), lr=0.001, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

model.train()
for epoch in range(300):
    optimizer.zero_grad()
    optimizer_top.zero_grad()
    out = model(host_data)
    out = top(out)
    loss = criterion(out[host_data.train_mask], host_data.y[host_data.train_mask])
    loss.backward()
    optimizer.step()
    optimizer_top.step()

    if (epoch+1) % 50 == 0:
        print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}')

model.eval()
top.eval()
pred = F.softmax(top(model(host_data)), dim=1).argmax(dim=1)
correct = (pred[host_data.test_mask] == host_data.y[host_data.test_mask]).sum()
acc = int(correct) / int(host_data.test_mask.sum())
print('Accuracy: {:.4f}'.format(acc))
