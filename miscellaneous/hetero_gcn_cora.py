import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

from dp.utils import (
    calculate_noise_multiplier,
    clip_and_aggregate,
    add_noise,
)
from dp.hooks import (
    capture_input_Linear, 
    capture_gradoutput_Linear,
    capture_input_GCNConv,
    capture_gradoutput_GCNConv,
)
from lri.functional import (
    compute_auc,
    latent_representation_approximation,
)
from vfl.data_utils import party_partition
from vfl.models import DNN, GCN

# load data
dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]
print("Cora info: ", data)

hr = 0.5
print("host feature ratio: ", hr)
host_data, gx = party_partition(data, hr)
train_mask = host_data.train_mask
test_mask = host_data.test_mask
print(train_mask.sum(), test_mask.sum())
host_bottom_dim = 32
guest_bottom_dim = 32
host_top_dim = host_bottom_dim + guest_bottom_dim
host_bottom = GCN(host_data.num_node_features, hidden_features=128, out_features=host_bottom_dim)
guest_bottom = DNN(gx.shape[1], 128, guest_bottom_dim)
host_top = DNN(host_top_dim, 32, dataset.num_classes)

dp = False
# settings for DP-SGD calculation
if dp:
    host_bottom.conv1.register_forward_hook(capture_input_GCNConv)
    host_bottom.conv1.register_full_backward_hook(capture_gradoutput_GCNConv)
    host_bottom.conv2.register_forward_hook(capture_input_GCNConv)
    host_bottom.conv2.register_full_backward_hook(capture_gradoutput_GCNConv)
    host_top.linear1.register_forward_hook(capture_input_Linear)
    host_top.linear1.register_full_backward_hook(capture_gradoutput_Linear)
    host_top.linear2.register_forward_hook(capture_input_Linear)
    host_top.linear2.register_full_backward_hook(capture_gradoutput_Linear)

train = True
test_appr = False
load = False
save = False
if load and not dp:
    host_top.load_state_dict(torch.load('checkpoints/cora/host_top.pth'))
    host_bottom.load_state_dict(torch.load('checkpoints/cora/host_bottom.pth'))
    guest_bottom.load_state_dict(torch.load('checkpoints/cora/guest_bottom.pth'))
elif load and dp:
    host_top.load_state_dict(torch.load('checkpoints/cora/host_top_dp.pth'))
    host_bottom.load_state_dict(torch.load('checkpoints/cora/host_bottom_dp.pth'))
    guest_bottom.load_state_dict(torch.load('checkpoints/cora/guest_bottom_dp.pth'))

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
host_top = host_top.to(device)
host_bottom = host_bottom.to(device)
guest_bottom = guest_bottom.to(device)
host_data = host_data.to(device)
gx = gx.to(device)

# dp budget
epsilon = 0.9
delta = 1e-5
max_grad_norm = 0.2
noise_multiplier = calculate_noise_multiplier(epsilon, delta)
print(f"using epsilon={epsilon}, delta={delta}, sigma={noise_multiplier}, C={max_grad_norm}")

# training
if dp:
    print("DP is on")
if train:
    criterion = torch.nn.CrossEntropyLoss()
    host_top_optimizer = torch.optim.Adam(host_top.parameters(), lr=0.001)
    host_bottom_optimizer = torch.optim.Adam(host_bottom.parameters(), lr=0.001)
    guest_bottom_optimizer = torch.optim.Adam(guest_bottom.parameters(), lr=0.001)

    host_top.train()
    host_bottom.train()
    guest_bottom.train()
    meet_max_grad_norm = 0

    if dp:
        noise_multiplier = calculate_noise_multiplier(epsilon, delta)

    for epoch in range(300):
        host_top_optimizer.zero_grad()
        host_bottom_optimizer.zero_grad()
        guest_bottom_optimizer.zero_grad()
        host_out = host_bottom(host_data)

        guest_out = guest_bottom(gx)
        out = torch.cat([host_out, guest_out], dim=1)
        out = host_top(out)
        loss = criterion(out[train_mask], host_data.y[train_mask])
        loss.backward()

        for param in host_top.parameters():
            if torch.norm(param.grad) > meet_max_grad_norm:
                meet_max_grad_norm = torch.norm(param.grad)

        if dp:
            for module in host_bottom.modules():
                if isinstance(module, GCNConv):
                    grad_weight = clip_and_aggregate(module.per_sample_grad_weight, max_grad_norm)
                    grad_weight = add_noise(grad_weight, noise_multiplier, max_grad_norm)
                    module.lin.weight.grad = grad_weight
                    if module.bias is not None:
                        grad_bias = clip_and_aggregate(module.per_sample_grad_bias, max_grad_norm)
                        grad_bias = add_noise(grad_bias, noise_multiplier, max_grad_norm)
                        module.bias.grad = grad_bias
            for module in host_top.modules():
                if isinstance(module, torch.nn.Linear):
                    grad_weight = clip_and_aggregate(module.per_sample_grad_weight, max_grad_norm)
                    grad_weight = add_noise(grad_weight, noise_multiplier, max_grad_norm)
                    module.weight.grad = grad_weight
                    if module.bias is not None:
                        grad_bias = clip_and_aggregate(module.per_sample_grad_bias, max_grad_norm)
                        grad_bias = add_noise(grad_bias, noise_multiplier, max_grad_norm)
                        module.bias.grad = grad_bias

        host_top_optimizer.step()
        host_bottom_optimizer.step()
        guest_bottom_optimizer.step()

        if (epoch+1) % 50 == 0:
            print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}')

if save and not dp:
    torch.save(host_top.state_dict(), 'checkpoints/cora/host_top.pth')
    torch.save(host_bottom.state_dict(), 'checkpoints/cora/host_bottom.pth')
    torch.save(guest_bottom.state_dict(), 'checkpoints/cora/guest_bottom.pth')
elif save and dp:
    torch.save(host_top.state_dict(), 'checkpoints/cora/host_top_dp.pth')
    torch.save(host_bottom.state_dict(), 'checkpoints/cora/host_bottom_dp.pth')
    torch.save(guest_bottom.state_dict(), 'checkpoints/cora/guest_bottom_dp.pth')

# testing
print("meet_max_grad_norm: ", meet_max_grad_norm)
host_top.eval()
host_bottom.eval()
guest_bottom.eval()
with torch.no_grad():
    pred = F.softmax(host_top(torch.cat([host_bottom(host_data), guest_bottom(gx)], dim=1)), dim=1).argmax(dim=1)
    correct = (pred[test_mask] == host_data.y[test_mask]).sum()
    acc = int(correct) / int(test_mask.sum())
    print(f'Accuracy: {acc:.4f}')

# conduct LRI
#! should not perform latent representation approximation under the `with torch.no_grad()` context
with torch.no_grad():
    v_tar = host_bottom(host_data)
    v_adv = guest_bottom(gx)
    v_top = host_top(torch.cat([v_tar, v_adv], dim=1))

print("Attack v_tar AUC: ", compute_auc(v_tar, host_data.edge_index))
print("Attack v_adv AUC: ", compute_auc(v_adv, host_data.edge_index))
print("Attack gx AUC: ", compute_auc(gx, host_data.edge_index))
print("Attack v_top AUC: ", compute_auc(v_top, host_data.edge_index))

if test_appr:
    print("start v_tar latent representation approximation")
    import time
    start = time.time()
    appr_v_tar = latent_representation_approximation(host_top, v_top, v_adv, host_bottom_dim, 0.01, 0.1)
    end = time.time()
    print(f"Latent representation approximation time: {end-start} sec")
    print("Attack appr_v_tar AUC: ", compute_auc(appr_v_tar, host_data.edge_index))
