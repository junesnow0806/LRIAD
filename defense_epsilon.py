import os
import pickle
import sys

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid, Amazon

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


dataset_name = sys.argv[1]
assert dataset_name in ['Cora', 'CiteSeer', 'Computers', 'Photo']
if dataset_name in ['Cora', 'CiteSeer']:
    dataset = Planetoid(root='./data', name=dataset_name)
else:
    dataset = Amazon(root='./data', name=dataset_name)
data = dataset[0]
print(f"{dataset_name} info: ", data)

host_bottom_dim = 32
guest_bottom_dim = 32
host_top_dim = host_bottom_dim + guest_bottom_dim

EPOCHS = 300
epsilon_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
delta = 1e-5
C = 0.2

acc_list = []
auc_tar_list = []
auc_adv_list = []
auc_gx_list = []
auc_top_list = []
auc_appr_list =[]

if os.path.exists(f"statistics/{dataset_name}/auc_appr_list_epsilon.pkl"):
    with open(f"statistics/{dataset_name}/acc_list_epsilon.pkl", "rb") as f:
        acc_list = pickle.load(f)
    with open(f"statistics/{dataset_name}/auc_tar_list_epsilon.pkl", "rb") as f:
        auc_tar_list = pickle.load(f)
    with open(f"statistics/{dataset_name}/auc_adv_list_epsilon.pkl", "rb") as f:
        auc_adv_list = pickle.load(f)
    with open(f"statistics/{dataset_name}/auc_gx_list_epsilon.pkl", "rb") as f:
        auc_gx_list = pickle.load(f)
    with open(f"statistics/{dataset_name}/auc_top_list_epsilon.pkl", "rb") as f:
        auc_top_list = pickle.load(f)
    with open(f"statistics/{dataset_name}/auc_appr_list_epsilon.pkl", "rb") as f:
        auc_appr_list = pickle.load(f)

stored_i = len(auc_appr_list) - 1

hr = 0.5
for i, epsilon in enumerate(epsilon_range):
    if i <= stored_i:
        continue

    print("epsilon: ", epsilon)
    sigma = calculate_noise_multiplier(epsilon, delta)
    host_data, gx = party_partition(data, hr)
    train_mask = host_data.train_mask
    test_mask = host_data.test_mask

    host_bottom = GCN(host_data.num_features, hidden_features=128, out_features=host_bottom_dim)
    guest_bottom = DNN(gx.shape[1], 128, guest_bottom_dim)
    host_top = DNN(host_top_dim, 32, dataset.num_classes)

    # settings for DP-SGD calculation
    hc1_forward_hook = host_bottom.conv1.register_forward_hook(capture_input_GCNConv)
    hc1_backward_hook = host_bottom.conv1.register_full_backward_hook(capture_gradoutput_GCNConv)
    hc2_forward_hook = host_bottom.conv2.register_forward_hook(capture_input_GCNConv)
    hc2_backward_hook = host_bottom.conv2.register_full_backward_hook(capture_gradoutput_GCNConv)
    hl1_forward_hook = host_top.linear1.register_forward_hook(capture_input_Linear)
    hl1_backward_hook = host_top.linear1.register_full_backward_hook(capture_gradoutput_Linear)
    hl2_forward_hook = host_top.linear2.register_forward_hook(capture_input_Linear)
    hl2_backward_hook = host_top.linear2.register_full_backward_hook(capture_gradoutput_Linear)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    host_bottom = host_bottom.to(device)
    guest_bottom = guest_bottom.to(device)
    host_top = host_top.to(device)
    gx = gx.to(device)
    host_data = host_data.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    host_top_optimizer = torch.optim.Adam(host_top.parameters(), lr=0.001)
    host_bottom_optimizer = torch.optim.Adam(host_bottom.parameters(), lr=0.001)
    guest_bottom_optimizer = torch.optim.Adam(guest_bottom.parameters(), lr=0.001)

    if not os.path.exists(f"checkpoints/{dataset_name}/guest_bottom_dp_e{epsilon}.pth"):
        host_top.train()
        host_bottom.train()
        guest_bottom.train()
        print("training...")
        for epoch in range(EPOCHS):
            host_top_optimizer.zero_grad()
            host_bottom_optimizer.zero_grad()
            guest_bottom_optimizer.zero_grad()
            host_out = host_bottom(host_data)
            guest_out = guest_bottom(gx)
            out = torch.cat((host_out, guest_out), dim=1)
            out = host_top(out)
            loss = criterion(out[train_mask], host_data.y[train_mask])

            loss.backward()

            # DP
            for module in host_bottom.modules():
                if isinstance(module, GCNConv):
                    grad_weight = clip_and_aggregate(module.per_sample_grad_weight, C)
                    grad_weight = add_noise(grad_weight, sigma, C)
                    module.lin.weight.grad = grad_weight
                    if module.bias is not None:
                        grad_bias = clip_and_aggregate(module.per_sample_grad_bias, C)
                        grad_bias = add_noise(grad_bias, sigma, C)
                        module.bias.grad = grad_bias
            for module in host_top.modules():
                if isinstance(module, torch.nn.Linear):
                    grad_weight = clip_and_aggregate(module.per_sample_grad_weight, C)
                    grad_weight = add_noise(grad_weight, sigma, C)
                    module.weight.grad = grad_weight
                    if module.bias is not None:
                        grad_bias = clip_and_aggregate(module.per_sample_grad_bias, C)
                        grad_bias = add_noise(grad_bias, sigma, C)
                        module.bias.grad = grad_bias

            host_top_optimizer.step()
            host_bottom_optimizer.step()
            guest_bottom_optimizer.step()
            if (epoch+1) % 10 == 0:
                print(f"epoch {epoch+1} done")
        torch.save(host_top.state_dict(), f"checkpoints/{dataset_name}/host_top_dp_e{epsilon}.pth")
        torch.save(host_bottom.state_dict(), f"checkpoints/{dataset_name}/host_bottom_dp_e{epsilon}.pth")
        torch.save(guest_bottom.state_dict(), f"checkpoints/{dataset_name}/guest_bottom_dp_e{epsilon}.pth")
    else:
        host_top.load_state_dict(torch.load(f"checkpoints/{dataset_name}/host_top_dp_e{epsilon}.pth"))
        host_bottom.load_state_dict(torch.load(f"checkpoints/{dataset_name}/host_bottom_dp_e{epsilon}.pth"))
        guest_bottom.load_state_dict(torch.load(f"checkpoints/{dataset_name}/guest_bottom_dp_e{epsilon}.pth"))
    
    host_top.eval()
    host_bottom.eval()
    guest_bottom.eval()
    hc1_forward_hook.remove()
    hc1_backward_hook.remove()
    hc2_forward_hook.remove()
    hc2_backward_hook.remove()
    hl1_forward_hook.remove()
    hl1_backward_hook.remove()
    hl2_forward_hook.remove()
    hl2_backward_hook.remove()
    print("evaluating...")
    with torch.no_grad():
        pred = F.softmax(host_top(torch.cat([host_bottom(host_data), guest_bottom(gx)], dim=1)), dim=1).argmax(dim=1)
        correct = (pred[test_mask] == host_data.y[test_mask]).sum()
        acc = int(correct) / int(test_mask.sum())
    print("evaluation done")

    #! should not perform latent representation approximation under the `with torch.no_grad()` context
    print("calculating attack auc...")
    with torch.no_grad():
        v_tar = host_bottom(host_data)
        v_adv = guest_bottom(gx)
        v_top = host_top(torch.cat([v_tar, v_adv], dim=1))

        auc_tar = compute_auc(v_tar, host_data.edge_index)
        auc_adv = compute_auc(v_adv, host_data.edge_index)
        auc_gx = compute_auc(gx, host_data.edge_index)
        auc_top = compute_auc(v_top, host_data.edge_index)

    v_appr = latent_representation_approximation(host_top, v_top, v_adv, host_bottom_dim, 0.01, 0.1)
    auc_appr = compute_auc(v_appr, host_data.edge_index)

    acc_list.append(acc)
    auc_tar_list.append(auc_tar)
    auc_adv_list.append(auc_adv)
    auc_gx_list.append(auc_gx)
    auc_top_list.append(auc_top)
    auc_appr_list.append(auc_appr)
    print("acc: ", acc)
    print("auc_tar: ", auc_tar)
    print("auc_adv: ", auc_adv)
    print("auc_gx: ", auc_gx)
    print("auc_top: ", auc_top)
    print("auc_appr: ", auc_appr)

    print("saving results...")
    import pickle
    with open(f"statistics/{dataset_name}/acc_list_epsilon.pkl", "wb") as f:
        pickle.dump(acc_list, f)
    with open(f"statistics/{dataset_name}/auc_tar_list_epsilon.pkl", "wb") as f:
        pickle.dump(auc_tar_list, f)
    with open(f"statistics/{dataset_name}/auc_adv_list_epsilon.pkl", "wb") as f:
        pickle.dump(auc_adv_list, f)
    with open(f"statistics/{dataset_name}/auc_gx_list_epsilon.pkl", "wb") as f:
        pickle.dump(auc_gx_list, f)
    with open(f"statistics/{dataset_name}/auc_top_list_epsilon.pkl", "wb") as f:
        pickle.dump(auc_top_list, f)
    with open(f"statistics/{dataset_name}/auc_appr_list_epsilon.pkl", "wb") as f:
        pickle.dump(auc_appr_list, f)

print(f"defense change epsilon task of {dataset_name} all done")
