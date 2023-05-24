import os
import pickle
import sys

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Amazon

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

hr_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
acc_list = []
auc_tar_list = []
auc_adv_list = []
auc_gx_list = []
auc_top_list = []
auc_appr_list =[]

if os.path.exists(f"statistics/{dataset_name}/auc_appr_list.pkl"):
    with open(f"statistics/{dataset_name}/acc_list.pkl", "rb") as f:
        acc_list = pickle.load(f)
    with open(f"statistics/{dataset_name}/auc_tar_list.pkl", "rb") as f:
        auc_tar_list = pickle.load(f)
    with open(f"statistics/{dataset_name}/auc_adv_list.pkl", "rb") as f:
        auc_adv_list = pickle.load(f)
    with open(f"statistics/{dataset_name}/auc_gx_list.pkl", "rb") as f:
        auc_gx_list = pickle.load(f)
    with open(f"statistics/{dataset_name}/auc_top_list.pkl", "rb") as f:
        auc_top_list = pickle.load(f)
    with open(f"statistics/{dataset_name}/auc_appr_list.pkl", "rb") as f:
        auc_appr_list = pickle.load(f)

stored_i = len(auc_appr_list) - 1

EPOCHS = 300

for i, hr in enumerate(hr_range):
    if i <= stored_i:
        continue

    print("hr: ", hr)
    host_data, gx = party_partition(data, hr)
    train_mask = host_data.train_mask
    test_mask = host_data.test_mask
    host_bottom = GCN(host_data.num_node_features, hidden_features=128, out_features=host_bottom_dim)
    guest_bottom = DNN(gx.shape[1], 128, guest_bottom_dim)
    host_top = DNN(host_top_dim, 32, dataset.num_classes)

    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    host_top = host_top.to(device)
    host_bottom = host_bottom.to(device)
    guest_bottom = guest_bottom.to(device)
    host_data = host_data.to(device)
    gx = gx.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    host_top_optimizer = torch.optim.Adam(host_top.parameters(), lr=0.001)
    host_bottom_optimizer = torch.optim.Adam(host_bottom.parameters(), lr=0.001)
    guest_bottom_optimizer = torch.optim.Adam(guest_bottom.parameters(), lr=0.001)

    if not os.path.exists(f"checkpoints/{dataset_name}/guest_bottom_hr{hr}.pth"):
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
            out = torch.cat([host_out, guest_out], dim=1)
            out = host_top(out)
            loss = criterion(out[train_mask], host_data.y[train_mask])

            loss.backward()
            host_top_optimizer.step()
            host_bottom_optimizer.step()
            guest_bottom_optimizer.step()
        torch.save(host_top.state_dict(), f"checkpoints/{dataset_name}/host_top_hr{hr}.pth")
        torch.save(host_bottom.state_dict(), f"checkpoints/{dataset_name}/host_bottom_hr{hr}.pth")
        torch.save(guest_bottom.state_dict(), f"checkpoints/{dataset_name}/guest_bottom_hr{hr}.pth")
        print("training done")
    else:
        host_top.load_state_dict(torch.load(f"checkpoints/{dataset_name}/host_top_hr{hr}.pth"))
        host_bottom.load_state_dict(torch.load(f"checkpoints/{dataset_name}/host_bottom_hr{hr}.pth"))
        guest_bottom.load_state_dict(torch.load(f"checkpoints/{dataset_name}/guest_bottom_hr{hr}.pth"))

    host_top.eval()
    host_bottom.eval()
    guest_bottom.eval()
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

    print(f"saving results of hr{hr}...")
    with open(f"statistics/{dataset_name}/acc_list.pkl", "wb") as f:
        pickle.dump(acc_list, f)
    with open(f"statistics/{dataset_name}/auc_tar_list.pkl", "wb") as f:
        pickle.dump(auc_tar_list, f)
    with open(f"statistics/{dataset_name}/auc_adv_list.pkl", "wb") as f:
        pickle.dump(auc_adv_list, f)
    with open(f"statistics/{dataset_name}/auc_gx_list.pkl", "wb") as f:
        pickle.dump(auc_gx_list, f)
    with open(f"statistics/{dataset_name}/auc_top_list.pkl", "wb") as f:
        pickle.dump(auc_top_list, f)
    with open(f"statistics/{dataset_name}/auc_appr_list.pkl", "wb") as f:
        pickle.dump(auc_appr_list, f)

print(f"attack task of {dataset_name} all done")
