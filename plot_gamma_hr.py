import pickle
import matplotlib.pyplot as plt
from dp.utils import compute_gamma


hr_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

fig = plt.figure(figsize=(25, 5))
axes = fig.subplots(1, 4)
datasets = ['Cora', 'CiteSeer', 'Computers', 'Photo']

for dataset, ax in zip(datasets, axes):
    with open(f"statistics/{dataset}/acc_list.pkl", "rb") as f:
        acc_list = pickle.load(f)
    with open(f"statistics/{dataset}/auc_tar_list.pkl", "rb") as f:
        auc_tar_list = pickle.load(f)
    with open(f"statistics/{dataset}/auc_adv_list.pkl", "rb") as f:
        auc_adv_list = pickle.load(f)
    with open(f"statistics/{dataset}/auc_gx_list.pkl", "rb") as f:
        auc_gx_list = pickle.load(f)
    with open(f"statistics/{dataset}/auc_top_list.pkl", "rb") as f:
        auc_top_list = pickle.load(f)
    with open(f"statistics/{dataset}/auc_appr_list.pkl", "rb") as f:
        auc_appr_list = pickle.load(f)

    acc_list.reverse()
    auc_tar_list.reverse()
    auc_adv_list.reverse()
    auc_gx_list.reverse()
    auc_top_list.reverse()
    auc_appr_list.reverse()

    with open(f"statistics/{dataset}/acc_list_dp.pkl", "rb") as f:
        acc_list_dp = pickle.load(f)
    with open(f"statistics/{dataset}/auc_tar_list_dp.pkl", "rb") as f:
        auc_tar_list_dp = pickle.load(f)
    with open(f"statistics/{dataset}/auc_adv_list_dp.pkl", "rb") as f:
        auc_adv_list_dp = pickle.load(f)
    with open(f"statistics/{dataset}/auc_gx_list_dp.pkl", "rb") as f:
        auc_gx_list_dp = pickle.load(f)
    with open(f"statistics/{dataset}/auc_top_list_dp.pkl", "rb") as f:
        auc_top_list_dp = pickle.load(f)
    with open(f"statistics/{dataset}/auc_appr_list_dp.pkl", "rb") as f:
        auc_appr_list_dp = pickle.load(f)

    acc_list_dp.reverse()
    auc_tar_list_dp.reverse()
    auc_adv_list_dp.reverse()
    auc_gx_list_dp.reverse()
    auc_top_list_dp.reverse()
    auc_appr_list_dp.reverse()

    gamma_adv_list = [compute_gamma(orig_auc, dp_auc, orig_acc, dp_acc) for orig_auc, dp_auc, orig_acc, dp_acc in zip(auc_adv_list, auc_adv_list_dp, acc_list, acc_list_dp)]
    gamma_top_list = [compute_gamma(orig_auc, dp_auc, orig_acc, dp_acc) for orig_auc, dp_auc, orig_acc, dp_acc in zip(auc_top_list, auc_top_list_dp, acc_list, acc_list_dp)]
    gamma_appr_list = [compute_gamma(orig_auc, dp_auc, orig_acc, dp_acc) for orig_auc, dp_auc, orig_acc, dp_acc in zip(auc_appr_list, auc_appr_list_dp, acc_list, acc_list_dp)]

    ax.set_title(dataset)
    ax.set_xticks(hr_range)
    # ax.set_yticks(acc_range)
    ax.set_xlabel("adversary feature ratio")
    ax.set_ylabel("gamma")

    ax.plot(hr_range, gamma_appr_list, label="v'_tar", color="blueviolet", marker='.')
    ax.plot(hr_range, gamma_top_list, label="v_top", color="red", marker='.')
    ax.plot(hr_range, gamma_adv_list, label="v_adv", color="limegreen", marker='.')

lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, bbox_to_anchor=(0.5, -0.08), ncol=5, loc="lower center", columnspacing=5.0)
plt.savefig(f"figs/gamma_hr.png", bbox_inches='tight')
