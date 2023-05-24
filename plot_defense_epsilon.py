import pickle
import matplotlib.pyplot as plt


hr_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
acc_range = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

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

    acc_list = [acc_list[4]] * 9
    auc_tar_list = [auc_tar_list[4]] * 9
    auc_adv_list = [auc_adv_list[4]] * 9
    auc_gx_list = [auc_gx_list[4]] * 9
    auc_top_list = [auc_top_list[4]] * 9
    auc_appr_list = [auc_appr_list[4]] * 9

    with open(f"statistics/{dataset}/acc_list_epsilon.pkl", "rb") as f:
        acc_list_dp = pickle.load(f)
    with open(f"statistics/{dataset}/auc_tar_list_epsilon.pkl", "rb") as f:
        auc_tar_list_dp = pickle.load(f)
    with open(f"statistics/{dataset}/auc_adv_list_epsilon.pkl", "rb") as f:
        auc_adv_list_dp = pickle.load(f)
    with open(f"statistics/{dataset}/auc_gx_list_epsilon.pkl", "rb") as f:
        auc_gx_list_dp = pickle.load(f)
    with open(f"statistics/{dataset}/auc_top_list_epsilon.pkl", "rb") as f:
        auc_top_list_dp = pickle.load(f)
    with open(f"statistics/{dataset}/auc_appr_list_epsilon.pkl", "rb") as f:
        auc_appr_list_dp = pickle.load(f)

    acc_list_dp.reverse()
    auc_tar_list_dp.reverse()
    auc_adv_list_dp.reverse()
    auc_gx_list_dp.reverse()
    auc_top_list_dp.reverse()
    auc_appr_list_dp.reverse()

    ax.set_title(dataset)
    ax.set_xticks(hr_range)
    ax.set_yticks(acc_range)
    ax.set_xlabel("epsilon")
    ax.set_ylabel("Accuracy/AUC")

    ax.bar(hr_range, acc_list, label="acc", align="center", width=0.08)
    ax.plot(hr_range, auc_tar_list, label="v_tar", color="blue", marker='.')
    ax.plot(hr_range, auc_appr_list, label="v'_tar", color="blueviolet", marker='.')
    ax.plot(hr_range, auc_top_list, label="v_top", color="red", marker='.')
    ax.plot(hr_range, auc_adv_list, label="v_adv", color="limegreen", marker='.')

    ax.bar(hr_range, acc_list_dp, label="acc_dp", align="center", width=0.08, color="orange")
    ax.plot(hr_range, auc_tar_list_dp, label="v_tar_dp", color="blue", marker='.', linestyle='dashed')
    ax.plot(hr_range, auc_appr_list_dp, label="v'_tar_dp", color="blueviolet", marker='.', linestyle='dashed')
    ax.plot(hr_range, auc_top_list_dp, label="v_top_dp", color="red", marker='.', linestyle='dashed')
    ax.plot(hr_range, auc_adv_list_dp, label="v_adv_dp", color="limegreen", marker='.', linestyle='dashed')

lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, bbox_to_anchor=(0.5, -0.08), ncol=5, loc="lower center", columnspacing=5.0)
plt.savefig(f"figs/defense_epsilon.png", bbox_inches='tight')
