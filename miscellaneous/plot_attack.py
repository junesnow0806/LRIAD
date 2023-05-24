import sys
import pickle
import matplotlib.pyplot as plt

dataset_name = sys.argv[1]
hr_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
acc_range = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

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

acc_list.reverse()
auc_tar_list.reverse()
auc_adv_list.reverse()
auc_gx_list.reverse()
auc_top_list.reverse()
auc_appr_list.reverse()

with open(f"statistics/{dataset_name}/acc_list_dp.pkl", "rb") as f:
    acc_list_dp = pickle.load(f)
with open(f"statistics/{dataset_name}/auc_tar_list_dp.pkl", "rb") as f:
    auc_tar_list_dp = pickle.load(f)
with open(f"statistics/{dataset_name}/auc_adv_list_dp.pkl", "rb") as f:
    auc_adv_list_dp = pickle.load(f)
with open(f"statistics/{dataset_name}/auc_gx_list_dp.pkl", "rb") as f:
    auc_gx_list_dp = pickle.load(f)
with open(f"statistics/{dataset_name}/auc_top_list_dp.pkl", "rb") as f:
    auc_top_list_dp = pickle.load(f)
with open(f"statistics/{dataset_name}/auc_appr_list_dp.pkl", "rb") as f:
    auc_appr_list_dp = pickle.load(f)

acc_list_dp.reverse()
auc_tar_list_dp.reverse()
auc_adv_list_dp.reverse()
auc_gx_list_dp.reverse()
auc_top_list_dp.reverse()
auc_appr_list_dp.reverse()

plt.title(dataset_name)
plt.xticks(hr_range)
plt.yticks(acc_range)
plt.xlabel("adversary feature ratio")
plt.ylabel("Accuracy/AUC")
plt.bar(hr_range, acc_list, label="acc", align="center", width=0.08)
plt.plot(hr_range, auc_tar_list, label="v_tar", color="blue", marker='.')
plt.plot(hr_range, auc_appr_list, label="v'_tar", color="blueviolet", marker='.')
plt.plot(hr_range, auc_top_list, label="v_top", color="red", marker='.')
plt.plot(hr_range, auc_adv_list, label="v_adv", color="limegreen", marker='.')

plt.bar(hr_range, acc_list_dp, label="acc_dp", align="center", width=0.08, color="orange")
plt.plot(hr_range, auc_tar_list_dp, label="v_tar_dp", color="blue", marker='.', linestyle='dashed')
plt.plot(hr_range, auc_appr_list_dp, label="v'_tar_dp", color="blueviolet", marker='.', linestyle='dashed')
plt.plot(hr_range, auc_top_list_dp, label="v_top_dp", color="red", marker='.', linestyle='dashed')
plt.plot(hr_range, auc_adv_list_dp, label="v_adv_dp", color="limegreen", marker='.', linestyle='dashed')

plt.legend()
plt.savefig(f"{dataset_name}.png")
