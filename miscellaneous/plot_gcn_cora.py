import matplotlib.pyplot as plt
import pickle


hr_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
dp = False
if dp:
    with open("statistics/cora/acc_list_dp.pkl", "rb") as f:
        acc_list = pickle.load(f)
    with open("statistics/cora/auc_tar_list_dp.pkl", "rb") as f:
        auc_tar_list = pickle.load(f)
    with open("statistics/cora/auc_adv_list_dp.pkl", "rb") as f:
        auc_adv_list = pickle.load(f)
    with open("statistics/cora/auc_gx_list_dp.pkl", "rb") as f:
        auc_gx_list = pickle.load(f)
    with open("statistics/cora/auc_top_list_dp.pkl", "rb") as f:
        auc_top_list = pickle.load(f)
    with open("statistics/cora/auc_appr_list_dp.pkl", "rb") as f:
        auc_appr_list = pickle.load(f)
else:
    with open("statistics/cora/acc_list.pkl", "rb") as f:
        acc_list = pickle.load(f)
    with open("statistics/cora/auc_tar_list.pkl", "rb") as f:
        auc_tar_list = pickle.load(f)
    with open("statistics/cora/auc_adv_list.pkl", "rb") as f:
        auc_adv_list = pickle.load(f)
    with open("statistics/cora/auc_gx_list.pkl", "rb") as f:
        auc_gx_list = pickle.load(f)
    with open("statistics/cora/auc_top_list.pkl", "rb") as f:
        auc_top_list = pickle.load(f)
    with open("statistics/cora/auc_appr_list.pkl", "rb") as f:
        auc_appr_list = pickle.load(f)

acc_list.reverse()
auc_tar_list.reverse()
auc_adv_list.reverse()
auc_gx_list.reverse()
auc_top_list.reverse()
auc_appr_list.reverse()

if dp:
    plt.title("Cora with DP")
else:
    plt.title("Cora")
plt.xlim(0, 1)
plt.xticks(hr_range)
plt.xlabel("adversary feature ratio")
plt.ylabel("Accuracy/AUC")
plt.bar(hr_range, acc_list, label="acc", align="center", width=0.08)
plt.plot(hr_range, auc_tar_list, label="v_tar", color="blue", marker='.')
plt.plot(hr_range, auc_appr_list, label="v'_tar", color="blueviolet", marker='.')
plt.plot(hr_range, auc_top_list, label="v_top", color="red", marker='.')
plt.plot(hr_range, auc_adv_list, label="v_adv", color="limegreen", marker='.')
# plt.plot(hr_range, auc_gx_list, label="gx", color="orange", marker='.')
plt.legend()
if dp:
    plt.savefig("results_cora_dp.png")
else:
    plt.savefig("results_cora.png")
print("finished")
