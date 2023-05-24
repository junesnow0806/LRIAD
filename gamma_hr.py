import sys
import pickle

from dp.utils import compute_gamma


dataset_name = sys.argv[1]
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

# gamma changes with feature ratio
# print(auc_adv_list)
# print(auc_adv_list_dp)
# print(acc_list)
# print(acc_list_dp)
gamma_adv_list = [compute_gamma(orig_auc, dp_auc, orig_acc, dp_acc) for orig_auc, dp_auc, orig_acc, dp_acc in zip(auc_adv_list, auc_adv_list_dp, acc_list, acc_list_dp)]
gamma_top_list = [compute_gamma(orig_auc, dp_auc, orig_acc, dp_acc) for orig_auc, dp_auc, orig_acc, dp_acc in zip(auc_top_list, auc_top_list_dp, acc_list, acc_list_dp)]
gamma_appr_list = [compute_gamma(orig_auc, dp_auc, orig_acc, dp_acc) for orig_auc, dp_auc, orig_acc, dp_acc in zip(auc_appr_list, auc_appr_list_dp, acc_list, acc_list_dp)]

# print(gamma_adv_list)
# print(gamma_top_list)
# print(gamma_appr_list)

with open(f"statistics/{dataset_name}/gamma_adv_list.pkl", "wb") as f:
    pickle.dump(gamma_adv_list, f)
with open(f"statistics/{dataset_name}/gamma_top_list.pkl", "wb") as f:
    pickle.dump(gamma_top_list, f)
with open(f"statistics/{dataset_name}/gamma_appr_list.pkl", "wb") as f:
    pickle.dump(gamma_appr_list, f)
