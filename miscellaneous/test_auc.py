import torch
from sklearn.metrics import roc_auc_score

# y_true = torch.randint(low=0, high=2, size=[2708, 2708])
y_true = torch.ones([2708, 2708])
y_score = torch.ones([2708, 2708]) * 0.5
# y_score[0, 0] = 1 - y_score[0, 0]
print(roc_auc_score(y_true, y_score))
