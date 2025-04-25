import wandb
from functions import *

# group_names = ["30_0_30mm_no_aug", "30_0_30mm_pca"]
#
# plot_multiple(group_names, key_x="_step", key_y="eval/DiscountedReturn", group_labels=["No augmentation", "PCA Augmentation"], filename="30_0_30mm_comparison_dis_return.pdf")
# plot_multiple(group_names, key_x="_step", key_y="eval/Success", group_labels=["No augmentation", "PCA Augmentation"], filename="30_0_30mm_comparison_eval.pdf")

group_names = ["10_0_0mm_pca", "30_0_30mm_pca"]

plot_multiple(group_names, key_x="_step", key_y="rollout/Success", group_labels=["10_0_0mm", "30_0_30mm"], filename="mismatch_size_rollout_comparison.pdf")
