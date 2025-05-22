from functions2 import *

# ‣ “one group, many metrics”  (train vs eval case)
plot_curves(
    group_names = ["pca_test_ortho_mismatch_forreal", "10_0_0mm_pca", "30_0_30mm_pca"],
    y_keys      = ["rollout/Success"],
    y_labels    = ["Train"],
    key_x       = "_step",
    filename    = "pca_test_train_vs_eval_success.pdf"
)
