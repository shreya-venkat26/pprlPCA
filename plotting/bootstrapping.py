import wandb
from functions import *

group_name = "30_0_0mm_no_aug"

list_of_runs = get_runs_by_group(group_name)

plot_runs(list_of_runs=list_of_runs, key_x="_step", key_y="eval/DiscountedReturn", num_bootstrap_samples=1000, filename= "plots/" + group_name + "_dis_ret.pdf")
plot_runs(list_of_runs=list_of_runs, key_x="_step", key_y="eval/Success", num_bootstrap_samples=1000, filename= "plots/" + group_name + "_eval_success.pdf")
