import wandb
from functions import *



api = wandb.Api()

run1_string = "l68cj5gy"
run2_string = "u10hzx4c"
run3_string = "93vx0ko4"
run4_string = "gtdrdpqj"
run5_string = "a78goea7"
run6_string = "qog0zrzf"

# plotting 30mm 0 30mm pca

run1 = api.run("/michael-bezick-purdue-university/pprl/runs/" + run1_string)
run2 = api.run("/michael-bezick-purdue-university/pprl/runs/" + run2_string)
run3 = api.run("/michael-bezick-purdue-university/pprl/runs/" + run3_string)
run4 = api.run("/michael-bezick-purdue-university/pprl/runs/" + run4_string)
run5 = api.run("/michael-bezick-purdue-university/pprl/runs/" + run5_string)
run6 = api.run("/michael-bezick-purdue-university/pprl/runs/" + run6_string)

list_of_runs = [run1, run2, run3, run4, run5, run6]

plot_runs(list_of_runs=list_of_runs, key_x="_step", key_y="eval/DiscountedReturn", num_bootstrap_samples=1000, filename="small_mismatch_no_aug_dis_ret.pdf")
plot_runs(list_of_runs=list_of_runs, key_x="_step", key_y="eval/Success", num_bootstrap_samples=1000, filename="small_mismatch_no_aug_eval_success.pdf")
