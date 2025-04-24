import wandb
from functions import *



api = wandb.Api()

run1_string = "5u0oy2dd"
run2_string = "cgygtgtv"
run3_string = "c70ewr80"
run4_string = "hw6uwr0a"
run5_string = "9orwioi4"
run6_string = "nrshhq3b"

# plotting 30mm 0 30mm pca

run1 = api.run("/michael-bezick-purdue-university/pprl/runs/" + run1_string)
run2 = api.run("/michael-bezick-purdue-university/pprl/runs/" + run2_string)
run3 = api.run("/michael-bezick-purdue-university/pprl/runs/" + run3_string)
run4 = api.run("/michael-bezick-purdue-university/pprl/runs/" + run4_string)
run5 = api.run("/michael-bezick-purdue-university/pprl/runs/" + run5_string)
run6 = api.run("/michael-bezick-purdue-university/pprl/runs/" + run6_string)

list_of_runs = [run1, run2, run3, run4, run5, run6]

plot_runs(list_of_runs=list_of_runs, key_x="_step", key_y="eval/DiscountedReturn", num_bootstrap_samples=1000, filename="small_mismatch_PCA_dis_ret.pdf")
plot_runs(list_of_runs=list_of_runs, key_x="_step", key_y="eval/Success", num_bootstrap_samples=1000, filename="small_mismatch_PCA_eval_success.pdf")
