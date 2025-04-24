import wandb
from functions import *



api = wandb.Api()

# plotting 30mm 0 30mm pca

run1 = api.run("/michael-bezick-purdue-university/pprl/runs/5319l3di")
run2 = api.run("/michael-bezick-purdue-university/pprl/runs/v5qwdkgi")
run3 = api.run("/michael-bezick-purdue-university/pprl/runs/h786zn4l")
run4 = api.run("/michael-bezick-purdue-university/pprl/runs/1i5mim3q")
run5 = api.run("/michael-bezick-purdue-university/pprl/runs/oe84cmx6")
run6 = api.run("/michael-bezick-purdue-university/pprl/runs/4yk8r4dx")

list_of_runs = [run1, run2, run3, run4, run5, run6]

plot_runs(list_of_runs=list_of_runs, key_x="_step", key_y="eval/DiscountedReturn", num_bootstrap_samples=1000, filename="large_mismatch_PCA.pdf")
