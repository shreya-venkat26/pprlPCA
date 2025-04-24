import wandb
from functions import *



api = wandb.Api()

# plotting 30mm 0 30mm pca

run1 = api.run("/michael-bezick-purdue-university/pprl/runs/1vsyf739")
run2 = api.run("/michael-bezick-purdue-university/pprl/runs/5ovv122x")
run3 = api.run("/michael-bezick-purdue-university/pprl/runs/dqaqbpqj")
run4 = api.run("/michael-bezick-purdue-university/pprl/runs/jo7lmavq")
run5 = api.run("/michael-bezick-purdue-university/pprl/runs/fj4za6pf")
run6 = api.run("/michael-bezick-purdue-university/pprl/runs/knq0wkgz")

list_of_runs = [run1, run2, run3, run4, run5, run6]

plot_runs(list_of_runs=list_of_runs, key_x="_step", key_y="eval/DiscountedReturn", num_bootstrap_samples=1000, filename="large_mismatch_no_aug.pdf")
