import matplotlib.pyplot as plt
import numpy as np
import collections
import wandb

def get_runs_by_group(group_name, entity="michael-bezick-purdue-university", project="pprl"):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    return [run for run in runs if run.group == group_name]

def get_data(run, key1, key2):

    data1 = []
    data2 = []
    for i, row in run.history(keys=[key2]).iterrows():
        data1.append(row[key1])
        data2.append(row[key2])

    return data1, data2

def construct_bootstrapped_CI(data, num_samples):

    boot_means = []
    
    for _ in range(num_samples):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))

    lower = np.percentile(boot_means, 2.5)
    upper = np.percentile(boot_means, 97.5)

    return lower, upper

def plot_runs(list_of_runs, key_x, key_y, num_bootstrap_samples, filename):

    # key 2 will be y variable

    x_to_ys = collections.defaultdict(list)
    means, lowers, uppers = [], [], []


    for run in list_of_runs:
        data_x, data_y = get_data(run, key_x, key_y)
        for x, y in zip(data_x, data_y):
            x_to_ys[x].append(y)

    sorted_xs = sorted(x_to_ys.keys())

    for x in sorted_xs:
        y_values = x_to_ys[x]
        mean = np.mean(y_values)
        lower, upper = construct_bootstrapped_CI(y_values, num_bootstrap_samples)
        means.append(mean)
        lowers.append(lower)
        uppers.append(upper)

    # Convert to arrays for plotting
    xs = np.array(sorted_xs)
    means = np.array(means)
    lowers = np.array(lowers)
    uppers = np.array(uppers)


    # Plot with confidence interval
    plt.figure(figsize=(8, 5))
    plt.plot(xs, means, label="Mean " + key_y)
    plt.fill_between(xs, lowers, uppers, color='blue', alpha=0.3, label="95% CI")
    plt.xlabel(key_x)
    plt.ylabel(key_y)
    plt.title(f"{key_y} with 95% Bootstrap CI")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)

def plot_runs_comparison(list_of_runs1, list_of_runs2, key_x, key_y, num_bootstrap_samples, filename):

    # key 2 will be y variable

    x_to_ys = collections.defaultdict(list)
    means, lowers, uppers = [], [], []


    for run in list_of_runs1:
        data_x, data_y = get_data(run, key_x, key_y)
        for x, y in zip(data_x, data_y):
            x_to_ys[x].append(y)

    sorted_xs = sorted(x_to_ys.keys())

    for x in sorted_xs:
        y_values = x_to_ys[x]
        mean = np.mean(y_values)
        lower, upper = construct_bootstrapped_CI(y_values, num_bootstrap_samples)
        means.append(mean)
        lowers.append(lower)
        uppers.append(upper)

    # Convert to arrays for plotting
    xs = np.array(sorted_xs)
    means = np.array(means)
    lowers = np.array(lowers)
    uppers = np.array(uppers)


    # Plot with confidence interval
    plt.figure(figsize=(8, 5))
    plt.plot(xs, means, label="Mean " + key_y)
    plt.fill_between(xs, lowers, uppers, color='blue', alpha=0.3, label="95% CI")
    plt.xlabel(key_x)
    plt.ylabel(key_y)
    # plt.title(f"{key_y} with 95% Bootstrap CI")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    x_to_ys = collections.defaultdict(list)
    means, lowers, uppers = [], [], []


    for run in list_of_runs2:
        data_x, data_y = get_data(run, key_x, key_y)
        for x, y in zip(data_x, data_y):
            x_to_ys[x].append(y)

    sorted_xs = sorted(x_to_ys.keys())

    for x in sorted_xs:
        y_values = x_to_ys[x]
        mean = np.mean(y_values)
        lower, upper = construct_bootstrapped_CI(y_values, num_bootstrap_samples)
        means.append(mean)
        lowers.append(lower)
        uppers.append(upper)

    # Convert to arrays for plotting
    xs = np.array(sorted_xs)
    means = np.array(means)
    lowers = np.array(lowers)
    uppers = np.array(uppers)

    plt.figure(figsize=(8, 5))
    plt.plot(xs, means, label="Mean " + key_y)
    plt.fill_between(xs, lowers, uppers,  alpha=0.3, label="95% CI")
    plt.xlabel(key_x)
    plt.ylabel(key_y)
    plt.title(f"{key_y} with 95% Bootstrap CI")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(filename)
