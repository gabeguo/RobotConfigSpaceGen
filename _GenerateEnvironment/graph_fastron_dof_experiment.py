import matplotlib.pyplot as plt
import re
import os
import json
from constants import *
import numpy as np

POSSIBLE_BETAS = [1, 500]
POSSIBLE_NUM_ROBOTS = [1, 2, 3, 4]
MARKERS = ['o', 'x']

# Thanks ChatGPT!
def find_files(directory, pattern):
    regex = re.compile(pattern)
    for root, dirs, files in os.walk(directory):
        for file in files:
            if regex.match(file):
                yield os.path.join(root, file)
"""
For the given beta and num_robots, gets the mean and std of metric over all seeds
"""
def get_data_for_metric(all_data, metric, beta, num_robots):
    curr_experiment = all_data[(beta, num_robots)]
    metric_vals = list()
    for json_dict in curr_experiment:
        metric_vals.append(json_dict[metric])
    avg_metric_val = np.mean(metric_vals)
    sem_metric_val = np.std(metric_vals, ddof=1)
    return avg_metric_val, sem_metric_val

"""
Graphs results for given metric
"""
def graph_results_by_metric(all_data, metric):
    # one line per beta
    for i in range(len(POSSIBLE_BETAS)):
        beta = POSSIBLE_BETAS[i]
        metric_means = list()
        metric_stds = list()
        for num_robots in POSSIBLE_NUM_ROBOTS:
            curr_mean, curr_std = get_data_for_metric(all_data, metric, beta, num_robots)
            metric_means.append(curr_mean)
            metric_stds.append(curr_std)
        DOFs = 7 * np.array(POSSIBLE_NUM_ROBOTS)
        plt.errorbar(DOFs, metric_means, metric_stds, label=f'$\\beta$ = {beta}', marker=MARKERS[i], capsize=2)
    plt.xlabel('DoF')
    plt.ylabel(metric.upper())
    plt.xticks(DOFs)
    plt.title(f'Fastron Performance: {metric.upper()} as a function of DoF')
    plt.grid()
    plt.legend()
    plt.show()
    os.makedirs(GRAPH_FOLDER_NAME, exist_ok=True)
    plt.savefig(title=f'{GRAPH_FOLDER_NAME}/{metric}_by_dof.pdf')
    plt.clf()

    return

def graph_results(all_data):
    for metric in [TPR, TNR, ACCURACY]:
        graph_results_by_metric(all_data, metric)
    return

def collect_data():
    all_data = dict() # all_data[beta][num_robots] = list of JSON dicts that tried that beta and num_robots (only diff is seed)
    for beta in POSSIBLE_BETAS:
        for num_robots in POSSIBLE_NUM_ROBOTS:
            pattern_str = f"fastronResults_forwardKinematics_25000Samples_{beta}Beta_5Gamma_{num_robots}robots_25obstacles_seed(.*?)_.json"
            # Use the pattern string in the regex search
            pattern = re.compile(r"" + pattern_str)
            curr_data = list()
            for file in find_files('approximation_results', pattern):
                with open(file, 'r') as fin:
                    curr_data.append(json.load(fin))
            all_data[(beta, num_robots)] = curr_data
    
    return all_data

def main():
    all_data = collect_data()
    graph_results(all_data)
    return

if __name__ == "__main__":
    main()

