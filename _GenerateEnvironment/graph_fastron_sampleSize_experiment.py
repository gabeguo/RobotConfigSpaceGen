import matplotlib.pyplot as plt
import re
import os
import json
from constants import *
import numpy as np

POSSIBLE_BETAS = [1, 500]
POSSIBLE_NUM_SAMPLES = [1000, 10000]#, 100000, 500000, 900000]
MARKERS = ['o', 'x']

# Thanks ChatGPT!
def find_files(directory, pattern):
    regex = re.compile(pattern)
    for root, dirs, files in os.walk(directory):
        for file in files:
            if regex.match(file):
                yield os.path.join(root, file)
"""
For the given beta and num_samples, gets the mean and std of metric over all seeds
"""
def get_data_for_metric(all_data, metric, beta, num_samples):
    curr_experiment = all_data[(beta, num_samples)]
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
        for num_samples in POSSIBLE_NUM_SAMPLES:
            curr_mean, curr_std = get_data_for_metric(all_data, metric, beta, num_samples)
            metric_means.append(curr_mean)
            metric_stds.append(curr_std)
        plt.errorbar(POSSIBLE_NUM_SAMPLES, metric_means, metric_stds, label=f'$\\beta$ = {beta}', marker=MARKERS[i], capsize=2)
        for j in range(len(POSSIBLE_NUM_SAMPLES)):
            plt.text(POSSIBLE_NUM_SAMPLES[j] + 2, metric_means[j], f'{metric_means[j]:.3f}', ha='right')
    plt.xlabel('Number of Samples')
    plt.ylabel(metric.upper())
    plt.xticks(POSSIBLE_NUM_SAMPLES)
    plt.title(f'Fastron Performance: {metric.upper()} as a function of Number of Samples')
    plt.grid()
    plt.legend()
    #plt.show()
    os.makedirs(GRAPH_FOLDER_NAME, exist_ok=True)
    plt.savefig(fname=f'{GRAPH_FOLDER_NAME}/{metric}_by_numSamples.pdf')
    plt.savefig(fname=f'{GRAPH_FOLDER_NAME}/{metric}_by_numSamples.png')
    plt.clf()

    return

def graph_results(all_data):
    for metric in [TPR, TNR, ACCURACY]:
        graph_results_by_metric(all_data, metric)
    return

def collect_data():
    all_data = dict() # all_data[beta][num_samples] = list of JSON dicts that tried that beta and num_samples (only diff is seed)
    for beta in POSSIBLE_BETAS:
        for num_samples in POSSIBLE_NUM_SAMPLES:
            pattern_str = f"fastronResults_forwardKinematics_{num_samples}Samples_{beta}Beta_5Gamma_3robots_25obstacles_seed(.*?)_.json"
            # Use the pattern string in the regex search
            pattern = re.compile(r"" + pattern_str)
            curr_data = list()
            for file in find_files('approximation_results', pattern):
                with open(file, 'r') as fin:
                    curr_data.append(json.load(fin))
            all_data[(beta, num_samples)] = curr_data
    
    return all_data

def main():
    all_data = collect_data()
    graph_results(all_data)
    return

if __name__ == "__main__":
    main()

