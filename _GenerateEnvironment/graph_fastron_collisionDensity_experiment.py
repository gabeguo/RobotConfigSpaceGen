import matplotlib.pyplot as plt
import re
import os
import json
from constants import *
import numpy as np

from graph_fastron_dof_experiment import find_files

POSSIBLE_KERNELS = ['normalizedAngles', 'forwardKinematics']
POSSIBLE_BETAS = [1, 500]
POSSIBLE_NUM_OBSTACLES = [10, 20, 30, 40, 50, 60]
POSSIBLE_ROBOT_INTERACTIONS = ['colliding', 'separate']

MARKERS = ['o', '*', 's', 'P']#['o', 'v', '^', '<', '>', 's', '*', 'P', 'p']

def collect_data():
    all_data = dict() # all_data[kernel][beta][robot_interaction] = list of JSON dicts that tried specific kernel, beta, robot_interaction (only diff is seed & num_obstacles)
    for kernel in POSSIBLE_KERNELS:
        for beta in POSSIBLE_BETAS:
            #for robot_interaction in POSSIBLE_ROBOT_INTERACTIONS:
            pattern_str = f'fastronResults_{kernel}_25000Samples_{beta}Beta_5Gamma_3robots_(.*?)obstacles_seed(.*?)_(.*?)Robots.json'
            # Use the pattern string in the regex search
            pattern = re.compile(r"" + pattern_str)
            curr_data = list()
            for file in find_files('approximation_results', pattern):
                with open(file, 'r') as fin:
                    curr_data.append(json.load(fin))
            all_data[(kernel, beta)] = curr_data
    
    return all_data

"""
For given list of json_dicts that correspondings to a setting (kernel, beta, robot_interaction), get:
-> list of collision densities
-> their corresponding metric values
"""
def get_data_for_metric(data_at_setting, metric):
    all_pairs = list()
    for json_dict in data_at_setting:
        num_collisions = json_dict[TP_NAME] + json_dict[FN_NAME]
        num_free = json_dict[TN_NAME] + json_dict[FP_NAME]
        all_pairs.append((num_collisions/(num_collisions+num_free), json_dict[metric]))
    all_pairs.sort(key=lambda tup: tup[0])
    collision_densities = [x[0] for x in all_pairs]
    metric_vals = [x[1] for x in all_pairs]
    return collision_densities, metric_vals

"""
Graphs results for given metric
"""
def graph_results_by_metric(all_data, metric):
    # one line per (kernel, beta, robot_interaction)
    for setting, curr_marker in zip(all_data.keys(), MARKERS):
        kernel, beta = setting
        json_list = all_data[setting] # list of json dicts
        collision_densities, metric_vals = get_data_for_metric(json_list, metric)
        plt.plot(collision_densities, metric_vals, label=f'{kernel}, $\\beta$ = {beta}', marker=curr_marker)

    plt.xlabel('Collision Percentage')
    plt.ylabel(metric.upper())
    plt.title(f'Fastron Performance: {metric.upper()} as a function of Collision Percentage')
    plt.grid()
    plt.legend()
    #plt.show()
    os.makedirs(GRAPH_FOLDER_NAME, exist_ok=True)
    plt.savefig(fname=f'{GRAPH_FOLDER_NAME}/{metric}_by_collisionPercentage.pdf')
    plt.savefig(fname=f'{GRAPH_FOLDER_NAME}/{metric}_by_collisionPercentage.png')
    plt.clf()

    return

def graph_results(all_data):
    for metric in [TPR, TNR, ACCURACY]:
        graph_results_by_metric(all_data, metric)
    return

def main():
    all_data = collect_data()
    graph_results(all_data)
    return

if __name__ == "__main__":
    main()