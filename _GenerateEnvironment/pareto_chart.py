import json
import matplotlib.pyplot as plt
from constants import *
import numpy as np
import os
import argparse
import re
import pandas as pd

# Iff these are the same between two runs, we say they use the same model
COMPARISON_VARIABLES = {
    'train_size',
    'test_size',
    'model_name',
    'num_training_samples',
    'num_testing_samples',
    'dataset_name',
    'forward_kinematics_kernel',
    'g',
    'beta',
    'maxUpdates',
    'maxSupportPoints',
    'bias',
    'num_freq',
    'sigma',
    'lr',
    'batch_size',
    'train_percent',
    'epochs'
}
CLF_TO_MARKER = {DL: 'o', FASTRON: 'x'}
CLF_TO_COLOR = {DL: '#228822', FASTRON: '#882222'}

# Thanks ChatGPT!
def load_json_files(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), 'r') as f:
                file_data = json.load(f)
                data.append(file_data)
    return data

# Thanks ChatGPT!
def load_json_files_pd(args):
    # Load all JSON files in the directory into a list of DataFrames
    dataframes = []
    for filename in os.listdir(args.data_directory):
        if filename.endswith(".json"):
            with open(os.path.join(args.data_directory, filename), 'r') as f:
                data = json.load(f)
                dataframes.append(df)

    # Concatenate all the DataFrames into a single DataFrame
    df = pd.concat(dataframes, ignore_index=True)

    # Group by the comparison variables and compute the mean and std of args.x_metric and args.y_metric
    df_mean_std = df.groupby(list(COMPARISON_VARIABLES)).agg(
        {
            args.x_metric: ['mean', 'std'],
            args.y_metric: ['mean', 'std']
        }
    ).reset_index()

    print(df_mean_std)

# Thanks ChatGPT!
def get_seed_number(string):
    match = re.search('seed(\d+)', string)
    if match:
        return int(match.group(1))
    else:
        return None

def load_and_plot_points(the_data, x_y_pairs, args):
    for curr_run in the_data:
        if get_seed_number(curr_run['dataset_name']) not in args.seeds:
            continue
        x = curr_run[args.x_metric]
        y = curr_run[args.y_metric]
        # process x
        if args.invert_x:
            x = 1 - x
        if args.unit_rate_x:
            x /= curr_run[TEST_SIZE]
        # process y
        if args.invert_y:
            y = 1 - y
        if args.unit_rate_y:
            y /= curr_run[TEST_SIZE]
        clf_name = curr_run['model_name']

        plt.scatter(x=[x], \
            y=[y], \
            label=clf_name, s=75, marker=CLF_TO_MARKER[clf_name], c=CLF_TO_COLOR[clf_name], zorder=2)
        x_y_pairs.append((x, y))
    return

def plot_pareto(args):
    plt.rcParams.update({'figure.figsize': (8, 6)})
    plt.rcParams.update({'font.size': 11})

    the_data = load_json_files(args.data_directory)

    # storing all the data points for use in pareto front
    x_y_pairs = list()

    # get all the data points
    load_and_plot_points(the_data, x_y_pairs, args)   
    load_json_files_pd(args) 

    # draw pareto front
    x_y_pairs.sort()
    pareto_front = list()
    for x, y in x_y_pairs:
        viable = True
        for x_, y_ in pareto_front:
            if x >= x_ and y >= y_: # overshadowed
                viable = False
                break
        if viable:
            if len(pareto_front) > 0:
                pareto_front.append((x, pareto_front[-1][1]))
            pareto_front.append((x, y))

    plt.plot([pair[0] for pair in pareto_front], [pair[1] for pair in pareto_front], \
        linestyle='-', color='k', alpha=0.7, zorder=1)
    
    # calculate collision percentage


    # labels & legends
    x_label = args.x_label if args.x_label else args.x_metric
    y_label = args.y_label if args.y_label else args.y_metric
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'{y_label} as a function of {x_label}: ')

    # Create a custom legend with unique labels - Thanks ChatGPT!
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.show()

    return


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add the arguments
    parser.add_argument("--data_directory", type=str)
    parser.add_argument("--x_metric", type=str)
    parser.add_argument("--y_metric", type=str)
    parser.add_argument('--invert_x', action='store_true', help='1 - x instead of x', default=False)
    parser.add_argument('--invert_y', action='store_true', help='1 - y instead of y', default=False)
    parser.add_argument('--unit_rate_x', action='store_true', help='Divide x by number of samples', default=False)
    parser.add_argument('--unit_rate_y', action='store_true', help='Divide y by number of samples', default=False)
    parser.add_argument('--x_label', type=str, default=None)
    parser.add_argument('--y_label', type=str, default=None)
    parser.add_argument("--seeds", nargs='+', type=int, default=[0])
    parser.add_argument("--save_location", type=str, default='pareto_charts')

    # Execute the parse_args() method
    args = parser.parse_args()

    plot_pareto(args)
