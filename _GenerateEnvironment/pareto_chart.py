import json
import matplotlib.pyplot as plt
from constants import *
import numpy as np
import os
import argparse
import re
import pandas as pd

import matplotlib
matplotlib.use('Agg')

# Iff these are the same between two runs, we say they use the same model
COMPARISON_VARIABLES = {
    'model_name',
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
FULL_MODEL_NAME = {DL: 'DeepCollide', FASTRON: 'Fastron FK'}

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
# Function to calculate the Pareto frontier
def calculate_pareto_frontier(x, y):
    # Sort the points in ascending order of x
    sorted_points = sorted(zip(x, y))
    pareto_frontier = [sorted_points[0]]
    
    # For each point, if it is better than the last point on the frontier, add it to the frontier
    for point in sorted_points[1:]:
        if point[1] < pareto_frontier[-1][1]:
            pareto_frontier.append(point)
    
    # Return the frontier as separate lists of x and y coordinates
    return zip(*pareto_frontier)


# Thanks ChatGPT!
def load_json_files_pd(args):
    # Load all JSON files in the directory into a list of DataFrames
    dataframes = []
    for filename in os.listdir(args.data_directory):
        if filename.endswith(".json"):
            with open(os.path.join(args.data_directory, filename), 'r') as f:
                data = json.load(f)
                if get_seed_number(data['dataset_name']) not in args.seeds:
                    continue
                df = pd.json_normalize(data)
                # transform data
                if args.invert_x:
                    df[args.x_metric] = 1 - df[args.x_metric]
                if args.unit_rate_x:
                    df[args.x_metric] /= df[TEST_SIZE]
                if args.invert_y:
                    df[args.y_metric] = 1 - df[args.y_metric]
                if args.unit_rate_y:
                    df[args.y_metric] /= df[TEST_SIZE]

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

    title = args.title.replace('\\n', '_')
    df_mean_std.to_csv(os.path.join(args.save_location, f'Full Data_{title}.csv'), index=False)
    # print(df_mean_std[(df_mean_std['model_name'] == DL)][['model_name', 'bias', 'num_freq']])
    # print(df_mean_std[(df_mean_std['model_name'] == FASTRON)][['g', 'beta', 'maxUpdates', 'maxSupportPoints']])

    return df_mean_std

def plot_pareto(df_mean_std, args):
    # Create a scatter plot with a different color for each 'model_name'
    for model_name in [DL, FASTRON]:
        df_model = df_mean_std[df_mean_std['model_name'] == model_name]
        
        # Extract means and standard deviations for x_metric and y_metric
        x_means = df_model[(args.x_metric, 'mean')]
        x_stds = df_model[(args.x_metric, 'std')]
        y_means = df_model[(args.y_metric, 'mean')]
        y_stds = df_model[(args.y_metric, 'std')]

        # Create a scatter plot of the means of x_metric vs y_metric for this model_name
        plt.scatter(x_means, y_means, color=CLF_TO_COLOR[model_name], marker=CLF_TO_MARKER[model_name], s=35, label=FULL_MODEL_NAME[model_name])

        # Use errorbars to show standard deviation
        plt.errorbar(x_means, y_means, xerr=x_stds, yerr=y_stds, linestyle='None', color=CLF_TO_COLOR[model_name], alpha=0.1)

    # Add labels
    x_label = args.x_label if args.x_label else args.x_metric
    y_label = args.y_label if args.y_label else args.y_metric
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    pareto_x, pareto_y = calculate_pareto_frontier(df_mean_std[(args.x_metric, 'mean')], 
                                                   df_mean_std[(args.y_metric, 'mean')])
    pareto_x = list(pareto_x)
    pareto_y = list(pareto_y)

    # set axis limits
    ymin, ymax = plt.ylim()
    if ymax <= 1:
        plt.ylim(ymin, min(ymax, 1))
    else:
        plt.ylim(ymin, ymax)
    xmin, xmax = plt.xlim()
    plt.xlim(max(xmin, -0.01), xmax)

    # extend the line
    pareto_x.insert(0, pareto_x[0])
    pareto_y.insert(0, ymax)#pareto_y.insert(0, max(df_mean_std[(args.y_metric, 'mean')] + df_mean_std[(args.y_metric, 'std')].fillna(0, inplace=False)))
    pareto_x.append(xmax)#pareto_x.append(max(df_mean_std[(args.x_metric, 'mean')] + df_mean_std[(args.x_metric, 'std')].fillna(0, inplace=False)))
    pareto_y.append(pareto_y[-1])

    # Draw the Pareto frontier as a step plot
    plt.step(pareto_x, pareto_y, color='black', where='post')

    # # Fill the area under and to the left of the Pareto frontier
    # plt.fill_between(pareto_x, pareto_y, color='black', alpha=0.1, step='post')

    plt.grid()

    # Create a custom legend with unique labels - Thanks ChatGPT!
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    title = args.title.replace('\\n', '\n')
    plt.title(title)

    title = title.replace('\n', '_')
    plt.savefig(os.path.join(args.save_location, title + '.pdf'))
    plt.savefig(os.path.join(args.save_location, title + '.png'))
    #plt.show()

    # Print Pareto optimal settings
    pareto_indices = []
    for x, y in zip(pareto_x, pareto_y):
        match_indices = df_mean_std[(df_mean_std[(args.x_metric, 'mean')] == x) & (df_mean_std[(args.y_metric, 'mean')] == y)].index.tolist()
        pareto_indices.extend(match_indices)
    pareto_df = df_mean_std.loc[pareto_indices]
    pd.set_option('display.max_rows', 30)
    pd.set_option('display.max_columns', 30)
    print('Pareto optimal models:\n', pareto_df)

    pareto_df.to_csv(os.path.join(args.save_location, f'{title}.csv'), index=False)

    return

# Thanks ChatGPT!
def get_seed_number(string):
    match = re.search('seed(\d+)', string)
    if match:
        return int(match.group(1))
    else:
        return None

def main(args):
    plt.rcParams.update({'figure.figsize': (8, 6)})
    plt.rcParams.update({'font.size': 11})

    # get all the data points
    df_mean_std = load_json_files_pd(args) 
    # plot pareto frontier
    plot_pareto(df_mean_std, args)

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
    parser.add_argument('--title', type=str, default='meh')
    parser.add_argument("--seeds", nargs='+', type=int, default=[0])
    parser.add_argument("--save_location", type=str, default='pareto_charts')

    # Execute the parse_args() method
    args = parser.parse_args()

    os.makedirs(args.save_location, exist_ok=True)

    main(args)
