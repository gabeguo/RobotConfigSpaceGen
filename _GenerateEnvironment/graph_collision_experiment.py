import json
import matplotlib.pyplot as plt
from constants import *
import numpy as np
import os
import argparse
import re
import pandas as pd
import scipy.stats as stats

CLF_TO_MAX_MARKER = {DL: 'o', FASTRON: 'x'}
CLF_TO_MEAN_MARKER = {DL: '^', FASTRON: 'v'}
CLF_TO_MAX_COLOR = {DL: (0.1, 0.8, 0.1, 1.0), FASTRON: (0.8, 0.1, 0.1, 1.0)}
CLF_TO_MEAN_COLOR = {DL: (0.2, 0.7, 0.2, 0.5), FASTRON: (0.7, 0.2, 0.2, 0.5)}
COLLISION_DENSITY_KEY = 'collision_density'

# Thanks ChatGPT!
def load_json_files_pd(args):
    global DOF, NUM_TRAIN_SAMPLES
    DOF = None
    NUM_TRAIN_SAMPLES = None
    # Load all JSON files in the directory into a list of DataFrames
    dataframes = []
    for filename in os.listdir(args.data_directory):
        if filename.endswith(".json"):
            with open(os.path.join(args.data_directory, filename), 'r') as f:
                data = json.load(f)
                if get_seed_number(data['dataset_name']) not in args.seeds:
                    continue
                df = pd.json_normalize(data)
                # get DoF
                df[DOF_KEY] = 7 * df['dataset_name'].str.extract('(\d+)').astype(int)
                assert DOF is None or DOF == df[DOF_KEY].unique().tolist()[0]
                DOF = df[DOF_KEY].unique().tolist()[0]
                # get num train samples
                curr_num_train_samples = df['num_training_samples'].astype(int).unique().tolist()[0]
                assert NUM_TRAIN_SAMPLES is None or NUM_TRAIN_SAMPLES == curr_num_train_samples
                NUM_TRAIN_SAMPLES = curr_num_train_samples
                # get collision density
                df[COLLISION_DENSITY_KEY] = (df[TP_NAME] + df[FN_NAME]) / (df[TP_NAME] + df[TN_NAME] + df[FP_NAME] + df[FN_NAME])

                # transform data
                if args.invert_metric:
                    df[args.metric] = 1 - df[args.metric]
                if args.unit_rate_metric:
                    df[args.metric] /= df[TEST_SIZE]

                dataframes.append(df)
    #print(len(dataframes))
    # Concatenate all the DataFrames into a single DataFrame
    df = pd.concat(dataframes, ignore_index=True)

    #print(df[COLLISION_DENSITY_KEY].unique().tolist())
    # Group by the comparison variables and compute the mean and std of args.x_metric and args.y_metric
    return df

def plot_results(df, args):
    for model_name in [DL, FASTRON]:
        df_model = df[df['model_name'] == model_name]
        
        # Extract maxes, means, and standard deviations for x_metric and y_metric
        unique_x_values_list = df_model[COLLISION_DENSITY_KEY].unique().tolist()
        unique_x_values_list.sort()
        assert len(unique_x_values_list) == 36 # number of distinct collision densities

        y_maxes = list()
        y_medians = list()
        y_iqrs = list()
        for x_val in unique_x_values_list:
            all_rows_with_x_val = df_model[df_model[COLLISION_DENSITY_KEY] == x_val]
            
            maximum_metric_val = all_rows_with_x_val[args.metric].max()
            y_maxes.append(maximum_metric_val)

            median_metric_val = all_rows_with_x_val[args.metric].median()
            y_medians.append(median_metric_val)

            iqr_metric_val = stats.iqr(all_rows_with_x_val[args.metric].tolist())
            y_iqrs.append(iqr_metric_val)

        plt.plot(unique_x_values_list, y_maxes, 
                 color=CLF_TO_MAX_COLOR[model_name], marker=CLF_TO_MAX_MARKER[model_name], label=f'{model_name}: Best Hyperparameters')
        error_bars=plt.errorbar(unique_x_values_list, y_medians, y_iqrs, linestyle='--', elinewidth=1, capsize=1.5,
                     color=CLF_TO_MEAN_COLOR[model_name], marker=CLF_TO_MEAN_MARKER[model_name], label=f'{model_name}: Median Performance')
        error_bars[-1][0].set_linestyle('--')
    plt.ylim(plt.ylim()[0], 1.0)
    plt.xlabel('Collision Density')
    plt.ylabel(args.metric.capitalize())
    plt.legend()
    plt.grid()
    plt.title(f'Collision Density vs {args.metric.capitalize()}:\n{DOF} DoF, {NUM_TRAIN_SAMPLES} samples')
    plt.savefig(f'{args.save_location}/Collision Density vs {args.metric}_{DOF} DoF.pdf')
    plt.savefig(f'{args.save_location}/Collision Density vs {args.metric}_{DOF} DoF.png')
    #plt.show()

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
    plot_results(df_mean_std, args)

    return

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add the arguments
    parser.add_argument("--data_directory", type=str, default='obstacles_experiment_results')
    parser.add_argument("--metric", type=str, default='accuracy')
    parser.add_argument('--invert_metric', action='store_true', help='1 - metric instead of metric', default=False)
    parser.add_argument('--unit_rate_metric', action='store_true', help='Divide metric by number of samples', default=False)
    parser.add_argument("--seeds", nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument("--save_location", type=str, default='graphs')

    # Execute the parse_args() method
    args = parser.parse_args()

    os.makedirs(args.save_location, exist_ok=True)

    main(args)
