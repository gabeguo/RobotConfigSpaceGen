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
FULL_MODEL_NAME = {DL: 'DeepCollide', FASTRON: 'Fastron FK'}

# for each model being evaluated at certain DoF, 
# results are averaged over all seeds/environments with that DoF
COMPARISON_VARIABLES = {
    'num_training_samples', 
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

# Thanks ChatGPT!
def load_json_files_pd(args):
    global DOF, COLLISION_DENSITY, NUM_TEST_SAMPLES
    DOF, COLLISION_DENSITY, NUM_TEST_SAMPLES = None, None, None
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
                # get collision density
                df[COLLISION_DENSITY_KEY] = (df[TP_NAME] + df[FN_NAME]) / (df[TP_NAME] + df[TN_NAME] + df[FP_NAME] + df[FN_NAME])
                # get num test samples
                curr_num_test_samples = df['num_testing_samples'].astype(int).unique().tolist()[0]
                assert NUM_TEST_SAMPLES is None or NUM_TEST_SAMPLES == curr_num_test_samples
                NUM_TEST_SAMPLES = curr_num_test_samples
                # transform data
                if args.unit_rate_metric:
                    df[args.metric] /= df[TEST_SIZE]

                dataframes.append(df)
    #print(len(dataframes))
    # Concatenate all the DataFrames into a single DataFrame
    df = pd.concat(dataframes, ignore_index=True)

    # Take mean over all environments at certain DoF for each model (where distinct hyperparams mean distinct model)
    # Group by the comparison variables and compute the mean and std of args.metric
    df_mean_std = df.groupby(list(COMPARISON_VARIABLES)).agg(
        {
            args.metric: ['mean', 'std']
        }
    ).reset_index()

    # Group by the comparison variables and compute the mean and std of args.metric
    return df_mean_std

def plot_results(df, args):
    y_values = df[(args.metric, 'mean')].tolist()
    all_y_medians = list()
    all_y_iqrs = list()
    for model_name in [DL, FASTRON]:
        df_model = df[df['model_name'] == model_name]
        
        # Extract maxes, means, and standard deviations for x_metric and y_metric
        unique_x_values_list = df_model['num_training_samples'].unique().tolist()
        unique_x_values_list.sort()
        print(unique_x_values_list)
        assert len(unique_x_values_list) == 9 # number of distinct training set sizes

        y_best = list()
        y_medians = list()
        y_iqrs = list()
        for x_val in unique_x_values_list:
            all_rows_with_x_val = df_model[df_model['num_training_samples'] == x_val]
            
            best_metric_val = all_rows_with_x_val[(args.metric, 'mean')].min() \
                if args.invert_metric else all_rows_with_x_val[(args.metric, 'mean')].max()
            y_best.append(best_metric_val)

            median_metric_val = all_rows_with_x_val[(args.metric, 'mean')].median()
            y_medians.append(median_metric_val)

            iqr_metric_val = stats.iqr(all_rows_with_x_val[(args.metric, 'mean')].tolist())
            y_iqrs.append(iqr_metric_val)

        plt.plot(unique_x_values_list, y_best, 
                 color=CLF_TO_MAX_COLOR[model_name], marker=CLF_TO_MAX_MARKER[model_name], label=f'{FULL_MODEL_NAME[model_name]}: Best Hyperparameters')
        error_bars=plt.errorbar(unique_x_values_list, y_medians, y_iqrs, linestyle='--', elinewidth=1, capsize=1.5,
                     color=CLF_TO_MEAN_COLOR[model_name], marker=CLF_TO_MEAN_MARKER[model_name], label=f'{FULL_MODEL_NAME[model_name]}: Median Performance')
        error_bars[-1][0].set_linestyle('--')

        all_y_medians.extend(y_medians)
        all_y_iqrs.extend(y_iqrs)
    ymin = max(min(y_values),
               min([curr_y_val - curr_y_err \
                    for curr_y_val, curr_y_err \
                        in zip(all_y_medians, all_y_iqrs)]))
    ymax = min(max(y_values), 
               max([curr_y_val + curr_y_err \
                    for curr_y_val, curr_y_err \
                        in zip(all_y_medians, all_y_iqrs)]))
    yspan = ymax - ymin
    print(ymin, ymax)
    plt.ylim(ymin - yspan * 0.05 , ymax + yspan * 0.05)
    plt.xlabel('Number of Train Samples')
    #plt.xticks(unique_x_values_list)
    metric_name = args.metric.capitalize() if len(args.metric) >= 5 else args.metric.upper()
    if args.ylabel:
        plt.ylabel(args.ylabel)
    else:
        plt.ylabel(metric_name)
    plt.legend()
    plt.grid()
    plt.title(f'Number Train Samples vs {metric_name}:\n{DOF} DoF, {NUM_TEST_SAMPLES} Test Samples')
    plt.savefig(f'{args.save_location}/Train Samples vs {metric_name}.pdf')
    plt.savefig(f'{args.save_location}/Train Samples vs {metric_name}.png')
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
    parser.add_argument("--data_directory", type=str, default='trainSize_experiment_results')
    parser.add_argument("--metric", type=str, default='accuracy')
    parser.add_argument('--invert_metric', action='store_true', help='min metric is best value', default=False)
    parser.add_argument('--unit_rate_metric', action='store_true', help='Divide metric by number of samples', default=False)
    parser.add_argument("--ylabel", type=str, default=None)
    parser.add_argument("--seeds", nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument("--save_location", type=str, default='graphs')

    # Execute the parse_args() method
    args = parser.parse_args()

    os.makedirs(args.save_location, exist_ok=True)

    main(args)
