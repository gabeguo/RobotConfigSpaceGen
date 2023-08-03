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

# for each model being evaluated at certain DoF, 
# results are averaged over all seeds/environments with that DoF
COMPARISON_VARIABLES = {
    DOF_KEY, 
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

FULL_MODEL_NAME = {DL: 'DeepCollide', FASTRON: 'Fastron FK'}

# Thanks ChatGPT!
def load_json_files_pd(args):
    global DOF, NUM_TRAIN_SAMPLES, NUM_TEST_SAMPLES
    DOF = None
    NUM_TRAIN_SAMPLES = None
    NUM_TEST_SAMPLES = None
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
                # get num train samples
                curr_num_train_samples = df['num_training_samples'].astype(int).unique().tolist()[0]
                assert NUM_TRAIN_SAMPLES is None or NUM_TRAIN_SAMPLES == curr_num_train_samples
                NUM_TRAIN_SAMPLES = curr_num_train_samples
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
            args.metric: ['mean', 'std'], # mean and std over all ENVIRONMENTS; models still separate
            TP_NAME: ['mean', 'std'],
            TN_NAME: ['mean', 'std'],
            FP_NAME: ['mean', 'std'],
            FN_NAME: ['mean', 'std'],
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
        unique_x_values_list = df_model[DOF_KEY].unique().tolist()
        unique_x_values_list.sort()
        print(unique_x_values_list)
        assert len(unique_x_values_list) == 6 # number of distinct DoF

        y_best = list()
        y_medians = list()
        y_iqrs = list()
        baselines = list()
        for x_val in unique_x_values_list:
            all_rows_with_x_val = df_model[df_model[DOF_KEY] == x_val]
            
            best_metric_val = all_rows_with_x_val[(args.metric, 'mean')].min() \
                if args.invert_metric else all_rows_with_x_val[(args.metric, 'mean')].max()
            y_best.append(best_metric_val)

            median_metric_val = all_rows_with_x_val[(args.metric, 'mean')].median()
            y_medians.append(median_metric_val)

            iqr_metric_val = stats.iqr(all_rows_with_x_val[(args.metric, 'mean')].tolist())
            y_iqrs.append(iqr_metric_val)

            if args.metric.lower() in [ACCURACY.lower(), TPR.lower(), TNR.lower()]:
                tp = all_rows_with_x_val[(TP_NAME, 'mean')]
                tn = all_rows_with_x_val[(TN_NAME, 'mean')]
                fp = all_rows_with_x_val[(FP_NAME, 'mean')]
                fn = all_rows_with_x_val[(FN_NAME, 'mean')]

                number_collisions = (tp + fn).round().unique()
                number_free = (tn + fp).round().unique()

                print(f'\taverage number of collisions for {x_val} DoF: {number_collisions}')
                assert len(number_collisions) == 1
                number_collisions = number_collisions[0]
                assert len(number_free) == 1
                number_free = number_free[0]
                assert number_collisions + number_free == NUM_TEST_SAMPLES

                if args.metric.lower() == ACCURACY.lower():
                    numerator = max(number_collisions, number_free)
                    value = numerator / (number_collisions + number_free) # majority rule accuracy
                elif args.metric.lower() == TPR.lower():
                    value = number_collisions / (number_collisions + number_free) # random guess collision proportion
                elif args.metric.lower() == TNR.lower():
                    value = number_free / (number_collisions + number_free) # random guess free proportion
                baselines.append(value)

        plt.plot(unique_x_values_list, y_best, 
                 color=CLF_TO_MAX_COLOR[model_name], marker=CLF_TO_MAX_MARKER[model_name], label=f'{FULL_MODEL_NAME[model_name]}: Best Hyperparameters')
        error_bars=plt.errorbar(unique_x_values_list, y_medians, y_iqrs, linestyle='--', elinewidth=1.5, capsize=2,
                     color=CLF_TO_MEAN_COLOR[model_name], marker=CLF_TO_MEAN_MARKER[model_name], label=f'{FULL_MODEL_NAME[model_name]}: Median Performance')
        error_bars[-1][0].set_linestyle('--')

        all_y_medians.extend(y_medians)
        all_y_iqrs.extend(y_iqrs)

    if args.metric.lower() in [ACCURACY.lower(), TPR.lower(), TNR.lower()]:
        # plot baseline (should be same for both models)
        plt.plot(unique_x_values_list, baselines, color=(0.5, 0.5, 0.5, 0.5), 
                label='Majority Rule (Baseline)' if args.metric.lower() == ACCURACY.lower() else 'Distribution-Aware Guess (Baseline)')

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
    plt.xlabel('DoF')
    plt.xticks(unique_x_values_list)
    metric_name = args.metric.capitalize() if len(args.metric) >= 5 else args.metric.upper()
    if args.ylabel:
        plt.ylabel(args.ylabel)
    else:
        plt.ylabel(metric_name)
    plt.legend()
    plt.grid()
    plt.title(f'DoF vs {metric_name}:\n{NUM_TRAIN_SAMPLES} train, {NUM_TEST_SAMPLES} test')
    plt.savefig(f'{args.save_location}/DoF vs {metric_name}.pdf')
    plt.savefig(f'{args.save_location}/DoF vs {metric_name}.png')
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
    parser.add_argument("--data_directory", type=str, default='dof_experiment_results')
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
