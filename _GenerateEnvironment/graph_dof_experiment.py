import json
import matplotlib.pyplot as plt
from constants import *
import numpy as np
import os
import argparse
import re
import pandas as pd
import scipy.stats as stats

from common_functions import plot_results

import matplotlib
matplotlib.use('Agg')

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

FULL_MODEL_NAME = {DL: 'DeepCollide (Sequential)', FASTRON: 'Fastron FK', DL_CUDA: 'DeepCollide (Parallel)'}

# Thanks ChatGPT!
def load_json_files_pd(args):
    global DOF, NUM_TRAIN_SAMPLES, NUM_TEST_SAMPLES
    DOF = None
    NUM_TRAIN_SAMPLES = None
    NUM_TEST_SAMPLES = None
    # Load all JSON files in the directory into a list of DataFrames
    dataframes = []
    baseline_by_dof = {7:set(), 14:set(), 21:set(), 28:set(), 35:set(), 42:set()}
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

                # get PyBullet baseline
                baseline_simulation_data_path = f"{args.data_directory}/../simulation_data/argsAndResults_{data['dataset_name']}.json"
                assert os.path.exists(baseline_simulation_data_path)
                with open(baseline_simulation_data_path, 'r') as f_sim:
                    baseline_simulation_data = json.load(f_sim)
                    the_baseline_col_time = baseline_simulation_data[COLLISION_TIME]
                    if args.unit_rate_metric:
                        the_baseline_col_time /= baseline_simulation_data[SAMPLE_SIZE]
                    assert len(df[DOF_KEY]) == 1
                    baseline_by_dof[df[DOF_KEY][0]].add(the_baseline_col_time)
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

    # sanity check baseline_by_dof
    assert len(baseline_by_dof) == 6
    for the_curr_dof in baseline_by_dof:
        assert len(baseline_by_dof[the_curr_dof]) == 3, f"{len(baseline_by_dof[the_curr_dof])}, {baseline_by_dof[the_curr_dof]}"
        baseline_by_dof[the_curr_dof] = np.mean(list(baseline_by_dof[the_curr_dof]))
    # Group by the comparison variables and compute the mean and std of args.metric
    return df_mean_std, baseline_by_dof

def label_plot(args):
    plt.xlabel('DoF')
    plt.xticks([7, 14, 21, 28, 35, 42])
    metric_name = args.metric.capitalize() if len(args.metric) >= 5 else args.metric.upper()
    if args.ylabel:
        plt.ylabel(args.ylabel)
    else:
        plt.ylabel(metric_name)
    plt.legend(bbox_to_anchor=(0, -0.28, 1, -0.02), loc="lower left",
        mode="expand", borderaxespad=0, ncol=3, fontsize='small')
    plt.subplots_adjust(bottom=0.2)
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
    df_mean_std, baseline_by_dof = load_json_files_pd(args) 
    # plot pareto frontier
    plot_results(df=df_mean_std, baseline_by_level=baseline_by_dof, y_values=df_mean_std[(args.metric, 'mean')].tolist(),
                 the_metric_key=DOF_KEY, num_test_samples=NUM_TEST_SAMPLES, 
                 expected_unique_x_val_length=6, args=args)
    label_plot(args)

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
    parser.add_argument('--include_gpu', action='store_true')
    parser.add_argument('--disable_error_bars', action='store_true')

    # Execute the parse_args() method
    args = parser.parse_args()

    os.makedirs(args.save_location, exist_ok=True)

    main(args)
