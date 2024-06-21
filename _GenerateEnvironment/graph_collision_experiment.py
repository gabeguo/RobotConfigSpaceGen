import json
import matplotlib.pyplot as plt
from constants import *
import numpy as np
import os
import argparse
import re
import pandas as pd
import scipy.stats as stats

from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')

from common_functions import plot_results

# Thanks ChatGPT!
def load_json_files_pd(args):
    global DOF, NUM_TRAIN_SAMPLES, NUM_TEST_SAMPLES
    DOF = None
    NUM_TRAIN_SAMPLES = None
    NUM_TEST_SAMPLES = None
    # Load all JSON files in the directory into a list of DataFrames
    dataframes = []
    baseline_by_collision_density = dict()
    for filename in tqdm(os.listdir(args.data_directory)):
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
                # get num test samples
                curr_num_test_samples = df['num_testing_samples'].astype(int).unique().tolist()[0]
                assert NUM_TEST_SAMPLES is None or NUM_TEST_SAMPLES == curr_num_test_samples
                NUM_TEST_SAMPLES = curr_num_test_samples

                # get collision density
                df[COLLISION_DENSITY_KEY] = (df[TP_NAME] + df[FN_NAME]) / (df[TP_NAME] + df[TN_NAME] + df[FP_NAME] + df[FN_NAME])

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
                    assert len(df[COLLISION_DENSITY_KEY]) == 1
                    curr_collision_density = df[COLLISION_DENSITY_KEY][0]
                    if curr_collision_density not in baseline_by_collision_density:
                        baseline_by_collision_density[curr_collision_density] = set()
                    baseline_by_collision_density[curr_collision_density].add(the_baseline_col_time)

    #print(len(dataframes))
    # Concatenate all the DataFrames into a single DataFrame
    df = pd.concat(dataframes, ignore_index=True)

    # sanity check baseline_by_collision_density
    assert len(baseline_by_collision_density) == 36
    for curr_val in baseline_by_collision_density:
        assert len(baseline_by_collision_density[curr_val]) == 1
        baseline_by_collision_density[curr_val] = \
            np.mean(list(baseline_by_collision_density[curr_val]))

    #print(df[COLLISION_DENSITY_KEY].unique().tolist())
    # Group by the comparison variables
    return df, baseline_by_collision_density

# Thanks ChatGPT!
def get_seed_number(string):
    match = re.search('seed(\d+)', string)
    if match:
        return int(match.group(1))
    else:
        return None

def label_plot(args):
    plt.xlabel('Collision Density')
    metric_name = args.metric.capitalize() if len(args.metric) >= 5 else args.metric.upper()
    if args.ylabel:
        plt.ylabel(args.ylabel)
    else:
        plt.ylabel(metric_name)
    plt.legend(bbox_to_anchor=(0, -0.28, 1, -0.02), loc="lower left",
        mode="expand", borderaxespad=0, ncol=3, fontsize='small')
    plt.subplots_adjust(bottom=0.2)
    plt.grid()
    plt.title(f'Collision Density vs {metric_name}:\n{DOF} DoF, {NUM_TRAIN_SAMPLES} train, {NUM_TEST_SAMPLES} test')
    plt.savefig(f'{args.save_location}/Collision Density vs {metric_name}_{DOF} DoF.pdf')
    plt.savefig(f'{args.save_location}/Collision Density vs {metric_name}_{DOF} DoF.png')

    return

def main(args):
    plt.rcParams.update({'figure.figsize': (8, 6)})
    plt.rcParams.update({'font.size': 11})

    # get all the data points
    df_mean_std, baseline = load_json_files_pd(args) 
    # plot pareto frontier
    plot_results(df=df_mean_std, baseline_by_level=baseline, y_values=df_mean_std[args.metric].tolist(),
                 the_metric_key=COLLISION_DENSITY_KEY, num_test_samples=NUM_TEST_SAMPLES, 
                 expected_unique_x_val_length=36, args=args)

    label_plot(args)

    return

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add the arguments
    parser.add_argument("--data_directory", type=str, default='obstacles_experiment_results')
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
