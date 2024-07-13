import json
import matplotlib.pyplot as plt
from constants import *
import numpy as np
import os
import argparse
import pandas as pd
import scipy.stats as stats

from common_functions import *

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

def label_plot(args, DOF, NUM_TRAIN_SAMPLES, NUM_TEST_SAMPLES):
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

def main(args):
    plt.rcParams.update({'figure.figsize': (8, 6)})
    plt.rcParams.update({'font.size': 11})

    # get all the data points
    df_mean_std, baseline_by_dof, DOF, NUM_TRAIN_SAMPLES, NUM_TEST_SAMPLES \
        = load_json_files_pd(args=args, COMPARISON_VARIABLES=COMPARISON_VARIABLES,
                            the_x_var_key=DOF_KEY, expected_num_x_vals=6) 
    # plot pareto frontier
    plot_results(df=df_mean_std, baseline_by_level=baseline_by_dof, y_values=df_mean_std[(args.metric, 'mean')].tolist(),
                 the_x_var_key=DOF_KEY, num_test_samples=NUM_TEST_SAMPLES, 
                 expected_unique_x_val_length=6, args=args)
    label_plot(args, DOF=DOF, NUM_TRAIN_SAMPLES=NUM_TRAIN_SAMPLES, 
               NUM_TEST_SAMPLES=NUM_TEST_SAMPLES)

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
    parser.add_argument("--plot_medians", action='store_true')

    # Execute the parse_args() method
    args = parser.parse_args()

    os.makedirs(args.save_location, exist_ok=True)

    print('graph dof')
    main(args)
