import matplotlib.pyplot as plt
import numpy as np
from constants import *
import os
import json
import re
import pandas as pd

# Thanks ChatGPT!
def get_seed_number(string):
    match = re.search('seed(\d+)', string)
    if match:
        return int(match.group(1))
    else:
        return None

# Thanks ChatGPT!
def load_json_files_pd(args, COMPARISON_VARIABLES, the_x_var_key, expected_num_x_vals,
                       baseline_data_dir='simulation_data'):
    DOF = None
    NUM_TRAIN_SAMPLES = None
    NUM_TEST_SAMPLES = None
    # Load all JSON files in the directory into a list of DataFrames
    dataframes = []
    baseline_by_x_var = dict()
    for filename in os.listdir(args.data_directory):
        if filename.endswith(".json"):
            with open(os.path.join(args.data_directory, filename), 'r') as f:
                data = json.load(f)
                if get_seed_number(data['dataset_name']) not in args.seeds:
                    continue
                df = pd.json_normalize(data)
                # get DoF
                df[DOF_KEY] = 7 * df['dataset_name'].str.extract('(\d+)').astype(int)
                DOF = df[DOF_KEY].unique().tolist()[0]
                # get collision density
                df[COLLISION_DENSITY_KEY] = (df[TP_NAME] + df[FN_NAME]) / (df[TP_NAME] + df[TN_NAME] + df[FP_NAME] + df[FN_NAME])
                # get num train samples
                curr_num_train_samples = df['num_training_samples'].astype(int).unique().tolist()[0]
                NUM_TRAIN_SAMPLES = curr_num_train_samples
                # get num test samples
                curr_num_test_samples = df['num_testing_samples'].astype(int).unique().tolist()[0]
                NUM_TEST_SAMPLES = curr_num_test_samples

                # transform data
                if args.unit_rate_metric:
                    df[args.metric] /= df[TEST_SIZE]

                dataframes.append(df)

                # get PyBullet baseline
                baseline_simulation_data_path = f"{args.data_directory}/../{baseline_data_dir}/argsAndResults_{data['dataset_name']}.json"
                assert os.path.exists(baseline_simulation_data_path)
                with open(baseline_simulation_data_path, 'r') as f_sim:
                    baseline_simulation_data = json.load(f_sim)
                    the_baseline_col_time = baseline_simulation_data[COLLISION_TIME]
                    if args.unit_rate_metric:
                        the_baseline_col_time /= baseline_simulation_data[SAMPLE_SIZE]
                    assert len(df[the_x_var_key]) == 1
                    curr_x_val = df[the_x_var_key].iloc[0]
                    if isinstance(curr_x_val, pd.Series):
                        curr_x_val = tuple(curr_x_val)
                    if curr_x_val not in baseline_by_x_var:
                        baseline_by_x_var[curr_x_val] = set()
                    baseline_by_x_var[curr_x_val].add(the_baseline_col_time)
    #print(len(dataframes))
    # Concatenate all the DataFrames into a single DataFrame
    df = pd.concat(dataframes, ignore_index=True)

    if COMPARISON_VARIABLES is not None:
        # Take mean over all environments at certain DoF for each model (where distinct hyperparams mean distinct model)
        # Group by the comparison variables and compute the mean and std of args.metric
        df = df.groupby(list(COMPARISON_VARIABLES)).agg(
            {
                args.metric: ['mean', 'std'], # mean and std over all ENVIRONMENTS; models still separate
                TP_NAME: ['mean', 'std'],
                TN_NAME: ['mean', 'std'],
                FP_NAME: ['mean', 'std'],
                FN_NAME: ['mean', 'std'],
            }
        ).reset_index()
    else:
        assert the_x_var_key == COLLISION_DENSITY_KEY

    # sanity check baseline_by_x_var
    assert len(baseline_by_x_var) == expected_num_x_vals
    for the_curr_x_var in baseline_by_x_var:
        if the_x_var_key == COLLISION_DENSITY_KEY:
            assert len(baseline_by_x_var[the_curr_x_var]) == 1, f"{len(baseline_by_x_var[the_curr_x_var])}, {baseline_by_x_var[the_curr_x_var]}"
        else:
            assert len(baseline_by_x_var[the_curr_x_var]) == 3, f"{len(baseline_by_x_var[the_curr_x_var])}, {baseline_by_x_var[the_curr_x_var]}"
        baseline_by_x_var[the_curr_x_var] = np.mean(list(baseline_by_x_var[the_curr_x_var]))
    # Group by the comparison variables and compute the mean and std of args.metric
    return df, baseline_by_x_var, DOF, NUM_TRAIN_SAMPLES, NUM_TEST_SAMPLES


def plot_results(df, baseline_by_level, y_values, the_x_var_key, num_test_samples, 
                 expected_unique_x_val_length, args, 
                 x_val_to_label=None, all_model_names=None):
    all_y_medians = list()
    all_y_lowers = list()
    all_y_uppers = list()
    all_y_best = list()
    if all_model_names is None:
        all_model_names = [DL, DL_CUDA, FASTRON] if args.include_gpu else [DL, FASTRON]
    for model_name in all_model_names:
        df_model = df[df['model_name'] == model_name]
        
        # Extract maxes, means, and standard deviations for x_metric and y_metric
        if isinstance(the_x_var_key, list) or isinstance(the_x_var_key, tuple):
            unique_x_values_list = df_model[the_x_var_key].drop_duplicates().values.tolist()
            print(unique_x_values_list)
        else:
            unique_x_values_list = df_model[the_x_var_key].unique().tolist()
        unique_x_values_list.sort()
        assert len(unique_x_values_list) == expected_unique_x_val_length # number of distinct collision densities

        y_best = list()
        y_medians = list()
        y_uppers = list()
        y_lowers = list()
        baselines = list()
        for x_val in unique_x_values_list:
            print(df_model[the_x_var_key])
            if isinstance(the_x_var_key, tuple) or isinstance(the_x_var_key, list):
                mask = (df_model[the_x_var_key[0]] == x_val[0]) & \
                        (df_model[the_x_var_key[1]] == x_val[1]) & \
                        (df_model[the_x_var_key[2]] == x_val[2]) & \
                        (df_model[the_x_var_key[3]] == x_val[3])
                all_rows_with_x_val = df_model[mask]
            else:
                all_rows_with_x_val = df_model[df_model[the_x_var_key] == x_val]

            if the_x_var_key in [DOF_KEY, COLLISION_DENSITY_KEY]:
                if DL in model_name:
                    assert len(all_rows_with_x_val) == 27
                else:
                    assert len(all_rows_with_x_val) == 54
            else:
                assert the_x_var_key == 'num_training_samples' or \
                    isinstance(the_x_var_key, tuple) or isinstance(the_x_var_key, list)
                if DL in model_name:
                    assert len(all_rows_with_x_val) == 27
                else:
                    assert len(all_rows_with_x_val) == 16
            
            if the_x_var_key == COLLISION_DENSITY_KEY:
                best_metric_val = all_rows_with_x_val[args.metric].min() \
                    if args.invert_metric else all_rows_with_x_val[args.metric].max()
                median_metric_val = all_rows_with_x_val[args.metric].median()
                lower_bound = np.percentile(all_rows_with_x_val[args.metric], q=25)
                upper_bound = np.percentile(all_rows_with_x_val[args.metric], q=75)
            else:
                best_metric_val = all_rows_with_x_val[(args.metric, 'mean')].min() \
                    if args.invert_metric else all_rows_with_x_val[(args.metric, 'mean')].max()
                median_metric_val = all_rows_with_x_val[(args.metric, 'mean')].median()
                lower_bound = np.percentile(all_rows_with_x_val[(args.metric, 'mean')], q=25)
                upper_bound = np.percentile(all_rows_with_x_val[(args.metric, 'mean')], q=75)
            y_best.append(best_metric_val)
            y_medians.append(median_metric_val)
            y_lowers.append(lower_bound)
            y_uppers.append(upper_bound)

            # get baseline
            if args.metric.lower() in [ACCURACY.lower(), TPR.lower(), TNR.lower()]:
                if the_x_var_key == COLLISION_DENSITY_KEY:
                    tp = all_rows_with_x_val[TP_NAME]
                    tn = all_rows_with_x_val[TN_NAME]
                    fp = all_rows_with_x_val[FP_NAME]
                    fn = all_rows_with_x_val[FN_NAME]
                else:
                    tp = all_rows_with_x_val[(TP_NAME, 'mean')]
                    tn = all_rows_with_x_val[(TN_NAME, 'mean')]
                    fp = all_rows_with_x_val[(FP_NAME, 'mean')]
                    fn = all_rows_with_x_val[(FN_NAME, 'mean')]

                number_collisions = (tp + fn).round().unique()
                number_free = (tn + fp).round().unique()

                #print(f'\taverage number of collisions at {x_val} collision density: {number_collisions}')
                assert len(number_collisions) == 1
                number_collisions = number_collisions[0]
                assert len(number_free) == 1
                number_free = number_free[0]
                assert number_collisions + number_free == num_test_samples

                if args.metric.lower() == ACCURACY.lower():
                    numerator = max(number_collisions, number_free)
                    value = numerator / (number_collisions + number_free) # majority rule accuracy
                elif args.metric.lower() == TPR.lower():
                    value = number_collisions / (number_collisions + number_free) # random guess collision proportion
                elif args.metric.lower() == TNR.lower():
                    value = number_free / (number_collisions + number_free) # random guess free proportion
                baselines.append(value)

        if x_val_to_label is not None:
            assert isinstance(unique_x_values_list[0], tuple) or isinstance(unique_x_values_list[0], list)
            print('unique x values:', unique_x_values_list)
            horizontal_plot_values = [_ for _ in range(len(x_val_to_label))]
            horizontal_plot_labels = [x_val_to_label[tuple(the_x_val)] for the_x_val in unique_x_values_list]
            plt.xticks(ticks=horizontal_plot_values, labels=horizontal_plot_labels)
        else:
            horizontal_plot_values = unique_x_values_list
        plt.plot(horizontal_plot_values, y_best, 
                 color=CLF_TO_MAX_COLOR[model_name], marker=CLF_TO_MAX_MARKER[model_name], label=f'{FULL_MODEL_NAME[model_name]}: Best')
        y_errors = np.stack((np.array(y_medians) - np.array(y_lowers), 
                             np.array(y_uppers) - np.array(y_medians)), 
                             axis=0)
        assert y_errors.shape == (2, len(y_lowers))
        if args.plot_medians:
            plt.plot(horizontal_plot_values, y_medians, linestyle='--',
                        color=CLF_TO_MEAN_COLOR[model_name], marker=CLF_TO_MEAN_MARKER[model_name], label=f'{FULL_MODEL_NAME[model_name]}: Median')
            if args.disable_error_bars:
                y_errors = np.zeros_like(y_errors)
            else:
                plt.fill_between(horizontal_plot_values, y_lowers, y_uppers, color=CLF_TO_MEAN_COLOR[model_name], alpha=0.15,
                                linestyle='--')

        all_y_medians.extend(y_medians)
        all_y_lowers.extend(y_lowers)
        all_y_uppers.extend(y_uppers)
        all_y_best.extend(y_best)
    
    if args.metric.lower() in [ACCURACY.lower(), TPR.lower(), TNR.lower()]:
        # plot baseline (should be same for both models)
        plt.plot(horizontal_plot_values, baselines, color=(0.5, 0.5, 0.5, 0.5), 
                label='Majority Rule' if args.metric.lower() == ACCURACY.lower() else 'Distribution-Aware Guess')

    if not args.disable_error_bars:
        ymin = min(min(all_y_lowers), min(all_y_best))
        ymax = max(max(all_y_uppers), max(all_y_best))
    else:
        ymin = min(min(all_y_medians), min(all_y_best))
        ymax = max(max(all_y_medians), max(all_y_best))

    if args.metric.lower() == TEST_TIME.lower():
        the_values = [the_curr_val for \
                                   the_curr_val in baseline_by_level]
        the_values.sort()
        the_baseline_times = [baseline_by_level[the_curr_val] for \
                              the_curr_val in the_values]
        plt.plot(the_values, the_baseline_times,
                 color='purple', linestyle='-.', marker='p', alpha=0.5, label='GJK (PyBullet)')
        ymin = min(ymin, min(the_baseline_times))
        ymax = max(ymax, max(the_baseline_times))
    
    if args.include_gpu:
        plt.yscale('log')
    yspan = ymax - ymin
    plt.ylim(ymin - yspan * 0.1 , ymax + yspan * 0.1)

    return