import matplotlib.pyplot as plt
import numpy as np
from constants import *

def plot_results(df, baseline_by_level, y_values, the_metric_key, num_test_samples, 
                 expected_unique_x_val_length, args):
    all_y_medians = list()
    all_y_lowers = list()
    all_y_uppers = list()
    all_y_best = list()
    all_model_names = [DL, DL_CUDA, FASTRON] if args.include_gpu else [DL, FASTRON]
    for model_name in all_model_names:
        df_model = df[df['model_name'] == model_name]
        
        # Extract maxes, means, and standard deviations for x_metric and y_metric
        unique_x_values_list = df_model[the_metric_key].unique().tolist()
        unique_x_values_list.sort()
        assert len(unique_x_values_list) == expected_unique_x_val_length # number of distinct collision densities

        y_best = list()
        y_medians = list()
        y_uppers = list()
        y_lowers = list()
        baselines = list()
        for x_val in unique_x_values_list:
            all_rows_with_x_val = df_model[df_model[the_metric_key] == x_val]

            if the_metric_key in [DOF_KEY, COLLISION_DENSITY_KEY]:
                if DL in model_name:
                    assert len(all_rows_with_x_val) == 27
                else:
                    assert len(all_rows_with_x_val) == 54
            else:
                assert the_metric_key == 'num_training_samples'
                if DL in model_name:
                    assert len(all_rows_with_x_val) == 27
                else:
                    assert len(all_rows_with_x_val) == 16
            
            if the_metric_key == COLLISION_DENSITY_KEY:
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
                if the_metric_key == COLLISION_DENSITY_KEY:
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

        plt.plot(unique_x_values_list, y_best, 
                 color=CLF_TO_MAX_COLOR[model_name], marker=CLF_TO_MAX_MARKER[model_name], label=f'{FULL_MODEL_NAME[model_name]}: Best')
        y_errors = np.stack((np.array(y_medians) - np.array(y_lowers), 
                             np.array(y_uppers) - np.array(y_medians)), 
                             axis=0)
        assert y_errors.shape == (2, len(y_lowers))
        if args.disable_error_bars:
            y_errors = np.zeros_like(y_errors)
            plt.plot(unique_x_values_list, y_medians, linestyle='--',
                     color=CLF_TO_MEAN_COLOR[model_name], marker=CLF_TO_MEAN_MARKER[model_name], label=f'{FULL_MODEL_NAME[model_name]}: Median')
        else:
            error_bars=plt.errorbar(unique_x_values_list, y_medians, y_errors, linestyle='--', elinewidth=2, capsize=4,
                        color=CLF_TO_MEAN_COLOR[model_name], marker=CLF_TO_MEAN_MARKER[model_name], label=f'{FULL_MODEL_NAME[model_name]}: Median')
            error_bars[-1][0].set_linestyle('--')

        all_y_medians.extend(y_medians)
        all_y_lowers.extend(y_lowers)
        all_y_uppers.extend(y_uppers)
        all_y_best.extend(y_best)
    
    if args.metric.lower() in [ACCURACY.lower(), TPR.lower(), TNR.lower()]:
        # plot baseline (should be same for both models)
        plt.plot(unique_x_values_list, baselines, color=(0.5, 0.5, 0.5, 0.5), 
                label='Majority Rule (Baseline)' if args.metric.lower() == ACCURACY.lower() else 'Distribution-Aware Guess (Baseline)')

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

    #plt.show()