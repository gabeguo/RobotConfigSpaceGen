import json
import matplotlib.pyplot as plt
from constants import *
import numpy as np

RESULTS_FNAME = 'merged_sample_size_results.json'

def plot_results(results):

    # build results
    train_ratios = list()
    train_sizes = list()

    accs = {DUMMY:list(), XGBOOST:list(), SIMULATION:[1 for key in results]}
    rocs = {DUMMY:list(), XGBOOST:list(), SIMULATION:[1 for key in results]}
    inv_accs = {DUMMY:list(), XGBOOST:list(), SIMULATION:[0 for key in results]}
    inv_rocs = {DUMMY:list(), XGBOOST:list(), SIMULATION:[0 for key in results]}

    total_times = {DUMMY:list(), XGBOOST:list(), SIMULATION:list()}
    inference_times = {DUMMY:list(), XGBOOST:list(), SIMULATION:list()}

    for key in results:
        curr_results = results[key]

        test_ratio = float(key)
        train_ratio = 1 - test_ratio
        train_ratios.append(train_ratio)

        dof = curr_results[DOF_KEY]
        num_points = curr_results[SAMPLE_SIZE]
        num_training = int(train_ratio * num_points)
        num_testing = int(test_ratio * num_points)

        train_sizes.append(num_training)

        curr_sim_time = curr_results[SIMULATION_TIME]

        total_times[SIMULATION].append(curr_sim_time / SEC_PER_MIN)
        inference_times[SIMULATION].append(MS_PER_SEC * curr_sim_time / num_points)

        for clf in [DUMMY, XGBOOST]:
            curr_acc = curr_results[clf][ACCURACY]
            curr_roc = curr_results[clf][ROC_AUC]
            curr_total_time = (curr_results[clf][TRAIN_TIME] \
                + curr_results[clf][TEST_TIME] \
                + train_ratio * curr_sim_time) \
                / SEC_PER_MIN
            curr_inference_time = MS_PER_SEC * curr_results[clf][TEST_TIME] / (curr_results[SAMPLE_SIZE] * test_ratio)

            accs[clf].append(curr_acc)
            rocs[clf].append(curr_roc)
            inv_accs[clf].append(1 - curr_acc)
            inv_rocs[clf].append(1 - curr_roc)

            total_times[clf].append(curr_total_time)
            inference_times[clf].append(curr_inference_time)

    # do the plotting
    plt.rcParams.update({'font.size': 12})
    for metric_pair in zip(['Error (1 - Accuracy)', 'Error (1 - ROC_AUC)', 'Total Time (min)', 'Inference Time (ms)'], [inv_accs, inv_rocs, total_times, inference_times]):
        metric_name, metric_data = metric_pair
        plt.title('{} as a Function of Sample Size:\n{} DOF; {:,} Total Configurations'.format(metric_name, dof, num_points))
        styles = [':', '-', '--', '.-', '^-']
        markers = ['o', '*', '^']
        for clf in [SIMULATION, XGBOOST, DUMMY]:
            #plt.plot(train_ratios, metric_data[clf], styles.pop(0), alpha=0.7, label=clf)
            plt.plot(train_sizes, metric_data[clf], styles.pop(0), marker=markers.pop(0), alpha=0.7, label=clf)
        plt.grid()
        plt.subplots_adjust(left=0.15)
        #plt.xlabel('Train Ratio')
        plt.xlabel('Number of Training Points (millions)')
        plt.ylabel(metric_name)
        plt.legend()
        plt.savefig('{}/Train Ratio vs {}_{} DOF_{} Points.pdf'.format(GRAPH_FOLDER_NAME, \
            metric_name, dof, num_points))
        plt.show()

    return

if __name__ == "__main__":
    with open(RESULTS_FNAME) as fin:
        results = json.load(fin)
        print(json.dumps(results, indent=4))

    plot_results(results)
