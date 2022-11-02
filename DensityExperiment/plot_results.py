import json
import matplotlib.pyplot as plt
from constants import *
import numpy as np

RESULTS_FNAME = 'results.json'

def plot_results(results, METRIC, \
    SCALE_METRIC=None, SCALE_FACTOR=1, \
    ALT_METRIC_NAME=None, INCLUDE_SIMULATION_TIME=False, \
    SIMULATION_VALUE=None):

    plt.rcParams.update({'font.size': 12.5})

    if ALT_METRIC_NAME is None:
        ALT_METRIC_NAME = METRIC

    y_by_clf = dict()
    nums_obstacles = list()
    collision_percentages = list()

    y_by_clf[SIMULATION] = list()

    for num_obstacles in results:
        nums_obstacles.append(num_obstacles)

        curr_experiment = results[num_obstacles]

        collision_percent = curr_experiment[PERCENT_COLLISION]
        collision_percentages.append(collision_percent)

        # only when analyzing test time or total time #
        if INCLUDE_SIMULATION_TIME:
            if METRIC == TEST_TIME:
                y_by_clf[SIMULATION].append(SCALE_FACTOR * curr_experiment[SIMULATION_TIME] / curr_experiment[SAMPLE_SIZE])
            elif METRIC == TOTAL_TIME:
                y_by_clf[SIMULATION].append(SCALE_FACTOR * curr_experiment[SIMULATION_TIME])
            else:
                raise ValueError('only include simulation time for testing or total time')
        else:
            y_by_clf[SIMULATION].append(SCALE_FACTOR * SIMULATION_VALUE)
        # only when analyzing test time or total time #

        for clf_name in curr_experiment:
            if type(curr_experiment[clf_name]) is not dict:
                continue
            if clf_name not in y_by_clf:
                y_by_clf[clf_name] = list()
            curr_clf = curr_experiment[clf_name]
            # special case for total time, as it's not stored in results, so we need to calculate it
            if METRIC == TOTAL_TIME:
                train_data_gather_time = curr_experiment[SIMULATION_TIME] \
                    * (curr_experiment[XGBOOST][TRAIN_SIZE] / (curr_experiment[SAMPLE_SIZE]))
                total_time = curr_experiment[clf_name][TRAIN_TIME] + \
                    curr_experiment[clf_name][TEST_TIME] + \
                    train_data_gather_time
                y_by_clf[clf_name].append(total_time)
                continue
            if SCALE_METRIC is None:
                y_by_clf[clf_name].append(SCALE_FACTOR * curr_clf[METRIC])
            else:
                y_by_clf[clf_name].append(SCALE_FACTOR * curr_clf[METRIC] / curr_clf[SCALE_METRIC])

    styles = [':', '-', '--', '.-', '^-']
    for clf_name in y_by_clf:
        plt.plot(collision_percentages, y_by_clf[clf_name], styles.pop(0), marker='o', alpha=0.7, label=clf_name)
        """
        # from https://www.tutorialspoint.com/showing-points-coordinates-in-a-plot-in-python-using-matplotlib
        for i, j in zip(dofs, y_by_clf[clf_name]):
            plt.text(i, j, '({}, {})'.format(i, round(j, 3)))
        """

    plt.xlabel('Collision density')
    plt.ylabel(ALT_METRIC_NAME)
    plt.xticks([i / 10 for i in range(0, 10+1)])
    #plt.xticks(collision_percentages)
    plt.legend()
    plt.grid()
    title='Collision density vs {}'.format(ALT_METRIC_NAME)
    plt.title(title)

    plt.savefig('{}/{}.pdf'.format(GRAPH_FOLDER_NAME, title))
    plt.show()

    return

def plot_roc_auc(results):
    plot_results(results, ROC_AUC, ALT_METRIC_NAME='ROC AUC', \
        SIMULATION_VALUE=1)
    return

def plot_accuracy(results):
    plot_results(results, ACCURACY, ALT_METRIC_NAME='Accuracy', \
        SIMULATION_VALUE=1)
    return

def plot_inference_time(results):
    plot_results(results, TEST_TIME, SCALE_METRIC=TEST_SIZE, SCALE_FACTOR=MS_PER_SEC, \
        ALT_METRIC_NAME='time (ms) per inference', INCLUDE_SIMULATION_TIME=True)
    return

def plot_train_time(results):
    plot_results(results, TRAIN_TIME, ALT_METRIC_NAME='Train Time (s)', \
        SIMULATION_VALUE=0)
    return

def plot_total_time(results):
    plot_results(results, TOTAL_TIME, ALT_METRIC_NAME='Total Time (s)', \
        INCLUDE_SIMULATION_TIME=True)
    return

if __name__ == "__main__":
    with open(RESULTS_FNAME) as fin:
        results = json.load(fin)
        print(json.dumps(results, indent=4))
    plot_roc_auc(results)
    plot_accuracy(results)
    plot_inference_time(results)
    # plot_train_time(results)
    plot_total_time(results)
