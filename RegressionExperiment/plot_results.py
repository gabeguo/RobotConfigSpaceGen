import json
import matplotlib.pyplot as plt
from constants import *
import numpy as np

RESULTS_FNAME = 'results.json'

def plot_results(results, METRIC, \
    SCALE_METRIC=None, SCALE_FACTOR=1, \
    ALT_METRIC_NAME=None, INCLUDE_SIMULATION_TIME=False, \
    SIMULATION_VALUE=None, inverse=False):

    plt.rcParams.update({'font.size': 12.5})

    if ALT_METRIC_NAME is None:
        ALT_METRIC_NAME = METRIC

    y_by_clf = dict()
    epsilon_by_clf = dict()

    for num_robots in results:
        dof = int(num_robots) * DOF_PER_ROBOT
        curr_experiment = results[num_robots]

        # only when analyzing test time or total time #
        y_by_clf[SIMULATION] = [1 for i in range(11)]
        epsilon_by_clf[SIMULATION] = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

        # go through results for every classifier
        for clf_name in curr_experiment:
            # edge cases
            if type(curr_experiment[clf_name]) is not dict:
                continue

            # initialize data storage
            y_by_clf[clf_name] = list()
            epsilon_by_clf[clf_name] = list()

            curr_clf = curr_experiment[clf_name]

            for epsilon in curr_clf:
                try:
                    float(epsilon)
                except ValueError:
                    continue
                curr_y = curr_clf[epsilon][METRIC]
                epsilon = float(epsilon)

                epsilon_by_clf[clf_name].append(epsilon)
                y_by_clf[clf_name].append(curr_y)

    # actually do the plotting
    styles = [':', '-', '--', '.-', '^-']
    markers = ['o', '*', '^']
    for clf_name in y_by_clf:
        if inverse:
            y_vals = [1 - y for y in y_by_clf[clf_name]]
        else:
            y_vals = [y for y in y_by_clf[clf_name]]
        plt.plot(epsilon_by_clf[clf_name], y_vals, styles.pop(0), alpha=0.7, marker=markers.pop(0), label=clf_name)

    plt.xlabel('$\epsilon$')
    plt.ylabel(ALT_METRIC_NAME)
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    plt.legend()
    plt.grid()
    title='{} as function of $\epsilon$'.format(ALT_METRIC_NAME)
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

def plot_roc_auc_error(results):
    plot_results(results, ROC_AUC, ALT_METRIC_NAME='Error (1 - ROC_AUC)', \
        SIMULATION_VALUE=1, inverse=True)
    return

def plot_accuracy_error(results):
    plot_results(results, ACCURACY, ALT_METRIC_NAME='Error (1 - Accuracy)', \
        SIMULATION_VALUE=1, inverse=True)
    return

def plot_percentage_included(results):
    plot_results(results, PERCENT_CONFIDENT, ALT_METRIC_NAME='Proportion of Points Included', \
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
    # plot_roc_auc(results)
    # plot_accuracy(results)
    plot_roc_auc_error(results)
    plot_accuracy_error(results)
    plot_percentage_included(results)
