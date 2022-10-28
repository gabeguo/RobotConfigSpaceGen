import json
import matplotlib.pyplot as plt
from constants import *
import numpy as np

RESULTS_FNAME = 'results.json'

def plot_results(results, METRIC, \
    SCALE_METRIC=None, SCALE_FACTOR=1, \
    ALT_METRIC_NAME=None, INCLUDE_SIMULATION_TIME=False, \
    SIMULATION_VALUE=None):

    if ALT_METRIC_NAME is None:
        ALT_METRIC_NAME = METRIC

    y_by_clf = dict()
    dofs = list()

    y_by_clf[SIMULATION] = list()

    for num_robots in results:
        dof = int(num_robots) * DOF_PER_ROBOT
        dofs.append(dof)
        curr_experiment = results[num_robots]

        # only when analyzing test time #
        if INCLUDE_SIMULATION_TIME:
            y_by_clf[SIMULATION].append(SCALE_FACTOR * curr_experiment[SIMULATION_TIME] / curr_experiment[SAMPLE_SIZE])
        else:
            y_by_clf[SIMULATION].append(SCALE_FACTOR * SIMULATION_VALUE)
        # only when analyzing test time #

        for clf_name in curr_experiment:
            if type(curr_experiment[clf_name]) is not dict:
                continue
            if clf_name not in y_by_clf:
                y_by_clf[clf_name] = list()
            curr_clf = curr_experiment[clf_name]
            if SCALE_METRIC is None:
                y_by_clf[clf_name].append(SCALE_FACTOR * curr_clf[METRIC])
            else:
                y_by_clf[clf_name].append(SCALE_FACTOR * curr_clf[METRIC] / curr_clf[SCALE_METRIC])

    styles = [':', '-', '--', '.-', '^-']
    for clf_name in y_by_clf:
        plt.plot(dofs, y_by_clf[clf_name], styles.pop(0), alpha=0.7, label=clf_name)

    plt.xlabel('DOF')
    plt.ylabel(ALT_METRIC_NAME)
    plt.xticks(dofs)
    plt.legend()
    plt.grid()
    title='DOF vs {}'.format(ALT_METRIC_NAME)
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

def plot_pareto(results, num_robots=3, show_total_time=True):
    plt.xlabel('Error = 1 - ROC_AUC')
    if show_total_time:
        plt.ylabel('Total Time Cost (s) = \nTrain + Test + Simulation')
    else:
        plt.ylabel('Time per Inference (ms)')
    plt.grid()

    curr_experiment = results[str(num_robots)]
    dof = num_robots * DOF_PER_ROBOT
    plt.title('Pareto Chart\nDOF = {}, |Train| = {}, |Test| = {}, % Collision = {}'.format(\
        dof, curr_experiment[XGBOOST][TRAIN_SIZE], curr_experiment[XGBOOST][TEST_SIZE], \
        round(100 * curr_experiment[PERCENT_COLLISION], 1)))

    plt.scatter(x=[1-1], y=[curr_experiment[SIMULATION_TIME]], label='PyBullet Simulation')
    train_data_gather_time = curr_experiment[SIMULATION_TIME] \
        * (curr_experiment[XGBOOST][TRAIN_SIZE] / (curr_experiment[SAMPLE_SIZE]))
    for clf in [XGBOOST, KNN, DUMMY, DL]:
        if clf not in curr_experiment:
            continue
        if show_total_time:
            total_time = curr_experiment[clf][TRAIN_TIME] + \
                curr_experiment[clf][TEST_TIME] + \
                train_data_gather_time
            y_val = total_time
        else:
            time_per_inference = MS_PER_SEC * curr_experiment[clf][TEST_TIME] / curr_experiment[clf][TEST_SIZE]
            y_val = time_per_inference
            #print(clf, dof, y_val)
        plt.scatter(x=[1 - curr_experiment[clf][ROC_AUC]], \
            y=[y_val], \
            label=clf)

    plt.legend()
    plt.savefig('{}/Pareto_{}DOF_{}.pdf'.format(GRAPH_FOLDER_NAME, dof, 'totalTime' if show_total_time else 'inferenceTime'))
    plt.show()

    return


if __name__ == "__main__":
    with open(RESULTS_FNAME) as fin:
        results = json.load(fin)
        print(json.dumps(results, indent=4))
    plot_roc_auc(results)
    plot_accuracy(results)
    plot_inference_time(results)
    plot_train_time(results)
    for i in range(2, max([int(x) for x in results.keys()]) + 1):
        plot_pareto(results, num_robots=i, show_total_time=True)
        plot_pareto(results, num_robots=i, show_total_time=False)
