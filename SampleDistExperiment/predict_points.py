import csv
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, \
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import xgboost as xgb
import time
import math
from sklearn.utils import class_weight

from deep_learning import *
from constants import *

# CONSTANTS

TN = 0
TP = 1
FN = 2
FP = 3

N_NEIGHBORS = 5
UNCERTAIN = -1
MARGIN = 0.05

# METHODS

def read_data():
    FILENAME = 'collision_data.csv'
    X, Y = [], []
    with open(FILENAME, 'r') as input:
        reader = csv.reader(input)
        next(reader)

        for row in reader:
            X.append([float(item) for item in row[:-1]])
            Y.append(int(row[-1]))
    return X, Y

def rad2deg(X):
    return [[the_theta / np.pi * 180 for the_theta in x] for x in X]

def plot_results(X, Y_actual, Y_pred):
    # first show confusion matrix
    # cm = confusion_matrix(Y_actual, Y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['free space', 'collision'])
    # disp.plot()
    # plt.title('confusion matrix for configuration space prediction of 14DOF arm')
    # plt.show()

    # traditional accuracy, including uncertain points
    print()
    accuracy = accuracy_score(y_true=Y_actual, y_pred=Y_pred)
    precision = precision_score(y_true=Y_actual, y_pred=Y_pred)
    recall = recall_score(y_true=Y_actual, y_pred=Y_pred)
    f1 = f1_score(y_true=Y_actual, y_pred=Y_pred)
    roc_auc = roc_auc_score(y_true=Y_actual, y_score=Y_pred)
    print('accuracy with all points:', round(accuracy, 3))
    print('precision with all points:', round(precision, 3))
    print('recall with all points:', round(recall, 3))
    print('f1 with all points:', round(f1, 3))
    print('roc_auc with all points:', round(roc_auc, 3))

    return {ACCURACY: round(accuracy, 3), \
            PRECISION: round(precision, 3), \
            RECALL: round(recall, 3), \
            F1: round(f1, 3), \
            ROC_AUC: round(roc_auc, 3)}

def set_train_data_distribution(X_train, Y_train, num_points=20000, percent_collision=0.5):
    n_coll_points = int(percent_collision * num_points)
    n_free_points = num_points - n_coll_points

    n_available_coll_points = Y_train.count(1)
    n_available_free_points = Y_train.count(0)

    print(n_free_points, n_coll_points, n_available_free_points, n_available_coll_points)

    assert n_coll_points <= n_available_coll_points
    assert n_free_points <= n_available_free_points

    indices_coll_points = [i for i in range(len(Y_train)) if Y_train[i] == 1]
    indices_free_points = [i for i in range(len(Y_train)) if Y_train[i] == 0]

    desired_indices = indices_coll_points[:n_coll_points] + indices_free_points[:n_free_points]

    return [X_train[i] for i in desired_indices], [Y_train[i] for i in desired_indices]

def evaluate(X, Y, test_size):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, \
        test_size=test_size, random_state=42)

    train_set_size = 20000
    test_set_size = 80000

    X_test, Y_test = X_test[:test_set_size], Y_test[:test_set_size]

    X_train_small, Y_train_small = X_train[:train_set_size], Y_train[:train_set_size]
    X_train_balanced, Y_train_balanced = set_train_data_distribution(X_train, Y_train, num_points=train_set_size, percent_collision=0.5)

    clf_xgb = xgb.XGBClassifier(booster='gbtree', \
        n_estimators=100, \
        tree_method='hist', \
        eta=0.1, \
        max_depth=10, \
        objective="binary:logistic", \
        random_state=1)
    clf_dummy = DummyClassifier(strategy="most_frequent")

    clfs = {XGBOOST: clf_xgb, DUMMY: clf_dummy}

    all_clf_results_both_splits = dict()

    for dist, X_train, Y_train in zip([ORIGINAL_DIST, BALANCED_DIST], [X_train_small, X_train_balanced], [Y_train_small, Y_train_balanced]):
        print('training dataset size:', len(X_train))
        print('testing dataset size:', len(X_test))
        print('# DOF:', len(X_test[0]))

        num_collision = len([i for i in Y if i == 1])
        num_free = len(Y) - num_collision
        collision_ratio = num_collision / len(Y)
        print('% of points that are collision: {}'.format(collision_ratio))

        all_clf_results = {PERCENT_COLLISION : round(collision_ratio, 3)}

        for clf_name in clfs:
            print('\n***\nClassifier:', clf_name, '\n***')
            clf = clfs[clf_name]

            ## START TRAIN ##
            start = time.time()
            # START ACTION
            if clf_name == 'DL':
                clf.fit(X_train, Y_train, DOF = len(X_train[0]))
            else:
                clf.fit(X_train, Y_train)
            # END ACTION
            end = time.time()
            elapsed_train = round(end - start, 3)
            print('train time (s) on {} points: {}'.format(len(X_train), elapsed_train))
            ## END TRAIN ##

            ## START TEST ##
            start = time.time()
            # START ACTION
            Y_pred = clf.predict(X_test)
            # END ACTION
            end = time.time()
            elapsed_test = round(end - start, 3)
            print('test time (s) on {} points: {}'.format(len(X_test), elapsed_test))
            ## END TEST ##

            time_results = {TRAIN_TIME : elapsed_train, TEST_TIME : elapsed_test, TRAIN_SIZE : len(X_train), TEST_SIZE : len(X_test)}
            correctness_results = plot_results(X=X_test, Y_actual=Y_test, Y_pred=Y_pred)

            full_individual_results = {**time_results, **correctness_results}

            all_clf_results[clf_name] = full_individual_results

        all_clf_results[DOF_KEY] = len(X_test[0])

        all_clf_results_both_splits[dist] = all_clf_results

    return all_clf_results_both_splits

def main(test_size=0.8):
    X, Y = read_data()

    res = evaluate(X, Y, test_size=test_size)

    return res

if __name__ == "__main__":
    main()
