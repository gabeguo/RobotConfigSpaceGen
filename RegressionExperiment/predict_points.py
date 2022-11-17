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

# Gets correctness of points, only including those that are within +/- epsilon
# of the true label
def get_correctness(Y_actual, Y_pred, epsilon):
    # traditional accuracy, including uncertain points
    print('\nepsilon = {}'.format(epsilon))

    # Use threshold to calculate binary values & throw out points which we aren't confident enough in
    considered_indices = [i \
        for i in range(len(Y_pred)) \
        if min(Y_pred[i], 1 - Y_pred[i]) <= epsilon]

    Y_groundTruth = [Y_actual[i] for i in considered_indices]
    Y_binarized = [int(round(Y_pred[i])) for i in considered_indices]

    # Get percentage of points thrown out
    percent_of_points_considered = len(considered_indices) / len(Y_pred)
    print('{} of points considered'.format(percent_of_points_considered))

    # Get scores
    accuracy = accuracy_score(y_true=Y_groundTruth, y_pred=Y_binarized)
    precision = precision_score(y_true=Y_groundTruth, y_pred=Y_binarized)
    recall = recall_score(y_true=Y_groundTruth, y_pred=Y_binarized)
    f1 = f1_score(y_true=Y_groundTruth, y_pred=Y_binarized)
    roc_auc = roc_auc_score(y_true=Y_groundTruth, y_score=Y_binarized)
    print('accuracy with all points:', round(accuracy, 3))
    print('precision with all points:', round(precision, 3))
    print('recall with all points:', round(recall, 3))
    print('f1 with all points:', round(f1, 3))
    print('roc_auc with all points:', round(roc_auc, 3))

    return {ACCURACY: round(accuracy, 3), \
            PRECISION: round(precision, 3), \
            RECALL: round(recall, 3), \
            F1: round(f1, 3), \
            ROC_AUC: round(roc_auc, 3), \
            PERCENT_CONFIDENT: round(percent_of_points_considered, 3)}

def evaluate(X, Y, test_size):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, \
        test_size=test_size, random_state=42)

    clf_dummy = DummyClassifier(strategy="most_frequent")
    clf_xgb = xgb.XGBRegressor(booster='gbtree', \
        n_estimators=100, \
        tree_method='hist', \
        eta=0.1, \
        max_depth=10, \
        objective="reg:squarederror", \
        random_state=1)

    clfs = {XGBOOST: clf_xgb, DUMMY: clf_dummy}

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
        full_individual_results = {**time_results}
        # test different thresholds
        if clf_name != DUMMY:
            for epsilon in [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
                curr_correctness_results = get_correctness(Y_actual=Y_test, Y_pred=Y_pred, epsilon=epsilon)
                full_individual_results[epsilon] = curr_correctness_results

        #full_individual_results = {**time_results, **correctness_results}

        all_clf_results[clf_name] = full_individual_results

    all_clf_results[DOF_KEY] = len(X_test[0])

    return all_clf_results

def main(test_size=0.8):
    X, Y = read_data()

    res = evaluate(X, Y, test_size=test_size)

    return res

if __name__ == "__main__":
    main()
