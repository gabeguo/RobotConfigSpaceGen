import pyximport
pyximport.install()

import os
import sys
sys.path.append('../fastron_python')
import numpy as np
from fastronWrapper.fastronWrapper import PyFastron
import xgboost as xgb

from sklearn.metrics import confusion_matrix
import time

import argparse
import json

from constants import *

"""
--num_training_samples 10000 \
--dataset_name "1robots_25obstacles_seed0_" \
--g 10 \
--beta 500 \
--maxUpdates 10000 \
--maxSupportPoints 10000 \
--forward_kinematics_kernel \
"""

def run_fastron(args):
    # load data
    if args.forward_kinematics_kernel:
        config_filename = '{}/linkPositions_{}.npy'.format(DATA_FOLDER, args.dataset_name)
    else: # use normalized joint angles
        config_filename = '{}/configs_{}.npy'.format(DATA_FOLDER, args.dataset_name)
    all_data = np.load(config_filename)
    print('min and max: {:.2f}, {:.2f}'.format(all_data.min(), all_data.max()))

    y = np.load('{}/labels_{}.npy'.format(DATA_FOLDER, args.dataset_name))
    y = np.reshape(y, (-1, 1)).astype(float)

    # test on 25K, unless we don't have enough data (can't overlap with train set)
    first_test_index = max(args.num_training_samples, len(all_data) - 25000)
    data_train = all_data[:args.num_training_samples]
    data_test = all_data[first_test_index:]
    y_train = y[:args.num_training_samples]
    y_test = y[first_test_index:]

    # sanity check
    print('inputs:', data_train[-2:])
    print('mean of training data:', data_train.mean())
    print('std of training data:', data_train.std())

    # Initialize XGBoost
    clf_xgb = xgb.XGBRegressor(booster='gbtree', \
        n_estimators=100, \
        tree_method='hist', \
        eta=0.5, \
        max_depth=25, \
        objective="binary:logistic", \
        random_state=1)

    # Initialize PyFastron
    fastron = PyFastron(data_train) # where data.shape = (N, d)

    fastron.y = y_train # where y.shape = (N,)

    fastron.g = args.g # from paper for high DOF: 5
    fastron.maxUpdates = args.maxUpdates
    fastron.maxSupportPoints = args.maxSupportPoints
    fastron.beta = args.beta # from paper for high DOF: 500

    for model_name, model in zip([FASTRON, XGBOOST], [fastron, clf_xgb]):
        print('\n', model_name)

        # Train model
        start = time.time()
        if model_name == FASTRON:
            model.updateModel()
        else:
            model.fit(data_train, y_train)
        end = time.time()
        elapsed_train = end - start
        print('time elapsed in fitting on {} points for {} dof: {:.3f} seconds'.format(len(data_train), data_train.shape[1] / (3 if args.forward_kinematics_kernel else 1), elapsed_train))

        # Predict values for a test set
        start = time.time()
        if model_name == FASTRON:
            pred = fastron.eval(data_test) # where data_test.shape = (N_test, d) 
        else:    
            pred = model.predict(data_test)
        end = time.time()
        elapsed_test = end - start
        print('time elapsed in testing on {} points for {} dof: {:.3f} seconds'.format(len(data_test), data_test.shape[1]  / (3 if args.forward_kinematics_kernel else 1), elapsed_test))

        # Get metrics
        cm = confusion_matrix(y_true=y_test.astype(int).flatten(), y_pred=pred.astype(int).flatten())
        print(cm)

        TN = cm[0][0]
        FN = cm[1][0]
        TP = cm[1][1]
        FP = cm[0][1]

        # traditional accuracy, including uncertain points
        acc = (TP+TN)/(TP+TN+FP+FN)
        tpr = TP/(TP+FN)
        tnr = TN/(TN+FP)
        print()
        print('accuracy:', round(acc, 3))
        print('sensitivity (TPR):', round(tpr, 3))
        print('specificity (TNR):', round(tnr, 3))

        # log results & args
        assert args.num_training_samples == data_train.shape[0]
        results = {
            ACCURACY: acc,
            TPR: tpr,
            TNR: tnr,
            TP_NAME: int(TP),
            TN_NAME: int(TN),
            FP_NAME: int(FP),
            FN_NAME: int(FN), 
            TRAIN_TIME:elapsed_train,
            TEST_TIME:elapsed_test,
            TRAIN_SIZE:args.num_training_samples,
            TEST_SIZE:data_test.shape[0],
        }
        results.update(vars(args)) # update with args

        os.makedirs(RESULTS_FOLDER, exist_ok=True)
        filename = '{}/{}Results_{}_{}_{}_{}_{}.json'.format(
            RESULTS_FOLDER, 
            model_name,
            'forwardKinematics' if args.forward_kinematics_kernel else 'normalizedAngles',
            f'{args.num_training_samples}Samples',
            f'{args.beta}Beta',
            f'{args.g}Gamma',
            args.dataset_name)
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
    return

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_training_samples', type=int, default=10000)
    parser.add_argument('--dataset_name', type=str, default="1robots_25obstacles_seed0_")
    parser.add_argument('--g', type=int, default=10)
    parser.add_argument('--beta', type=int, default=500)
    parser.add_argument('--maxUpdates', type=int, default=10000)
    parser.add_argument('--maxSupportPoints', type=int, default=10000)
    parser.add_argument('--forward_kinematics_kernel', action='store_true')

    args = parser.parse_args()

    run_fastron(args)

if __name__ == '__main__':
    main()
