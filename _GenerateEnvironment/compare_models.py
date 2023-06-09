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

from cspace_net import CSpaceNet
import torch
import torch.nn as nn

import random

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
    y = np.reshape(y, (-1, 1)).astype(float) # -1, 1
    y_binary = np.where(y < 0, 0, y).astype(int) # 0, 1

    # all_data = (all_data - all_data.mean()) / all_data.std()

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

    # Initialize Neural Network
    dl_model = CSpaceNet(dof=data_train.shape[1], num_freq=8, sigma=1).cuda()

    # Initialize PyFastron
    fastron = PyFastron(data_train) # where data.shape = (N, d)

    fastron.y = y_train # where y.shape = (N,)

    fastron.g = args.g # from paper for high DOF: 5
    fastron.maxUpdates = args.maxUpdates
    fastron.maxSupportPoints = args.maxSupportPoints
    fastron.beta = args.beta # from paper for high DOF: 500

    for model_name, model in zip([DL], [dl_model]):#zip([FASTRON, DL], [fastron, dl_model]):
        print('\n', model_name)

        # Train model
        start = time.time()
        if model_name == FASTRON:
            model.updateModel()
        else:
            train_deep_learning(model=model, X_train=data_train, Y_train=y_train)
        end = time.time()
        elapsed_train = end - start
        print('time elapsed in fitting on {} points for {} dof: {:.3f} seconds'.format(len(data_train), data_train.shape[1] / (3 if args.forward_kinematics_kernel else 1), elapsed_train))

        # Predict values for a test set
        start = time.time()
        if model_name == FASTRON:
            pred = model.eval(data_test) # where data_test.shape = (N_test, d) 
            print(fastron.G.shape)
            print(fastron.N)
        else:    
            pred = test_deep_learning(model, X_test=data_test)
            # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/6
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            print(params)
        end = time.time()
        elapsed_test = end - start
        print('time elapsed in testing on {} points for {} dof: {:.3f} seconds'.format(len(data_test), data_test.shape[1]  / (3 if args.forward_kinematics_kernel else 1), elapsed_test))

        # Get metrics
        pred = np.array(pred).astype(int)
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

def train_deep_learning(model, X_train, Y_train, learning_rate=1e-3, batch_size=512, train_percent=0.95):
    #model = CSpaceNet(dof=7*4, num_freq=128, sigma=1.5).cuda()

    #print(model)

    #X_train = (X_train - X_train.mean())/X_train.std()

    EPOCHS = 50

    best_val_loss = 1e6
    best_epoch = -1

    first_val_index = int(len(Y_train) * train_percent)
    X_train = torch.FloatTensor(X_train).cuda()
    Y_train = torch.FloatTensor(Y_train).cuda()

    X_val = X_train[first_val_index:]
    Y_val = Y_train[first_val_index:]
    X_train = X_train[:first_val_index]
    Y_train = Y_train[:first_val_index]

    beta = 1
    biases = biases = torch.Tensor([beta if curr_y > 0 else 1 for curr_y in Y_train]).cuda()

    percent_collision = torch.sum(Y_train == 1) / len(Y_train)
    print('percent collision:', percent_collision)

    criterion = nn.L1Loss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    for i in range(EPOCHS):
        total_loss = 0

        # train
        model.train()

        indices = list(range(0, len(X_train), batch_size))
        #print(indices)
        random.shuffle(indices)
        for idx in indices:
            #print(idx)
            model.zero_grad()

            x = X_train[idx:idx+batch_size, :]
            #print(x.shape)
            y = Y_train[idx:idx+batch_size]

            y_pred = model(x)
            
            loss = criterion(biases[idx:idx+batch_size] * y_pred.flatten(), y.flatten())
            
            total_loss += loss
            loss.backward()
            optimizer.step()
        if i % 5 == 0:
            print('loss in epoch {}: {}'.format(i, total_loss / len(Y_train)))

        # calculate validation loss
        model.eval()

        total_val_loss = 0
        for idx in range(0, len(Y_val), batch_size):
            x = X_val[idx:idx+batch_size, :]
            y = Y_val[idx:idx+batch_size]
            val_loss = criterion(model(x).flatten(), y.flatten())
            total_val_loss += val_loss
        if i % 5 == 0:
            print('\tvalidation loss epoch {}: {}'.format(i, val_loss / len(Y_val)))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = i
            torch.save(model.state_dict(), 'best_model.pth')
        
        scheduler.step()

    print('best epoch: {}'.format(best_epoch))

    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    return model

def test_deep_learning(model, X_test):
    model = model.cpu()
    #X_test = (X_test - X_test.mean()) / X_test.std()
    X_test = torch.FloatTensor(X_test)#.cuda()
    return [1 if y > 0 else -1 for y in model(X_test).flatten().tolist()]
    #return [int(y + 0.5) for y in model(X_test[:, :]).flatten().tolist()]

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_training_samples', type=int, default=30000)
    parser.add_argument('--dataset_name', type=str, default="3robots_25obstacles_seed0_")
    parser.add_argument('--g', type=int, default=5)
    parser.add_argument('--beta', type=int, default=1)
    parser.add_argument('--maxUpdates', type=int, default=100000)
    parser.add_argument('--maxSupportPoints', type=int, default=20000)
    parser.add_argument('--forward_kinematics_kernel', action='store_true')

    args = parser.parse_args()

    run_fastron(args)

if __name__ == '__main__':
    main()
