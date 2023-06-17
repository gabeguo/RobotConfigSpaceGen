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
from datetime import datetime

def log_results(y_test, pred, elapsed_train, elapsed_test, args):
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
        TEST_SIZE:y_test.flatten().shape[0],
    }
    results.update(vars(args)) # update with args

    now = datetime.now()
    # Format the date and time as a string in the format 'yy_mm_dd_hh_mm_ss'
    formatted_timestamp = now.strftime('%y_%m_%d_%H_%M_%S')

    os.makedirs(args.results_folder, exist_ok=True)
    filename = '{}/{}_{}.json'.format(
        args.results_folder, args.model_name, formatted_timestamp)
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)

def run_model(args):
    # load data
    if args.forward_kinematics_kernel:
        config_filename = '{}/linkPositions_{}.npy'.format(DATA_FOLDER, args.dataset_name)
    else: # use normalized joint angles
        config_filename = '{}/configs_{}.npy'.format(DATA_FOLDER, args.dataset_name)
    all_data = np.load(config_filename)
    print('min and max: {:.2f}, {:.2f}'.format(all_data.min(), all_data.max()))

    y = np.load('{}/labels_{}.npy'.format(DATA_FOLDER, args.dataset_name))
    y = np.reshape(y, (-1, 1)).astype(float) # -1, 1

    # test on args.num_testing_samples, unless we don't have enough data (can't overlap with train set)
    assert args.num_training_samples + args.num_testing_samples <= len(all_data)
    first_test_index = max(args.num_training_samples, len(all_data) - args.num_testing_samples)
    data_train = all_data[:args.num_training_samples]
    data_test = all_data[first_test_index:]
    y_train = y[:args.num_training_samples]
    y_test = y[first_test_index:]

    # Initialize Neural Network
    if args.model_name == DL:
        model = CSpaceNet(dof=data_train.shape[1], num_freq=args.num_freq, sigma=args.sigma).cuda()
    # Initialize PyFastron
    elif args.model_name == FASTRON:
        model = PyFastron(data_train) # where data.shape = (N, d)

        model.y = y_train # where y.shape = (N,)

        model.g = args.g # from paper for high DOF: 5
        model.maxUpdates = args.maxUpdates
        model.maxSupportPoints = args.maxSupportPoints
        model.beta = args.beta # from paper for high DOF: 500
    else:
        raise ValueError('unsupported model type')

    print('\n', args.model_name)

    # Train model
    start = time.time()
    if args.model_name == FASTRON:
        model.updateModel()
    else:
        train_deep_learning(model=model, X_train=data_train, Y_train=y_train, args=args)
    end = time.time()
    elapsed_train = end - start
    print('time elapsed in fitting on {} points for {} dof: {:.3f} seconds'.format(len(data_train), data_train.shape[1] / (3 if args.forward_kinematics_kernel else 1), elapsed_train))

    # Predict values for a test set
    start = time.time()
    if args.model_name == FASTRON:
        pred = model.eval(data_test) # where data_test.shape = (N_test, d) 
        print(model.G.shape)
        print(model.N)
    else:    
        pred = test_deep_learning(model, X_test=data_test)
    end = time.time()
    elapsed_test = end - start
    print('time elapsed in testing on {} points for {} dof: {:.3f} seconds'.format(len(data_test), data_test.shape[1]  / (3 if args.forward_kinematics_kernel else 1), elapsed_test))

    # Get metrics
    log_results(y_test=y_test, pred=pred, elapsed_train=elapsed_train, elapsed_test=elapsed_test, args=args)

    # Sanity check
    assert args.num_training_samples == data_train.shape[0]

    # Get number of params in DL model
    if args.model_name == DL:
        # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/6
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(params)

    return

def train_deep_learning(model, X_train, Y_train, args):
    best_val_loss = 1e6
    best_epoch = -1

    first_val_index = int(len(Y_train) * args.train_percent)
    X_train = torch.FloatTensor(X_train).cuda()
    Y_train = torch.FloatTensor(Y_train).cuda()

    X_val = X_train[first_val_index:]
    Y_val = Y_train[first_val_index:]
    X_train = X_train[:first_val_index]
    Y_train = Y_train[:first_val_index]

    biases = torch.Tensor([args.bias if curr_y > 0 else 1 for curr_y in Y_train]).cuda()

    percent_collision = torch.sum(Y_train == 1) / len(Y_train)
    print('percent collision:', percent_collision)

    criterion = nn.L1Loss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    for i in range(args.epochs):
        total_loss = 0

        # train
        model.train()

        indices = list(range(0, len(X_train), args.batch_size))
        #print(indices)
        random.shuffle(indices)
        for idx in indices:
            model.zero_grad()

            x = X_train[idx:idx+args.batch_size, :]
            y = Y_train[idx:idx+args.batch_size]

            y_pred = model(x)
            
            loss = criterion(biases[idx:idx+args.batch_size] * y_pred.flatten(), y.flatten())
            
            total_loss += loss
            loss.backward()
            optimizer.step()
        if i % 20 == 0:
            print('loss in epoch {}: {}'.format(i, total_loss / len(Y_train)))

        # calculate validation loss
        model.eval()

        total_val_loss = 0
        for idx in range(0, len(Y_val), args.batch_size):
            x = X_val[idx:idx+args.batch_size, :]
            y = Y_val[idx:idx+args.batch_size]
            val_loss = criterion(model(x).flatten(), y.flatten())
            total_val_loss += val_loss
        if i % 20 == 0:
            print('\tvalidation loss epoch {}: {:.3f}'.format(i, val_loss / len(Y_val)))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = i
            torch.save(model.state_dict(), f'{args.results_folder}/best_model.pth')
        
        scheduler.step()

    print('best epoch: {}'.format(best_epoch))

    model.load_state_dict(torch.load(f'{args.results_folder}/best_model.pth'))
    model.eval()
    return model

def test_deep_learning(model, X_test, use_cuda=False):
    if use_cuda:
        X_test = torch.FloatTensor(X_test).cuda()
    else:
        model = model.cpu()
        X_test = torch.FloatTensor(X_test)
    return [1 if y > 0 else -1 for y in model(X_test).flatten().tolist()]

def main():
    parser = argparse.ArgumentParser()

    # which model to pick
    parser.add_argument('--model_name', type=str, default=DL)
    # general experimental params
    parser.add_argument('--num_training_samples', type=int, default=30000)
    parser.add_argument('--num_testing_samples', type=int, default=5000)
    parser.add_argument('--dataset_name', type=str, default="3robots_25obstacles_seed0_")
    parser.add_argument('--forward_kinematics_kernel', action='store_true')
    # fastron-specific params
    parser.add_argument('--g', type=int, default=5)
    parser.add_argument('--beta', type=int, default=1)
    parser.add_argument('--maxUpdates', type=int, default=100000)
    parser.add_argument('--maxSupportPoints', type=int, default=20000)
    # dl-specific params
    parser.add_argument('--bias', type=float, default=1)
    parser.add_argument('--num_freq', type=int, default=8)
    parser.add_argument('--sigma', type=float, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--train_percent', type=float, default=0.95)
    parser.add_argument('--epochs', type=int, default=50)
    # where to log output
    parser.add_argument('--results_folder', type=str, default='comparison_results')

    args = parser.parse_args()

    os.makedirs(args.results_folder, exist_ok=True)

    run_model(args)

if __name__ == '__main__':
    main()
