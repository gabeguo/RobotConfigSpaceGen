# DEEP LEARNING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import time

class WideNNReLU(nn.Module):
    def __init__(self, DOF):
        super(WideNNReLU, self).__init__()
        self.DOF = DOF
        self.model = nn.Sequential(
            nn.Linear(self.DOF, 4 * self.DOF),
            nn.ReLU(),
            nn.Linear(4 * self.DOF, self.DOF),
            nn.ReLU(),
            nn.Linear(self.DOF, 2)
        )
        return
    def forward(self, x):
        return self.model(x)

class WideNNTanh(nn.Module):
    def __init__(self, DOF):
        super(WideNNTanh, self).__init__()
        self.DOF = DOF
        self.model = nn.Sequential(
            nn.Linear(self.DOF, 4 * self.DOF),
            nn.Tanh(),
            nn.Linear(4 * self.DOF, self.DOF),
            nn.Tanh(),
            nn.Linear(self.DOF, 2)
        )
        return
    def forward(self, x):
        return self.model(x)

class DeepNNReLU(nn.Module):
    def __init__(self, DOF):
        super(DeepNNReLU, self).__init__()
        self.DOF = DOF
        self.model = nn.Sequential(
            nn.Linear(self.DOF, 2 * self.DOF),
            nn.ReLU(),
            nn.Linear(2 * self.DOF, 2 * self.DOF),
            nn.ReLU(),
            nn.Linear(2 * self.DOF, 2 * self.DOF),
            nn.ReLU(),
            nn.Linear(2 * self.DOF, self.DOF),
            nn.ReLU(),
            nn.Linear(self.DOF, 2)
        )
        return
    def forward(self, x):
        return self.model(x)

class DeepNNTanh(nn.Module):
    def __init__(self, DOF):
        super(DeepNNTanh, self).__init__()
        self.DOF = DOF
        self.model = nn.Sequential(
            nn.Linear(self.DOF, 2 * self.DOF),
            nn.Tanh(),
            nn.Linear(2 * self.DOF, 2 * self.DOF),
            nn.Tanh(),
            nn.Linear(2 * self.DOF, 2 * self.DOF),
            nn.Tanh(),
            nn.Linear(2 * self.DOF, self.DOF),
            nn.Tanh(),
            nn.Linear(self.DOF, 2)
        )
        return
    def forward(self, x):
        return self.model(x)

class LightweightNNReLU(nn.Module):
    def __init__(self, DOF):
        super(LightweightNNReLU, self).__init__()
        self.DOF = DOF
        self.model = nn.Sequential(
            nn.Linear(self.DOF, self.DOF),
            nn.ReLU(),
            nn.Linear(self.DOF, 2)
        )
        return
    def forward(self, x):
        return self.model(x)

class LightweightNNTanh(nn.Module):
    def __init__(self, DOF):
        super(LightweightNNTanh, self).__init__()
        self.DOF = DOF
        self.model = nn.Sequential(
            nn.Linear(self.DOF, self.DOF),
            nn.Tanh(),
            nn.Linear(self.DOF, 2)
        )
        return
    def forward(self, x):
        return self.model(x)

class MyNN():
    def __init__(self, model_type=WideNNReLU, DOF=21, learning_rate=0.01, batch_size=64, epochs=10):
        self.model_type = model_type
        self.DOF = DOF
        self.model = model_type(DOF)

        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        return

    def fit(self, X_train, Y_train):
        Y_train = [[int(i == int(y)) for i in range(2)] for y in Y_train]

        tensor_x = torch.Tensor(X_train)
        tensor_y = torch.Tensor(Y_train)
        my_dataset = TensorDataset(tensor_x, tensor_y)
        my_dataloader = DataLoader(my_dataset, batch_size=self.batch_size, num_workers=0, shuffle=True)

        start = time.time()

        for i in range(self.epochs):
            total_loss = 0
            # train
            for inputs, labels in my_dataloader:
                x = inputs.to(self.device)
                y = labels.to(self.device)

                self.optimizer.zero_grad()

                y_pred = self.model(x)

                loss = self.criterion(y_pred, y)
                total_loss += loss

                loss.backward()
                self.optimizer.step()

        end = time.time()

        time_elapsed = round(end - start, 3)

        print('NN train time:', time_elapsed)

        return

    def predict(self, X_test):
        return [0 if y[0] > y[1] else 1 for y in self.model(torch.tensor(X_test, device=self.device, dtype=torch.float32)).tolist()]
