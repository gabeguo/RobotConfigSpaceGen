# DEEP LEARNING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import time

class MyNN():
    def fit(self, X_train, Y_train, learning_rate=0.01, DOF=21, BATCH_SIZE=64, EPOCHS=10):
        self.DOF = DOF
        for data_point in X_train:
            while len(data_point) < self.DOF:
                data_point.append(0)

        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        print('device: {}'.format(self.device))

        model = nn.Sequential(
            nn.Linear(self.DOF, 4 * self.DOF),
            nn.ReLU(),
            nn.Linear(4 * self.DOF, self.DOF),
            nn.ReLU(),
            nn.Linear(self.DOF, 2)
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        Y_train = [[int(i == int(y)) for i in range(2)] for y in Y_train]

        tensor_x = torch.Tensor(X_train)
        tensor_y = torch.Tensor(Y_train)
        my_dataset = TensorDataset(tensor_x, tensor_y)
        my_dataloader = DataLoader(my_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)

        start = time.time()

        for i in range(EPOCHS):
            total_loss = 0
            # train
            for inputs, labels in my_dataloader:
                x = inputs.to(self.device)
                y = labels.to(self.device)

                optimizer.zero_grad()

                y_pred = model(x)

                loss = criterion(y_pred, y)
                total_loss += loss

                loss.backward()
                optimizer.step()

        self.model = model

        end = time.time()

        time_elapsed = round(end - start, 3)

        print('DL train time:', time_elapsed)

        return

    def predict(self, X_test):
        for data_point in X_test:
            while len(data_point) < self.DOF:
                data_point.append(0)
        return [0 if y[0] > y[1] else 1 for y in self.model(torch.tensor(X_test, device=self.device, dtype=torch.float32)).tolist()]
