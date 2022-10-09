# DEEP LEARNING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import time

class MyNN():
    def fit(self, X_train, Y_train, learning_rate=0.1, DOF=35, BATCH_SIZE=64, EPOCHS=100):
        self.DOF = DOF
        for data_point in X_train:
            while len(data_point) < self.DOF:
                data_point.append(0)

        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        print('device: {}'.format(self.device))

        model = nn.Sequential(
              nn.Linear(self.DOF, 100),
              nn.Tanh(),
              nn.Linear(100, 150),
              nn.Tanh(),
              nn.Linear(150, 200),
              nn.Tanh(),
              nn.Linear(200, 250),
              nn.Tanh(),
              nn.Linear(250, 1)
            ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.05)

        tensor_x = torch.Tensor(X_train)
        tensor_y = torch.Tensor(Y_train)
        my_dataset = TensorDataset(tensor_x, tensor_y)
        my_dataloader = DataLoader(my_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)

        start = time.time()

        for i in range(EPOCHS):
            total_loss = 0
            # train
            for inputs, labels in my_dataloader:
                x = inputs.to(self.device)
                y = labels.to(self.device)

                optimizer.zero_grad()
            #for t in range(0, len(Y_train), BATCH_SIZE):
                #curr_batch_size = min(BATCH_SIZE, len(Y_train) - t)
                #x = torch.tensor(X_train[t:t+curr_batch_size], device=self.device, dtype=torch.float32)
                #y = torch.tensor([Y_train[t:t+curr_batch_size]], device=self.device, dtype=torch.float32).reshape((curr_batch_size, 1))

                y_pred = model(x).reshape(-1)

                #print('y_pred size', y_pred.shape, '; y size', y.shape)
                
                loss = criterion(y_pred, y)
                total_loss += loss
               
                #model.zero_grad()
                
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
        return [1 if y[0] >= 0.5 else 0 for y in self.model(torch.tensor(X_test, device=self.device, dtype=torch.float32)).tolist()]
