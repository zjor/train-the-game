import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

import multiprocessing


class Model(nn.Module):
    def __init__(self, in_features=12, hidden=[24, 24], out_features=3):
        super().__init__()
        layer_sizes = [in_features] + hidden
        layers = []

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(layer_sizes[-1], out_features))
        self.layers = nn.Sequential(*layers)

        
    def forward(self, x):
        return self.layers(x)


class TorchCortex:
    def __init__(self):
        self.model = Model(in_features=24, hidden=[48, 48, 48])
        self.initialized = False


    def balance_classes(self, df):
        min_count = df["command"].value_counts().min()
        straight = df[df["command"] == 1].sample(n=min_count)
        left = df[df["command"] == 0].sample(n=min_count)
        right = df[df["command"] == 2].sample(n=min_count)        
        return pd.concat([left, straight, right])


    def load_data(self, filename):
        num_lidars = 24
        lidar_cols = [f"l{i}" for i in range(num_lidars)]
        cols = lidar_cols + ["command"]

        df = pd.read_csv(filename, header=None, names=cols, sep="\\s+")
        return self.balance_classes(df)


    def validate(self, criterion, X_test, y_test):
        with torch.no_grad():
            y_val = self.model.forward(X_test)
            loss = criterion(y_val, y_test)
        print(f"\nValidation loss: {loss:.8f}")  

        correct = 0
        with torch.no_grad():
            for i,data in enumerate(X_test):
                y_val = self.model.forward(data)
                # print(f'{i+1:2}. {str(y_val):38}  {y_test[i]}')
                if y_val.argmax().item() == y_test[i]:
                    correct += 1
        print(f'\n{correct} out of {len(y_test)} = {100*correct/len(y_test):.2f}% correct')              


    def train(self, df):

        X = df.drop('command',axis=1).values
        y = df['command'].values

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=33)

        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)

        trainloader = DataLoader(X_train, batch_size=60, shuffle=True)
        testloader = DataLoader(X_test, batch_size=60, shuffle=False)

        model = self.model
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        epochs = 1000
        losses = []

        for i in range(epochs):
            y_pred = model.forward(X_train)
            loss = criterion(y_pred, y_train)
            losses.append(loss)
            
            if i%100 == 1:
                print(f'epoch: {i:2}  loss: {loss.item():10.8f}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.validate(criterion, X_test, y_test)
        self.initialized = True


    def save(self, filename):
        torch.save(self.model.state_dict(), filename)


    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))
        self.initialized = True

    
    def load_state(self, state):
        self.model.load_state_dict(state)
        self.initialized = True        


    def predict(self, data):
        if not self.initialized:
            raise Exception("Model is not initialized")
        t = torch.tensor(data, dtype=torch.float)
        y_val = self.model.forward(t)
        predicted_class = y_val.argmax()
        return predicted_class.numpy()


    def predict_raw(self, data):
        if not self.initialized:
            raise Exception("Model is not initialized")
        t = torch.tensor(data, dtype=torch.float)
        y_val = self.model.forward(t)
        return y_val.detach().numpy()


class CortexWorker(multiprocessing.Process):
    def __init__(self, queue, data):
        super(CortexWorker, self).__init__()
        self.queue = queue
        self.data = data
        self.cortex = TorchCortex()


    def run(self):
        num_lidars = self.data.shape[1] - 1
        lidar_cols = [f"l{i}" for i in range(num_lidars)]
        cols = lidar_cols + ["command"]

        df = pd.DataFrame(self.data, columns=cols)
        df = self.cortex.balance_classes(df)
        self.cortex.train(df)
        self.queue.put(self.cortex.model.state_dict())




def load_as_numpy(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = list(map(float, line.strip().split(" ")))
            data.append(line)
    return np.matrix(data)



if __name__ == "__main__":
    import time
    torch.manual_seed(42)
    input_filename = "training_data.txt"
    # cortex = TorchCortex()
    # model_filename = "model.pt"
    # dataset = cortex.load_data("training_data.txt")
    # cortex.train(dataset)
    # cortex.save(model_filename)

    # cortex.load(model_filename)    



    q = multiprocessing.Queue()
    data = load_as_numpy(input_filename)
    print("Data loaded")
    w = CortexWorker(q, data)
    w.start()
    print("Trainging...", end="")
    while w.is_alive():
        time.sleep(1)
        print(".", end="")
    
    state = q.get()
    
    cortex = TorchCortex()
    cortex.load_state(state)
    row = "1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 0.6041666666666666 0.5166666666666667 0.4625 0.4375 0.4083333333333333 0.3875 0.36666666666666664 0.3458333333333333 0.325 0.30416666666666664 0.2791666666666667 0.2708333333333333 0.25416666666666665"
    pred = cortex.predict(list(map(float, row.split())))
    print(pred)



    # w.join()

    


    


