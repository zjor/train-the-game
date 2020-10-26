import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


class Model(nn.Module):
    def __init__(self, in_features=6, h1=15, h2=15, h3=15, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features,h1)    # input layer
        self.fc2 = nn.Linear(h1, h2)            # hidden layer
        self.fc3 = nn.Linear(h2, h3)            # hidden layer
        self.out = nn.Linear(h3, out_features)  # output layer
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return x


class TorchCortex:
    def __init__(self):
        self.model = Model()
        self.initialized = False


    def balance_classes(self, df):
        min_count = df["command"].value_counts().min()
        straight = df[df["command"] == 1].sample(n=min_count)
        left = df[df["command"] == 0].sample(n=min_count)
        right = df[df["command"] == 2].sample(n=min_count)        
        return pd.concat([left, straight, right])


    def load_data(self, filename):
        df = pd.read_csv(filename, header=None, names=["l1", "l2", "l3", "l4", "l5", "l6", "command"], sep="\\s+")
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
        
        epochs = 5000
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



if __name__ == "__main__":
    torch.manual_seed(42)
    cortex = TorchCortex()
    model_filename = "model.pt"
    dataset = cortex.load_data("training_data.txt")
    cortex.train(dataset)
    cortex.save(model_filename)

    row = "182 253 359 113 68 70"
    cortex.load(model_filename)    
    pred = cortex.predict(list(map(float, row.split())))
    print(pred)

    


