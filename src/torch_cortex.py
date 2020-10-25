import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader


class Model(nn.Module):
    def __init__(self, in_features=6, h1=12, h2=13, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features,h1)    # input layer
        self.fc2 = nn.Linear(h1, h2)            # hidden layer
        self.out = nn.Linear(h2, out_features)  # output layer
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


class TorchCortex:
    def __init__(self):
        self.model = Model()
        self.initialized = False


    def load_data(self, filename):
        data = []
        with open(filename) as f:
            for line in f.readlines():
                line = list(map(float, line.strip().split(" ")))
                data.append(line)
        data = np.matrix(data)
        input_size = 6
        X = torch.FloatTensor(data[:, :input_size])
        y = torch.LongTensor(data[:, input_size])
        return TensorDataset(X, y)


    def train(self, dataset):
        model = self.model
        loss_func = nn.CrossEntropyLoss()
        
        lr = 1e-4
        bs = 50

        opt = optim.Adam(model.parameters(), lr=lr)
        loader = DataLoader(dataset, batch_size=bs, shuffle=False)

        for epoch in range(150):
            for xb, yb in loader:
                pred = model(xb)                
                loss = loss_func(pred, yb)
                loss.backward()
                opt.step()
                opt.zero_grad()
            if epoch % 50 == 0:
                print(f"Epoch: {epoch} Loss: {loss}")
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


if __name__ == "__main__":
    torch.manual_seed(42)
    cortex = TorchCortex()
    model_filename = "model.pt"
    # dataset = cortex.load_data("../jupiter/data/data.txt")
    # dataset = cortex.load_data("training_data.txt")
    # cortex.train(dataset)
    # cortex.save(model_filename)

    row = "182 253 359 113 68 70"
    cortex.load(model_filename)    
    pred = cortex.predict(list(map(float, row.split())))
    print(pred)

    


