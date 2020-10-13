import math
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 2),
            nn.ReLU(),
            nn.Linear(2, 1)
            )

    def forward(self, x):
        return self.model(x)


class TorchCortex:
    def __init__(self):
        self.model = Net()
        self.initialized = False


    def load_data(self, filename):
        data = []
        with open(filename) as f:
            for line in f.readlines():
                line = list(map(float, line.strip().split(" ")))
                data.append(line)
        data = np.matrix(data)
        X = torch.tensor(data[:, 0:3], dtype=torch.float)
        y = torch.tensor(data[:, 3], dtype=torch.float)
        return TensorDataset(X, y)


    def train(self, dataset):
        model = self.model
        loss_func = F.mse_loss
        
        lr = 1e-7
        bs = 50

        opt = optim.SGD(model.parameters(), lr=lr)
        loader = DataLoader(dataset, batch_size=bs, shuffle=False)

        for epoch in range(2000):
            for xb, yb in loader:
                pred = model(xb)
                loss = loss_func(pred, yb)
                loss.backward()
                opt.step()
                opt.zero_grad()
            if epoch % 50 == 0:
                print(loss)
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
        return self.model(t).detach().numpy()[0]


if __name__ == "__main__":
    cortex = TorchCortex()
    # dataset = cortex.load_data("../jupiter/data/data.txt")
    dataset = cortex.load_data("training_data.txt")
    cortex.train(dataset)
    cortex.save("torch_model.dump")

    # cortex.load("torch_model.dump")
    # print(cortex.predict([200, 200]))


