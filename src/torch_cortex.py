import math
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.autograd import Variable


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(2, 1) / math.sqrt(2))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, xb):
        return xb @ self.weights + self.bias


class TorchCortex:
    def __init__(self):
        self.model = LinearRegression()
        self.initialized = False


    def load_data(self, filename):
        data = []
        with open(filename) as f:
            for line in f.readlines():
                line = list(map(float, line.strip().split(" ")))
                data.append(line)
        data = np.matrix(data)
        X = data[:, 0:2]
        y = data[:, 2]
        return torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.float)


    def train(self, X, y):
        model = self.model
        loss_func = F.mse_loss

        n = X.size()[0]
        lr = 1e-7
        bs = 50

        for epoch in range(2000):
            for i in range((n - 1) // bs + 1):
                start_i = i * bs
                end_i = start_i + bs
                xb = X[start_i:end_i]
                yb = y[start_i:end_i]
                pred = model(xb)
                loss = loss_func(pred, yb)
                loss.backward()
                with torch.no_grad():
                    for p in model.parameters():
                        p -= p.grad * lr
                    model.zero_grad()
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
    # X, y = cortex.load_data("../jupiter/data/data.txt")
    # cortex.train(X, y)
    # cortex.save("torch_model.dump")

    cortex.load("torch_model.dump")
    print(cortex.predict([200, 200]))


