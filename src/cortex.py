import numpy as np
import pandas as pd
import joblib
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class Cortex:
    def __init__(self):
        self.lm = linear_model.LinearRegression()
        self.model = None
        self.initialized = False


    def train(self, filename):
        data = []
        with open(filename) as f:
            for line in f.readlines():
                line = list(map(float, line.strip().split(" ")))
                data.append(line)
        data = np.matrix(data)
        X = pd.DataFrame(data[:, 0:2])
        y = pd.DataFrame(data[:, 2])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
        
        self.model = self.lm.fit(X_train,y_train)
        self.initialized = True

        predictions = self.model.predict(X_test)
        print(f"MSE: {mean_squared_error(y_test, predictions)}")

    
    def save(self, filename):
        joblib.dump(self.model, filename)

    
    def load(self, filename):
        self.model = joblib.load(filename)
        self.initialized = True

    def predict(self, data):
        if not self.initialized:
            raise Error("Model is not initialized")
        return self.model.predict([data])[0][0]


if __name__ == "__main__":
    training_filename = "training_data.txt"
    # cortex = Cortex()
    # cortex.train(training_filename)
    # cortex.save("model.dump")

    cortex = Cortex()
    cortex.load("model.dump")
    print(cortex.predict([200, 180]))





