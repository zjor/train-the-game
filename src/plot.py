import numpy as np
import matplotlib.pyplot as pp

data = []
with open("data.txt") as f:
	for line in f.readlines():
		line = list(map(float, line.strip().split(" ")))
		data.append(line)

data = np.matrix(data)

pp.plot(data[:, 0], label="road")
pp.plot(data[:, 1], label="car_x")
pp.plot(data[:, 1] - data[:, 0], label="diff")
pp.plot(data[:, 2]*5, label="car_angle")
pp.legend()
pp.grid(True)
pp.show()