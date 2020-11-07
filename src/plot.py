import numpy as np
import matplotlib.pyplot as pp

filename = "training_data.txt"

data = []
with open(filename) as f:
	for line in f.readlines():
		line = list(map(float, line.strip().split(" ")))
		data.append(line)

data = np.matrix(data)

for i in range(6):
	pp.plot(data[:, i], label=f"Lidar#{i}")

pp.plot(data[:, -1]*5, label="car_angle")
pp.legend()
pp.grid(True)
pp.show()
