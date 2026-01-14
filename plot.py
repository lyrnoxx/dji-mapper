import matplotlib.pyplot as plt
import numpy as np

traj_file = "pose1.txt"

xs, ys = [], []

with open(traj_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        xs.append(float(parts[1]))
        ys.append(float(parts[2]))

xs = np.array(xs)
ys = np.array(ys)

plt.figure(figsize=(6, 6))
plt.plot(xs, ys, '-k', linewidth=1)
plt.quiver(xs[:-1], ys[:-1],
           xs[1:] - xs[:-1],
           ys[1:] - ys[:-1],
           scale_units='xy', scale=1, color='r', width=0.003)

plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Drone Trajectory with Motion Direction")
plt.axis("equal")
plt.grid(True)
plt.show()
