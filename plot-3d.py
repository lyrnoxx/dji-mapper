import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

traj_file = "pose1.txt"

x, y, z = [], [], []

with open(traj_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        x.append(float(parts[1]))
        y.append(float(parts[2]))
        z.append(float(parts[3]))

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot(x, y, z, '-o', markersize=3)

ax.set_xlabel("X (meters, East)")
ax.set_ylabel("Y (meters, North)")
ax.set_zlabel("Z (meters, Altitude)")

ax.set_title("Drone 3D Trajectory")

plt.tight_layout()
plt.show()
