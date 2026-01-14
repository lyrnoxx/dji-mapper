import numpy as np
from scipy.spatial.transform import Rotation as R

def quat_to_rot_matrix(q):
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])

# Read first pose from pose file
with open('pose1.txt', 'r') as f:
    line = f.readline().strip()
    parts = line.split()
    name = parts[0]
    tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
    qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])

print("=" * 60)
print("POSE DIAGNOSTIC")
print("=" * 60)
print(f"\nImage: {name}")
print(f"Position (x, y, z): ({tx:.2f}, {ty:.2f}, {tz:.2f})")
print(f"Quaternion (x, y, z, w): ({qx:.6f}, {qy:.6f}, {qz:.6f}, {qw:.6f})")

# Convert to rotation matrix
R_mat = quat_to_rot_matrix([qx, qy, qz, qw])

print("\nRotation Matrix:")
print(R_mat)

# Camera pointing direction (+Z axis in camera frame = forward)
# In NED frame, down is +Z direction
camera_z_axis = np.array([0, 0, 1])
camera_direction_world = R_mat @ camera_z_axis

print(f"\nCamera +Z axis in world frame: {camera_direction_world}")
print(f"  X (North) component: {camera_direction_world[0]:.4f}")
print(f"  Y (East) component:  {camera_direction_world[1]:.4f}")
print(f"  Z (Down) component:  {camera_direction_world[2]:.4f}")

# Check if pointing down (in NED, down is +Z)
if camera_direction_world[2] > 0.9:
    print("\n✓ Camera is pointing DOWNWARD (correct for nadir in NED frame)")
elif camera_direction_world[2] < -0.9:
    print("\n✗ Camera is pointing UPWARD (WRONG!)")
else:
    angle_from_down = np.degrees(np.arccos(camera_direction_world[2]))
    print(f"\n✗ Camera is NOT pointing down (angle from down: {angle_from_down:.1f}°)")

# Convert to Euler angles for interpretation
rot = R.from_quat([qx, qy, qz, qw])

# Try different conventions
print("\n" + "=" * 60)
print("EULER ANGLE INTERPRETATION")
print("=" * 60)

for convention in ['xyz', 'zyx', 'zxy']:
    euler = rot.as_euler(convention, degrees=True)
    print(f"\n{convention.upper()} convention: ({euler[0]:.1f}°, {euler[1]:.1f}°, {euler[2]:.1f}°)")

# Check what coordinate system we're in
print("\n" + "=" * 60)
print("COORDINATE FRAME CHECK")
print("=" * 60)

# World frame axes in camera frame
world_x_in_cam = R_mat.T @ np.array([1, 0, 0])
world_y_in_cam = R_mat.T @ np.array([0, 1, 0])
world_z_in_cam = R_mat.T @ np.array([0, 0, 1])

print("\nWorld axes as seen from camera:")
print(f"  World X in camera frame: {world_x_in_cam}")
print(f"  World Y in camera frame: {world_y_in_cam}")
print(f"  World Z in camera frame: {world_z_in_cam}")

# Expected: For nadir camera at altitude looking down
# - World Z should align with camera -Z (pointing down)
# - If Z_world is in camera's -Z direction, that's correct

print("\n" + "=" * 60)
print("RECOMMENDATION")
print("=" * 60)

if camera_direction_world[2] < -0.9:
    print("\nThe camera is pointing UP instead of DOWN in NED frame.")
    print("In NED, Z points down, so camera +Z should be positive.")
elif camera_direction_world[2] < 0.5:
    print("\nThe camera is pointing too horizontal or wrong direction.")
    print(f"Camera +Z components: North={camera_direction_world[0]:.3f}, "
          f"East={camera_direction_world[1]:.3f}, Down={camera_direction_world[2]:.3f}")
    print("Check pitch angle conversion and Euler convention.")
else:
    print("\nCamera orientation looks correct!")
    print("The camera is pointing downward in NED frame (Z > 0.9)")