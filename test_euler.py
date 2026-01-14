import numpy as np
from scipy.spatial.transform import Rotation as R

# From your metadata
gimbal_roll = 0.0
gimbal_pitch = -89.9  # Nearly straight down
gimbal_yaw = -87.4

flight_roll = -1.1
flight_pitch = -2.4
flight_yaw = -88.4

print("Testing different Euler conventions for gimbal orientation")
print("=" * 70)
print(f"Gimbal: Roll={gimbal_roll:.1f}°, Pitch={gimbal_pitch:.1f}°, Yaw={gimbal_yaw:.1f}°")
print(f"Flight: Roll={flight_roll:.1f}°, Pitch={flight_pitch:.1f}°, Yaw={flight_yaw:.1f}°")
print()

# Test different conventions
conventions = ['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx']
angle_orders = [
    ('roll', 'pitch', 'yaw'),
    ('roll', 'yaw', 'pitch'),
    ('pitch', 'roll', 'yaw'),
    ('pitch', 'yaw', 'roll'),
    ('yaw', 'roll', 'pitch'),
    ('yaw', 'pitch', 'roll'),
]

print("Testing GIMBAL ONLY (should point nearly straight down):")
print("-" * 70)

for conv, order in zip(conventions, angle_orders):
    # Map order to values
    angle_map = {
        'roll': gimbal_roll,
        'pitch': gimbal_pitch,
        'yaw': gimbal_yaw
    }
    angles = [angle_map[o] for o in order]
    
    # Create rotation
    rot = R.from_euler(conv, angles, degrees=True)
    
    # Camera +Z axis (forward) in world frame
    cam_z = rot.apply([0, 0, 1])
    
    # Calculate angle from vertical (down is -Z in ENU)
    angle_from_down = np.degrees(np.arccos(-cam_z[2]))
    
    print(f"{conv.upper()} ({order[0][0]}{order[1][0]}{order[2][0]}): "
          f"Camera +Z = [{cam_z[0]:6.3f}, {cam_z[1]:6.3f}, {cam_z[2]:6.3f}] "
          f"  Angle from down: {angle_from_down:5.1f}°")

print("\n" + "=" * 70)
print("Looking for angle from down close to 0° (camera pointing down)")
print("=" * 70)

# Now test with body rotation included
print("\nTesting BODY + GIMBAL:")
print("-" * 70)

best_angle = 180
best_conv = None

for conv in conventions:
    # Body rotation
    R_body = R.from_euler(conv, [flight_yaw, flight_pitch, flight_roll], degrees=True)
    # Gimbal rotation  
    R_gimbal = R.from_euler(conv, [gimbal_yaw, gimbal_pitch, gimbal_roll], degrees=True)
    # Combined
    R_combined = R_body * R_gimbal
    
    # Camera direction
    cam_z = R_combined.apply([0, 0, 1])
    angle_from_down = np.degrees(np.arccos(np.clip(-cam_z[2], -1, 1)))
    
    print(f"{conv.upper()}: Camera +Z = [{cam_z[0]:6.3f}, {cam_z[1]:6.3f}, {cam_z[2]:6.3f}] "
          f"  Angle: {angle_from_down:5.1f}°")
    
    if angle_from_down < best_angle:
        best_angle = angle_from_down
        best_conv = conv

print(f"\n✓ Best convention: {best_conv.upper()} (angle from down: {best_angle:.1f}°)")

# Test the actual coordinate transformation needed
print("\n" + "=" * 70)
print("COORDINATE FRAME TRANSFORMATIONS")
print("=" * 70)

# DJI NED: X=North, Y=East, Z=Down
# Standard ENU: X=East, Y=North, Z=Up

print("\nDJI NED frame:")
print("  X = North")
print("  Y = East") 
print("  Z = Down")
print("  Pitch=-90° = pointing down in -Z direction")

print("\nStandard ENU frame:")
print("  X = East")
print("  Y = North")
print("  Z = Up")
print("  Camera should look in -Z direction (down)")

print("\nNED to ENU transformation:")
print("  ENU_X = NED_Y  (East)")
print("  ENU_Y = NED_X  (North)")
print("  ENU_Z = -NED_Z (Up)")