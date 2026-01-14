import numpy as np
from scipy.spatial.transform import Rotation as R

# Test different pitch interpretations
gimbal_pitch_raw = -89.9

print("=" * 70)
print("TESTING PITCH ANGLE INTERPRETATIONS")
print("=" * 70)
print(f"Raw gimbal pitch from EXIF: {gimbal_pitch_raw}°")
print()

# Test conversions
pitch_tests = [
    ("Raw (no change)", gimbal_pitch_raw),
    ("Negated", -gimbal_pitch_raw),
    ("90° offset", gimbal_pitch_raw + 90),
    ("90° - pitch", 90 - gimbal_pitch_raw),
    ("-90° - pitch", -90 - gimbal_pitch_raw),
]

for name, pitch in pitch_tests:
    # Simple rotation around Y axis (pitch only, no roll/yaw)
    rot = R.from_euler('y', pitch, degrees=True)
    
    # Camera forward direction (+Z)
    cam_forward = rot.apply([0, 0, 1])
    
    # In NED: Z=Down, so we want camera +Z to have large positive Z component
    down_component = cam_forward[2]
    
    print(f"{name:20s}: pitch={pitch:7.1f}° → Camera +Z = [{cam_forward[0]:6.3f}, {cam_forward[1]:6.3f}, {cam_forward[2]:6.3f}]")
    print(f"                         Down component: {down_component:6.3f}")
    
    if down_component > 0.99:
        print("                         ✓ POINTS DOWN!")
    elif down_component < -0.99:
        print("                         Points up")
    else:
        print(f"                         Tilted {np.degrees(np.arcsin(down_component)):.1f}° from horizontal")
    print()

print("=" * 70)
print("FULL TEST WITH BODY + GIMBAL")
print("=" * 70)

gimbal_roll = 0.0
gimbal_yaw = -87.4
flight_roll = -1.1
flight_pitch = -2.4
flight_yaw = -88.4

for name, pitch_conversion in pitch_tests:
    gimbal_pitch = pitch_conversion
    
    R_body = R.from_euler('xyz', [flight_roll, flight_pitch, flight_yaw], degrees=True)
    R_gimbal = R.from_euler('xyz', [gimbal_roll, gimbal_pitch, gimbal_yaw], degrees=True)
    R_camera = R_body * R_gimbal
    
    cam_z = R_camera.apply([0, 0, 1])
    down_comp = cam_z[2]
    
    print(f"{name:20s}: Camera +Z down component = {down_comp:6.3f}", end="")
    if down_comp > 0.9:
        print(" ✓ CORRECT!")
    else:
        print()

print("\n" + "=" * 70)
print("CHECKING IF DJI USES TAIT-BRYAN vs PROPER EULER")
print("=" * 70)
print("\nDJI might use aircraft convention where:")
print("  Pitch = 0° means level (horizontal)")
print("  Pitch = -90° means nose down (pointing down)")
print("\nBut in NED frame with Z=Down:")
print("  A camera level (horizontal) points in XY plane")
print("  A camera pitched -90° should point in +Z direction (down)")