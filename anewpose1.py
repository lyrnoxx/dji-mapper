import os, json, subprocess, numpy as np
from pyproj import Proj
from scipy.spatial.transform import Rotation as R

IMAGE_FOLDER = 'images-true'
OUTPUT_FILE = 'pose1.txt'

def get_meta(path):
    """Extract metadata from DJI image - we only need gimbal angles and GPS."""
    cmd = ['exiftool', '-n', '-j', 
           '-GPSLatitude', '-GPSLongitude', '-RelativeAltitude',
           '-GimbalRollDegree', '-GimbalPitchDegree', '-GimbalYawDegree',
           path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)[0]
    
    # Convert string values to float if needed
    for key in ['GPSLatitude', 'GPSLongitude', 'RelativeAltitude',
                'GimbalRollDegree', 'GimbalPitchDegree', 'GimbalYawDegree']:
        if key in data and isinstance(data[key], str):
            data[key] = float(data[key])
    
    return data

def compute_camera_pose(metadata):
    """
    Compute camera pose from DJI gimbal angles.
    
    CRITICAL DISCOVERY: DJI gimbal angles are ABSOLUTE (world frame),
    not relative to the drone body. The gimbal stabilizes the camera
    regardless of drone body orientation.
    
    DJI pitch convention:
    - Pitch 0° = horizontal (level)
    - Pitch -90° = pointing straight down (nadir)
    
    We add 90° to convert to standard rotation where pitch 0° = forward.
    """
    # Use gimbal angles directly - they're already in world frame!
    # Ignore FlightRollDegree, FlightPitchDegree, FlightYawDegree
    
    gimbal_pitch_corrected = metadata['GimbalPitchDegree'] + 90.0
    
    R_camera = R.from_euler('xyz', [
        metadata['GimbalRollDegree'],
        gimbal_pitch_corrected,
        metadata['GimbalYawDegree']
    ], degrees=True)
    
    return R_camera

# Process images
images = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith('.jpg')])

if not images:
    raise ValueError(f"No JPG images found in {IMAGE_FOLDER}")

# Setup UTM projection using first image coordinates
first = get_meta(os.path.join(IMAGE_FOLDER, images[0]))
utm_zone = int((first['GPSLongitude'] + 180) / 6) + 1
proj = Proj(proj='utm', zone=utm_zone, ellps='WGS84')

# Get origin coordinates
ox, oy = proj(first['GPSLongitude'], first['GPSLatitude'])

print(f"Processing {len(images)} images...")
print(f"UTM Zone: {utm_zone}")
print(f"Origin: ({ox:.2f}, {oy:.2f})")

with open(OUTPUT_FILE, 'w') as f:
    for i, name in enumerate(images):
        m = get_meta(os.path.join(IMAGE_FOLDER, name))
        
        # Convert GPS to UTM and make relative to origin
        tx, ty = proj(m['GPSLongitude'], m['GPSLatitude'])
        tx_rel = tx - ox
        ty_rel = ty - oy
        
        # In NED frame, altitude is negative (Z points down)
        # DJI RelativeAltitude is positive upward, so negate it
        tz = -m['RelativeAltitude']
        
        # Compute camera rotation
        R_cam = compute_camera_pose(m)
        q = R_cam.as_quat()  # Returns [x, y, z, w]
        
        # Write pose: name x y z qx qy qz qw
        f.write(f"{os.path.splitext(name)[0]} "
                f"{tx_rel:.6f} {ty_rel:.6f} {tz:.6f} "
                f"{q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f}\n")
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(images)} images")

print(f"\nPose file saved to: {OUTPUT_FILE}")
print(f"Total images processed: {len(images)}")