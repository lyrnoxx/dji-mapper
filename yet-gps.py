import os
import json
import subprocess
from pyproj import Proj
from scipy.spatial.transform import Rotation as R

# --- SETTINGS ---
IMAGE_FOLDER = 'images-true'  # As per your folder name
OUTPUT_FILE = 'new-new-trajectory.txt'
# Set to True to start your trajectory at (0, 0, 0) for the first image
USE_LOCAL_ORIGIN = True 

def get_dji_metadata(image_path):
    """Extracts specific DJI XMP tags as decimal/numeric values."""
    # -n: Numeric output (decimal instead of degrees)
    # -j: JSON output
    cmd = [
        'exiftool', '-n', '-j',
        '-GPSLatitude', '-GPSLongitude', 
        '-RelativeAltitude',
        '-GimbalRollDegree', '-GimbalPitchDegree', '-GimbalYawDegree',
        image_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # The JSON is a list containing one dictionary
    data = json.loads(result.stdout)[0]
    
    return {
        'lat': float(data.get('GPSLatitude', 0)),
        'lon': float(data.get('GPSLongitude', 0)),
        'alt': float(data.get('RelativeAltitude', 0)),
        'roll': float(data.get('GimbalRollDegree', 0)),
        'pitch': float(data.get('GimbalPitchDegree', 0)),
        'yaw': float(data.get('GimbalYawDegree', 0))
    }

def create_trajectory():
    # Filter for JPG files and sort them alphabetically
    images = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith('.jpg')])
    
    if not images:
        print(f"Error: No images found in '{IMAGE_FOLDER}' folder.")
        return

    # Initialize UTM projection using the first image to detect the zone
    first_path = os.path.join(IMAGE_FOLDER, images[0])
    first_meta = get_dji_metadata(first_path)
    
    # Auto-detect UTM zone based on longitude
    utm_zone = int((first_meta['lon'] + 180) / 6) + 1
    # Note: Use +south if you are in the Southern Hemisphere
    projection = Proj(proj='utm', zone=utm_zone, ellps='WGS84', preserve_units=False)
    
    origin_x, origin_y = (0, 0)
    if USE_LOCAL_ORIGIN:
        origin_x, origin_y = projection(first_meta['lon'], first_meta['lat'])
        print(f"Origin set to UTM Zone {utm_zone}: {origin_x}, {origin_y}")

    with open(OUTPUT_FILE, 'w') as f:
        for img_name in images:
            img_path = os.path.join(IMAGE_FOLDER, img_name)
            meta = get_dji_metadata(img_path)
            
            # 1. Filename (stripped of .JPG)
            img_id = os.path.splitext(img_name)[0]
            
            # 2. Position (tx, ty, tz) in Meters
            curr_x, curr_y = projection(meta['lon'], meta['lat'])
            tx = curr_x - origin_x
            ty = curr_y - origin_y
            tz = meta['alt']
            
            # 3. Orientation (qx, qy, qz, qw)
            # Nadir logic: DJI Pitch -90 means looking down.
            # In TUM SLAM format, the camera points down when it is rotated 180 deg around X.
            # We use the gimbal angles directly as an 'xyz' Euler sequence.
            rot = R.from_euler('xyz', [meta['roll'], meta['pitch'], meta['yaw']], degrees=True)
            qx, qy, qz, qw = rot.as_quat()
            
            # Format: id tx ty tz qx qy qz qw
            line = f"{img_id} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
            f.write(line)
            print(f"Processed: {img_id}")

    print(f"\nSuccess! '{OUTPUT_FILE}' has been created.")

if __name__ == "__main__":
    create_trajectory()
