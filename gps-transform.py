import os
import math
import numpy as np
from PIL import Image
import exifread

# ==============================
# CONFIG
# ==============================
IMAGE_DIR = "rgb"
OUTPUT_FILE = "latest-trajectory.txt"

DEFAULT_ALTITUDE = 50.0
EARTH_RADIUS = 6378137.0

# ==============================
# EXIF HELPERS
# ==============================
def dms_to_deg(dms, ref):
    deg = dms[0].num / dms[0].den
    min_ = dms[1].num / dms[1].den
    sec = dms[2].num / dms[2].den
    val = deg + min_ / 60.0 + sec / 3600.0
    return -val if ref in ['S', 'W'] else val

def read_gps(image_path):
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f, details=False)

    if 'GPS GPSLatitude' not in tags:
        return None

    lat = dms_to_deg(tags['GPS GPSLatitude'].values,
                     tags['GPS GPSLatitudeRef'].values)
    lon = dms_to_deg(tags['GPS GPSLongitude'].values,
                     tags['GPS GPSLongitudeRef'].values)

    alt = DEFAULT_ALTITUDE
    if 'GPS GPSAltitude' in tags:
        alt = tags['GPS GPSAltitude'].values[0].num / \
              tags['GPS GPSAltitude'].values[0].den

    return lat, lon, alt

# ==============================
# GPS → LOCAL ENU
# ==============================
def latlon_to_xy(lat, lon, lat0, lon0):
    lat, lon, lat0, lon0 = map(math.radians, [lat, lon, lat0, lon0])
    x = EARTH_RADIUS * (lon - lon0) * math.cos(lat0)
    y = EARTH_RADIUS * (lat - lat0)
    return x, y

# ==============================
# CALCULATE YAW FROM GPS TRACK
# ==============================
def calculate_yaw(x1, y1, x2, y2):
    """Calculate yaw angle (heading) from two consecutive positions"""
    dx = x2 - x1
    dy = y2 - y1
    yaw = math.atan2(dx, dy)  # North = 0, East = π/2
    return yaw

def nadir_quaternion(yaw=0.0):
    """
    Nadir-looking camera with specific yaw (heading)
    roll = 0
    pitch = -90 deg (pointing down)
    yaw = flight direction
    """
    pitch = -math.pi / 2
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)

    qx = sp * cy
    qy = 0.0
    qz = cp * sy
    qw = cp * cy
    return qx, qy, qz, qw

# ==============================
# MAIN
# ==============================
images = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

gps_data = []

for img in images:
    data = read_gps(os.path.join(IMAGE_DIR, img))
    if data:
        gps_data.append((img, *data))

if len(gps_data) == 0:
    raise RuntimeError("No GPS data found in images")

# Reference origin
lat0, lon0, _ = gps_data[0][1:]

# Convert all GPS to XY
positions = []
for img, lat, lon, alt in gps_data:
    x, y = latlon_to_xy(lat, lon, lat0, lon0)
    positions.append((img, x, y, alt))

# Calculate yaw for each image based on flight direction
with open(OUTPUT_FILE, 'w') as f:
    for i, (img, x, y, alt) in enumerate(positions):
        name = os.path.splitext(img)[0]
        
        # Calculate yaw from movement direction
        if i == 0:
            # First image: use direction to next image
            if len(positions) > 1:
                x_next, y_next = positions[i+1][1], positions[i+1][2]
                yaw = calculate_yaw(x, y, x_next, y_next)
            else:
                yaw = 0.0
        elif i == len(positions) - 1:
            # Last image: use direction from previous image
            x_prev, y_prev = positions[i-1][1], positions[i-1][2]
            yaw = calculate_yaw(x_prev, y_prev, x, y)
        else:
            # Middle images: average direction between prev and next
            x_prev, y_prev = positions[i-1][1], positions[i-1][2]
            x_next, y_next = positions[i+1][1], positions[i+1][2]
            yaw = calculate_yaw(x_prev, y_prev, x_next, y_next)
        
        # Generate nadir quaternion with calculated yaw
        qx, qy, qz, qw = nadir_quaternion(yaw)
        
        f.write(f"{name} {x:.3f} {y:.3f} {alt:.2f} "
                f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")

print(f"[OK] Saved trajectory to {OUTPUT_FILE}")
print(f"[INFO] Processed {len(positions)} images with direction-based orientation")