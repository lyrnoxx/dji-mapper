import os
import math
import numpy as np
from PIL import Image
import exifread

# ==============================
# CONFIG
# ==============================
IMAGE_DIR = "images-true"
OUTPUT_FILE = "new-trajectory.txt"

DEFAULT_ALTITUDE = 50.0   # meters (used if no EXIF altitude)
EARTH_RADIUS = 6378137.0  # meters

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
# NADIR QUATERNION (FIXED)
# ==============================
def nadir_quaternion(yaw=0.0):
    """
    roll = 0
    pitch = -90 deg
    yaw = input
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

with open(OUTPUT_FILE, 'w') as f:
    for img, lat, lon, alt in gps_data:
        name = os.path.splitext(img)[0]
        x, y = latlon_to_xy(lat, lon, lat0, lon0)
        qx, qy, qz, qw = (0.70710678, 0.0, 0.0, 0.70710678)

        f.write(f"{name} {x:.3f} {y:.3f} {alt:.2f} "
                f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")

print(f"[OK] Saved trajectory to {OUTPUT_FILE}")
