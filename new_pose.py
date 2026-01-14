import os, json, subprocess, numpy as np
from pyproj import Proj
from scipy.spatial.transform import Rotation as R

IMAGE_FOLDER, OUTPUT_FILE = 'images-true', 'pose1.txt'

def get_meta(path):
    cmd = ['exiftool', '-n', '-j', '-GPSLatitude', '-GPSLongitude', '-RelativeAltitude', 
           '-GimbalRollDegree', '-GimbalPitchDegree', '-GimbalYawDegree', path]
    data = json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)[0]
    return data

images = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith('.jpg')])
first = get_meta(os.path.join(IMAGE_FOLDER, images[0]))
proj = Proj(proj='utm', zone=int((first['GPSLongitude'] + 180) / 6) + 1, ellps='WGS84')
ox, oy = proj(first['GPSLongitude'], first['GPSLatitude'])

with open(OUTPUT_FILE, 'w') as f:
    for name in images:
        m = get_meta(os.path.join(IMAGE_FOLDER, name))
        tx, ty = proj(m['GPSLongitude'], m['GPSLatitude'])
        # DJI standard: Yaw, then Pitch, then Roll (zyx)
        q = R.from_euler('zyx', [m['GimbalYawDegree'], m['GimbalPitchDegree'], m['GimbalRollDegree']], degrees=True).as_quat()
        f.write(f"{os.path.splitext(name)[0]} {tx-ox} {ty-oy} {m['RelativeAltitude']} {q[0]} {q[1]} {q[2]} {q[3]}\n")