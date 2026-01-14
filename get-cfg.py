import math
from PIL import Image
import exifread
import numpy as np

def get_camera_parameters(image_path):
    # ---------------------------
    # Load image size
    # ---------------------------
    img = Image.open(image_path)
    W, H = img.size

    # ---------------------------
    # Read EXIF
    # ---------------------------
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f, details=False)

    # ---------------------------
    # Required EXIF fields
    # ---------------------------
    if 'EXIF FocalLength' not in tags:
        raise RuntimeError("FocalLength not found in EXIF")

    if 'EXIF FocalLengthIn35mmFilm' not in tags:
        raise RuntimeError("35mm equivalent focal length not found in EXIF")

    # Focal length in mm
    f_mm = tags['EXIF FocalLength'].values[0].num / \
           tags['EXIF FocalLength'].values[0].den

    # 35mm equivalent focal length (mm)
    f_35mm = int(tags['EXIF FocalLengthIn35mmFilm'].values[0])

    # ---------------------------
    # Compute sensor width (mm)
    # ---------------------------
    FULL_FRAME_WIDTH = 36.0  # mm
    crop_factor = f_35mm / f_mm
    sensor_width_mm = FULL_FRAME_WIDTH / crop_factor

    # ---------------------------
    # Compute intrinsics
    # ---------------------------
    fx = (f_mm / sensor_width_mm) * W
    fy = fx * (H / W)   # preserve aspect ratio

    cx = W / 2.0
    cy = H / 2.0

    camera_parameters = [W, H, fx, fy, cx, cy]

    return camera_parameters

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    img_path = "rgb/DJI_20251222151152_0001_V.jpg"

    params = get_camera_parameters(img_path)

    print("Camera.Parameters =")
    print(f"[{params[0]}, {params[1]}, "
          f"{params[2]:.3f}, {params[3]:.3f}, "
          f"{params[4]:.3f}, {params[5]:.3f}]")
