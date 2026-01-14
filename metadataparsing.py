import subprocess
import json

# Replace with one of your actual image paths
test_image = "images-true/DJI_20251222151152_0001_V.JPG" 

# -G1 shows the Group Name (e.g., XMP, DJI, EXIF)
# -n returns numerical values (decimal instead of degrees/minutes)
cmd = ['exiftool', '-G1', '-n', '-j', test_image]
result = subprocess.run(cmd, capture_output=True, text=True)
metadata = json.loads(result.stdout)[0]

for key, value in metadata.items():
    print(f"{key}: {value}")
