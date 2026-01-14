from PIL import Image
import os

input_dir = "images-true"
output_dir = "images"
max_side = 2000   # SAFE for ODM + low RAM

os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if fname.lower().endswith(('.jpg', '.jpeg')):
        img = Image.open(os.path.join(input_dir, fname))
        w, h = img.size

        scale = max_side / max(w, h)
        if scale >= 1:
            img.save(os.path.join(output_dir, fname))
            continue

        new_size = (int(w * scale), int(h * scale))
        img_resized = img.resize(new_size, Image.LANCZOS)

        exif = img.info.get("exif")
        img_resized.save(
            os.path.join(output_dir, fname),
            quality=95,
            exif=exif
        )

print("Resizing done.")
