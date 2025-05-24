# make_video.py

import os
import imageio.v2 as imageio
from natsort import natsorted

def make_video_from_folder(image_folder, output_video, video_name, fps=5):
    images = [f for f in os.listdir(image_folder) if f.endswith(".png")]
    images = natsorted(images)

    if not images:
        print(f"[ERROR] No se encontraron imágenes en {image_folder}")
        return

    os.makedirs(output_video, exist_ok=True)
    output_path = os.path.join(output_video, video_name)

    writer = imageio.get_writer(
        output_path, 
        fps=fps, 
        codec='libx264', 
        format='ffmpeg'
    )

    for img_name in images:
        img_path = os.path.join(image_folder, img_name)
        image = imageio.imread(img_path)
        writer.append_data(image)
        print(f"Añadido: {img_name}")

    writer.close()

    print(f"\n✅ Video MP4 guardado como: {output_path}")
