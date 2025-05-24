import subprocess
import os

def get_video_duration(path):
    """Usa ffprobe para obtener duración en segundos del video."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration", "-of",
                "default=noprint_wrappers=1:nokey=1", path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return float(result.stdout.strip())
    except Exception as e:
        print(f"[ERROR] No se pudo obtener la duración de {path}: {e}")
        return None

def mp4_to_gif_high_quality(mp4_path, gif_path, fps=10, start_time=None, duration=None):
    duration = get_video_duration(mp4_path)
    if duration is None:
        print(f"[ERROR] Saltando archivo por duración desconocida: {mp4_path}")
        return

    tmp_palette = "palette.png"
    filters = f"fps={fps},scale=iw:ih:flags=lanczos"

    os.makedirs(os.path.dirname(gif_path), exist_ok=True)

    # Comando para generar la paleta optimizada
    palette_cmd = ['ffmpeg', '-y']
    if start_time:
        palette_cmd += ['-ss', str(start_time)]
    palette_cmd += ['-t', str(duration), '-i', mp4_path]
    palette_cmd += ['-vf', f'{filters},palettegen', tmp_palette]

    # Comando para generar el GIF con la paleta
    gif_cmd = ['ffmpeg', '-y']
    if start_time:
        gif_cmd += ['-ss', str(start_time)]
    gif_cmd += ['-t', str(duration), '-i', mp4_path, '-i', tmp_palette]
    gif_cmd += ['-lavfi', f'{filters} [x]; [x][1:v] paletteuse', gif_path]

    print(f"🎨 Generando paleta para: {mp4_path}")
    subprocess.run(palette_cmd, check=True)

    print(f"🎬 Generando GIF: {gif_path}")
    subprocess.run(gif_cmd, check=True)

    os.remove(tmp_palette)
    print(f"✅ GIF generado correctamente: {gif_path}\n")

if __name__ == "__main__":
    videos_dir = "videos"
    gifs_dir = "gifs"
    fps = 5  # ⚠️ Ajusta aquí para controlar la velocidad (menos FPS = más lento)

    if not os.path.exists(videos_dir):
        print(f"[ERROR] Carpeta no encontrada: {videos_dir}")
        exit(1)

    for filename in os.listdir(videos_dir):
        if filename.lower().endswith(".mp4"):
            mp4_path = os.path.join(videos_dir, filename)
            base_name = os.path.splitext(filename)[0]
            gif_path = os.path.join(gifs_dir, f"{base_name}.gif")

            try:
                mp4_to_gif_high_quality(
                    mp4_path=mp4_path,
                    gif_path=gif_path,
                    fps=fps,
                    start_time=0  # puedes poner None si no necesitas recorte
                )
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Falló la conversión de: {mp4_path}\n{e}")
