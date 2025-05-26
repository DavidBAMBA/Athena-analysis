#!/usr/bin/env python
# density_movie.py
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from natsort import natsorted
import multiprocessing as mp
from make_video import make_video_from_folder

data_dir       = "/home/yo/magnetic_reconnection_data/psi_driving"
quantity_name  = "rho"
output_folder  = f"frames_{quantity_name}_psi_driving"
video_folder   = "videos"
video_filename = f"evolucion_{quantity_name}_psi_driving.mp4"
os.makedirs(output_folder, exist_ok=True)

x_min_dom, x_max_dom = -1.0, 1.0
y_min_dom, y_max_dom = -0.25, 0.25

NXb, NYb = 64, 64
block_dx = (x_max_dom - x_min_dom) / 16
block_dy = (y_max_dom - y_min_dom) / 8
NXtot = NXb * 16
NYtot = NYb * 8

vtk_files = natsorted([f for f in os.listdir(data_dir) 
                       if f.startswith("harris_laminar.block") and "prim" in f and f.endswith(".vtk")])

time_steps = sorted({match.group(1) for f in vtk_files if (match := re.search(r"\.(\d{5})\.vtk$", f))})

def plot_rho_frame(step):
    print(f"[PID {os.getpid()}] Procesando paso {step} …")
    step_files = [f for f in vtk_files if f".{step}.vtk" in f]
    if not step_files:
        print(f"[Aviso] No hay bloques para t={step}")
        return

    rho_full = np.zeros((NYtot, NXtot))

    for fname in step_files:
        mesh = pv.read(os.path.join(data_dir, fname))
        rho = mesh[quantity_name].reshape((NYb, NXb))

        b = mesh.bounds
        ix = int(round((b[0] - x_min_dom) / block_dx))
        iy = int(round((b[2] - y_min_dom) / block_dy))

        x0, x1 = ix * NXb, (ix + 1) * NXb
        y0, y1 = iy * NYb, (iy + 1) * NYb
        rho_full[y0:y1, x0:x1] = rho

    plt.figure(figsize=(8, 3))
    plt.imshow(
        rho_full,
        extent=[x_min_dom, x_max_dom, y_min_dom, y_max_dom],
        origin="lower",
        cmap="magma",
        vmin=0.0, vmax=1.0,
        aspect='auto'
    )
    plt.colorbar(label=r"$\rho$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{quantity_name}_{step}.png"))
    plt.close()

if __name__ == "__main__":
    with mp.Pool(2) as pool:
        pool.map(plot_rho_frame, time_steps)

    print(f"\n ¡Listo! Las figuras están en: {output_folder}")
    make_video_from_folder(output_folder, video_folder, video_filename, fps=5)
