#!/usr/bin/env python
# create_magnetic_field.py

import os
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from natsort import natsorted
import re
import multiprocessing as mp
from make_video import make_video_from_folder 

data_dir       = "/home/yo/no_perturbation"
quantity_name  = "Bcc"
output_folder  = f"frames_{quantity_name}_p1"
video_folder   = "videos"
video_filename = f"evolucion_{quantity_name}_p1.mp4"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(video_folder, exist_ok=True)

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

def plot_B_frame(step):
    print(f"[PID {os.getpid()}] Procesando paso {step} …")
    step_files = [f for f in vtk_files if f".{step}.vtk" in f]
    if not step_files:
        print(f"[Aviso] No hay bloques para t={step}")
        return

    Bmag_full = np.zeros((NYtot, NXtot))
    Bx_full   = np.zeros((NYtot, NXtot))
    By_full   = np.zeros((NYtot, NXtot))

    for fname in step_files:
        mesh = pv.read(os.path.join(data_dir, fname))
        Bvec = mesh[quantity_name]
        Bmag = np.linalg.norm(Bvec, axis=1)
        Bx = Bvec[:, 0]
        By = Bvec[:, 1]

        Bmag = Bmag.reshape((NYb, NXb))
        Bx = Bx.reshape((NYb, NXb))
        By = By.reshape((NYb, NXb))

        b = mesh.bounds
        ix = int(round((b[0] - x_min_dom) / block_dx))
        iy = int(round((b[2] - y_min_dom) / block_dy))

        x0, x1 = ix*NXb, (ix+1)*NXb
        y0, y1 = iy*NYb, (iy+1)*NYb

        Bmag_full[y0:y1, x0:x1] = Bmag
        Bx_full[y0:y1, x0:x1] = Bx
        By_full[y0:y1, x0:x1] = By

    x = np.linspace(x_min_dom, x_max_dom, NXtot)
    y = np.linspace(y_min_dom, y_max_dom, NYtot)
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(10, 4))
    plt.imshow(Bmag_full, extent=[x_min_dom, x_max_dom, y_min_dom, y_max_dom],
               origin="lower", cmap="inferno", vmin=0.0, vmax=1.0, aspect='auto')
    plt.colorbar(label=r"$|\vec{B}|$")
    plt.streamplot(X, Y, Bx_full, By_full, color="white", density=1.0, linewidth=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{quantity_name}_{step}.png"))
    plt.close()

if __name__ == "__main__":
    with mp.Pool(6) as pool:
        pool.map(plot_B_frame, time_steps)

    make_video_from_folder(output_folder, video_folder, video_filename,fps=5)
