#!/usr/bin/env python
import os
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from natsort import natsorted
import re
import multiprocessing as mp

from make_video import make_video_from_folder

data_dir       = "/home/yo/magnetic_reconnection_data/psi001000_eta1E-4"
quantity_name  = "vel"
output_folder  = f"frames_{quantity_name}_psi001000_eta1E-4"
video_folder   = "videos"
video_filename = f"evolucion_{quantity_name}_psi001000_eta1E-4.mp4"
os.makedirs(output_folder, exist_ok=True)

x_min_dom, x_max_dom = -1.0, 1.0
y_min_dom, y_max_dom = -0.25, 0.25

NXb, NYb = 64, 64
block_dx = (x_max_dom - x_min_dom) / 32
block_dy = (y_max_dom - y_min_dom) / 8
NXtot = NXb * 32
NYtot = NYb * 8

vtk_files = natsorted([f for f in os.listdir(data_dir) 
                       if f.startswith("harris_laminar.block") and "prim" in f and f.endswith(".vtk")])

time_steps = sorted({match.group(1) for f in vtk_files if (match := re.search(r"\.(\d{5})\.vtk$", f))})

def plot_velocity_frame(step):
    print(f"[PID {os.getpid()}] Procesando paso {step} …")
    step_files = [f for f in vtk_files if f".{step}.vtk" in f]
    if not step_files:
        print(f"[Aviso] No hay bloques para t={step}")
        return

    vmag_full = np.zeros((NYtot, NXtot))
    vx_full   = np.zeros((NYtot, NXtot))
    vy_full   = np.zeros((NYtot, NXtot))

    for fname in step_files:
        mesh = pv.read(os.path.join(data_dir, fname))
        vvec = mesh[quantity_name]  # (N, 3)
        vmag = np.linalg.norm(vvec, axis=1)
        vx = vvec[:, 0]
        vy = vvec[:, 1]

        vmag = vmag.reshape((NYb, NXb))
        vx = vx.reshape((NYb, NXb))
        vy = vy.reshape((NYb, NXb))

        b = mesh.bounds
        ix = int(round((b[0] - x_min_dom) / block_dx))
        iy = int(round((b[2] - y_min_dom) / block_dy))
        x0, x1 = ix * NXb, (ix + 1) * NXb
        y0, y1 = iy * NYb, (iy + 1) * NYb

        vmag_full[y0:y1, x0:x1] = vmag
        vx_full[y0:y1, x0:x1] = vx
        vy_full[y0:y1, x0:x1] = vy

    x = np.linspace(x_min_dom, x_max_dom, NXtot)
    y = np.linspace(y_min_dom, y_max_dom, NYtot)
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(10, 4))
    plt.imshow(vmag_full, extent=[x_min_dom, x_max_dom, y_min_dom, y_max_dom],
               origin="lower", cmap="viridis", vmin=0.0, vmax=0.1, aspect="auto")
    plt.colorbar(label=r"$|\vec{v}|$")
    plt.streamplot(X, Y, vx_full, vy_full, color="white", density=1.0, linewidth=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{quantity_name}_{step}.png"))
    plt.close() 




if __name__ == "__main__":
    with mp.Pool(6) as pool:
        pool.map(plot_velocity_frame, time_steps)
    print(f"\n ¡Listo! Las figuras están en: {output_folder}")
    make_video_from_folder(output_folder, video_folder, video_filename, fps=5)
