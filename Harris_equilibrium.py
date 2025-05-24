#!/usr/bin/env python3

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from natsort import natsorted
import multiprocessing as mp
from make_video import make_video_from_folder

# ========== CONFIGURACIÓN ==========
data_dir       = "/home/yo/no_perturbation"
output_folder  = "frames_pressure_equilibrium"
video_folder   = "videos"
video_filename = "evolucion_equilibrio_presiones.mp4"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(video_folder, exist_ok=True)

quantity_rho   = "rho"
quantity_pres  = "press"
quantity_B     = "Bcc"

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

# ========== FUNCIÓN DE PROCESO ==========
def plot_equilibrium_profile(step):
    print(f"[PID {os.getpid()}] Paso {step} …")
    step_files = [f for f in vtk_files if f".{step}.vtk" in f]
    if not step_files:
        print(f"[Aviso] No hay bloques para t={step}")
        return

    rho_2d = np.zeros((NYtot, NXtot))
    P_2d   = np.zeros((NYtot, NXtot))
    Bvec_3d = np.zeros((NYtot, NXtot, 3))

    for fname in step_files:
        mesh = pv.read(os.path.join(data_dir, fname))
        rho = mesh[quantity_rho].reshape((NYb, NXb))
        press = mesh[quantity_pres].reshape((NYb, NXb))
        Bvec = mesh[quantity_B].reshape((NYb, NXb, 3))

        b = mesh.bounds
        ix = int(round((b[0] - x_min_dom) / block_dx))
        iy = int(round((b[2] - y_min_dom) / block_dy))
        x0, x1 = ix * NXb, (ix + 1) * NXb
        y0, y1 = iy * NYb, (iy + 1) * NYb

        rho_2d[y0:y1, x0:x1]     = rho
        P_2d[y0:y1, x0:x1]       = press
        Bvec_3d[y0:y1, x0:x1, :] = Bvec

    # Coordenadas
    y = np.linspace(y_min_dom, y_max_dom, NYtot)
    x = np.linspace(x_min_dom, x_max_dom, NXtot)
    dy = y[1] - y[0]
    ix0 = np.argmin(np.abs(x - 0.0))

    # Perfiles en x = 0
    Pgas = P_2d[:, ix0]
    Bx, By, Bz = Bvec_3d[:, ix0, 0], Bvec_3d[:, ix0, 1], Bvec_3d[:, ix0, 2]
    Pmag = 0.5 * (Bx**2 + By**2 + Bz**2)
    Ptot = Pgas + Pmag

    # Derivada
    #dPtot_dy = np.gradient(Ptot, dy)
    dPtot_dy = np.zeros_like(Ptot)
    dPtot_dy[1:-1] = (Ptot[2:] - Ptot[:-2]) / (2 * dy)
    dPtot_dy[0]  = (Ptot[1] - Ptot[0]) / dy
    dPtot_dy[-1] = (Ptot[-1] - Ptot[-2]) / dy

    # ========== PLOTEO ==========
    fig, axs = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

    axs[0].plot(y, Pgas, label=r"$P_\mathrm{gas}$", color="blue")
    axs[0].plot(y, Pmag, label=r"$P_\mathrm{mag}$", color="orange")
    axs[0].plot(y, Ptot, label=r"$P_\mathrm{tot}$", color="black", linestyle="--")
    axs[0].set_ylabel("Presión")
    axs[0].set_title(f"Presión en x=0, paso {step}")
    axs[0].legend()

    axs[1].plot(y, dPtot_dy, label=r"$\partial_y P_\mathrm{tot}$", color="red")
    axs[1].axhline(0, color='gray', linestyle=':')
    axs[1].set_xlabel("y")
    axs[1].set_ylabel(r"$\partial_y P_\mathrm{tot}$")
    axs[1].legend()
    axs[1].set_ylim(-0.1, 0.1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"pressure_{step}.png"))
    plt.close()

# ========== EJECUCIÓN ==========
if __name__ == "__main__":
    with mp.Pool(4) as pool:
        pool.map(plot_equilibrium_profile, time_steps)

    make_video_from_folder(output_folder, video_folder, video_filename, fps=3)
