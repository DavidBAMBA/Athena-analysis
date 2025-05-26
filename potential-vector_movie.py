#!/usr/bin/env python
import os
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from natsort import natsorted
import re
from scipy.fft import fft2, ifft2, fftfreq
from make_video import make_video_from_folder

# ─────────── Configuración ───────────
data_dir       = "/home/yo/magnetic_reconnection_data/psi001000_eta1E-4"
quantity_name  = "Bcc"
output_folder  = f"frames_Az_potential_psi001000_eta1E-4"
video_folder   = "videos"
video_filename = f"evolucion_Az_potential_psi001000_eta1E-4.mp4"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(video_folder, exist_ok=True)

x_min_dom, x_max_dom = -1.0, 1.0
y_min_dom, y_max_dom = -0.25, 0.25
NXb, NYb = 64, 64
block_dx = (x_max_dom - x_min_dom) / 32
block_dy = (y_max_dom - y_min_dom) / 8
NXtot = NXb * 32
NYtot = NYb * 8

dx = (x_max_dom - x_min_dom) / NXtot
dy = (y_max_dom - y_min_dom) / NYtot

vtk_files = natsorted([f for f in os.listdir(data_dir) 
                       if f.startswith("harris_laminar.block") and "prim" in f and f.endswith(".vtk")])

time_steps = sorted({match.group(1) for f in vtk_files if (match := re.search(r"\.(\d{5})\.vtk$", f))})

# ─────────── Solver de Poisson ───────────
def solve_poisson(Jz, dx, dy):
    ny, nx = Jz.shape
    kx = fftfreq(nx, dx) * 2 * np.pi
    ky = fftfreq(ny, dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2
    K2[0,0] = 1.0  # evitar división por cero
    A_z_hat = fft2(-Jz) / K2
    A_z_hat[0,0] = 0.0  # quitar modo cero
    A_z = np.real(ifft2(A_z_hat))
    return A_z

# ─────────── Análisis por paso ───────────
def plot_Az_contours(step):
    print(f"[Paso {step}] Procesando…")
    step_files = [f for f in vtk_files if f".{step}.vtk" in f]
    if not step_files:
        print(f"[Aviso] No hay bloques para t={step}")
        return

    Bx_full = np.zeros((NYtot, NXtot))
    By_full = np.zeros((NYtot, NXtot))

    for fname in step_files:
        mesh = pv.read(os.path.join(data_dir, fname))
        Bvec = mesh[quantity_name]
        Bx = Bvec[:, 0].reshape((NYb, NXb))
        By = Bvec[:, 1].reshape((NYb, NXb))

        b = mesh.bounds
        ix = int(round((b[0] - x_min_dom) / block_dx))
        iy = int(round((b[2] - y_min_dom) / block_dy))
        x0, x1 = ix * NXb, (ix + 1) * NXb
        y0, y1 = iy * NYb, (iy + 1) * NYb

        Bx_full[y0:y1, x0:x1] = Bx
        By_full[y0:y1, x0:x1] = By

    # Calcular Jz
    dBy_dy, dBy_dx = np.gradient(By_full, dy, dx)
    dBx_dy, dBx_dx = np.gradient(Bx_full, dy, dx)
    Jz = dBy_dx - dBx_dy

    # Resolver A_z
    Az = solve_poisson(Jz, dx, dy)

    # Preparar malla
    x = np.linspace(x_min_dom, x_max_dom, NXtot)
    y = np.linspace(y_min_dom, y_max_dom, NYtot)
    X, Y = np.meshgrid(x, y)

    # Plotear solo contornos de Az
    plt.figure(figsize=(10, 4))
    plt.contour(X, Y, Az, levels=30, colors='black', linewidths=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Potencial vectorial $A_z$, t = {step}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"Az_contour_{step}.png"))
    plt.close()

# ─────────── Ejecución ───────────
if __name__ == "__main__":
    for step in time_steps:
        plot_Az_contours(step)

    make_video_from_folder(output_folder, video_folder, video_filename, fps=5)
