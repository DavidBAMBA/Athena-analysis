#!/usr/bin/env python
import os
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from natsort import natsorted
import re
import multiprocessing as mp

# Parámetros físicos
B0 = 1.0         # Campo magnético externo
rho0 = 1.0       # Densidad central
eta = 1e-4       # Resistividad
VA = B0 / np.sqrt(rho0)

# ───────────────────────────── Configuración ─────────────────────────────
data_dir       = "/home/yo/magnetic_reconnection_data/psi001000_eta1E-3"
quantity_name  = "Bcc"
output_folder  = f"frames_Jz_psi001000_eta1E-3"
os.makedirs(output_folder, exist_ok=True)

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

reconnection_rates = []

# ───────────────────────────── Análisis por paso ─────────────────────────────
def process_frame(step):
    print(f"[PID {os.getpid()}] Procesando paso {step} …")
    step_files = [f for f in vtk_files if f".{step}.vtk" in f]
    if not step_files:
        print(f"[Aviso] No hay bloques para t={step}")
        return None  # si falla, no guardar nada

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

    # Guardar figura
    x = np.linspace(x_min_dom, x_max_dom, NXtot)
    y = np.linspace(y_min_dom, y_max_dom, NYtot)
    plt.figure(figsize=(10, 4))
    plt.imshow(Jz, extent=[x_min_dom, x_max_dom, y_min_dom, y_max_dom],
               origin="lower", cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect='auto')
    plt.colorbar(label=r"$J_z$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Corriente $J_z$, t = {step}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"Jz_{step}.png"))
    plt.close()

    # Calcular reconexión en el centro
    cx, cy = NXtot // 2, NYtot // 2
    Jz_center = np.abs(Jz[cy, cx])
    Ez = eta * Jz_center
    R = Ez / (B0 * VA)

    return step, R

# ───────────────────────────── Paralelización ─────────────────────────────
if __name__ == "__main__":
    with mp.Pool(2) as pool:
        results = pool.map(process_frame, time_steps)

    # Filtrar resultados válidos
    reconnection_rates = [(int(step), R) for step, R in results if step is not None]

    # Guardar resultados en archivo
    reconnection_rates.sort()  # ordenar por paso de tiempo
    np.savetxt("reconnection_rate-2.txt", reconnection_rates, header="step R")

    print("\n✅ ¡Listo! Tasa de reconexión guardada en: reconnection_rate-2.txt")
    print("✅ Las figuras están en:", output_folder)
