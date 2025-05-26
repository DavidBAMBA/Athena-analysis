#!/usr/bin/env python

import os
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from natsort import natsorted
import re

# ================================
# CONFIGURACIÓN
# ================================
data_dir = "/home/yo/magnetic_reconnection_data/psi001000_eta1E-3"
quantity_name = "Bcc"
eta_ohm = 1e-3  # Resistividad
output_file = "reconnection_rate_vs_time.txt"

x_min_dom, x_max_dom = -1.0, 1.0
y_min_dom, y_max_dom = -0.25, 0.25
NXb, NYb = 64, 64
nblocks_x, nblocks_y = 32, 8
block_dx = (x_max_dom - x_min_dom) / nblocks_x
block_dy = (y_max_dom - y_min_dom) / nblocks_y
NXtot = NXb * nblocks_x
NYtot = NYb * nblocks_y

# Parámetros físicos para teoría
L = 2.0
B0 = 1.0
rho0 = 1.0
Va = B0 / np.sqrt(rho0)
S = (L * Va) / eta_ohm
R_sp_theory = Va * B0 * S**-0.5
delta_sp_theory = L * S**-0.5

print(f"Número de Lundquist S = {S:.1e}")
print(f"Tasa teórica Sweet-Parker (R_sp) ≈ {R_sp_theory:.3e}")
print(f"Grosor teórico Sweet-Parker (δ_sp) ≈ {delta_sp_theory:.3e}")

# ================================
# PROCESAR VTK Y CALCULAR R(t)
# ================================
vtk_files = natsorted([f for f in os.listdir(data_dir)
                       if f.startswith("harris_laminar.block") and "prim" in f and f.endswith(".vtk")])

time_steps = sorted({match.group(1) for f in vtk_files if (match := re.search(r"\.(\d+)\.vtk$", f))})

time_list = []
reconnection_rate_list = []
Jz_profiles = []
By_profiles = []

for step in time_steps:
    step_files = [f for f in vtk_files if f".{step}.vtk" in f]
    if not step_files:
        continue

    Bx_full = np.zeros((NYtot, NXtot))
    By_full = np.zeros((NYtot, NXtot))

    for fname in step_files:
        mesh = pv.read(os.path.join(data_dir, fname))
        Bvec = mesh[quantity_name]
        Bx = Bvec[:, 0]
        By = Bvec[:, 1]

        Bx = Bx.reshape((NYb, NXb))
        By = By.reshape((NYb, NXb))

        b = mesh.bounds
        ix = int(round((b[0] - x_min_dom) / block_dx))
        iy = int(round((b[2] - y_min_dom) / block_dy))

        x0, x1 = ix * NXb, (ix + 1) * NXb
        y0, y1 = iy * NYb, (iy + 1) * NYb

        Bx_full[y0:y1, x0:x1] = Bx
        By_full[y0:y1, x0:x1] = By

    # Derivadas centrales para Jz ≈ dBy/dx - dBx/dy
    x = np.linspace(x_min_dom, x_max_dom, NXtot)
    y = np.linspace(y_min_dom, y_max_dom, NYtot)
    dBy_dx = np.gradient(By_full, x, axis=1)
    dBx_dy = np.gradient(Bx_full, y, axis=0)
    Jz_full = dBy_dx - dBx_dy

    center_x_idx = NXtot // 2
    center_y_idx = NYtot // 2
    Ez_max = eta_ohm * np.max(np.abs(Jz_full[center_y_idx, :]))

    # Guardar perfiles verticales (a lo largo de y, en x centro)
    Jz_profiles.append(Jz_full[:, center_x_idx])
    By_profiles.append(By_full[:, center_x_idx])

    time = float(step) * 0.1  # Ajusta si tu paso real es diferente
    time_list.append(time)
    reconnection_rate_list.append(Ez_max)

    print(f"Step {step} | Time {time:.2f} | Ez_max = {Ez_max:.4e}")

# ================================
# ANALIZAR RESULTADOS
# ================================
time_arr = np.array(time_list)
R_arr = np.array(reconnection_rate_list)

# Tomar promedio después de t > 20 (ajusta si necesario)
mask = time_arr > 20
R_sim_mean = np.mean(R_arr[mask])

print(f"Tasa promedio simulada (t > 20): R_sim ≈ {R_sim_mean:.3e}")
relative_error = np.abs(R_sp_theory - R_sim_mean) / R_sp_theory * 100
print(f"Error relativo vs teórico ≈ {relative_error:.2f}%")

# ================================
# ESTIMAR GROSOR DE LÁMINA
# ================================
# Promediar perfil de |Jz(y)| en tiempos finales
Jz_profiles = np.array(Jz_profiles)
Jz_mean_profile = np.mean(np.abs(Jz_profiles[mask, :]), axis=0)

# Calcular grosor: ancho a mitad del máximo
Jz_max = np.max(Jz_mean_profile)
half_max = Jz_max / 2
indices = np.where(Jz_mean_profile >= half_max)[0]
y_span = y[indices]
delta_sim = np.abs(y_span[-1] - y_span[0])

print(f"Grosor promedio de lámina (δ_sim) ≈ {delta_sim:.3e}")
error_delta = np.abs(delta_sp_theory - delta_sim) / delta_sp_theory * 100
print(f"Error relativo del grosor vs teoría ≈ {error_delta:.2f}%")

# ================================
# GRAFICAR TASA EN EL TIEMPO
# ================================
plt.figure(figsize=(8, 5))
plt.plot(time_arr, R_arr, 'o-', label="Simulación")
plt.axhline(R_sp_theory, color='red', linestyle='--', label="Sweet-Parker teórico")
plt.xlabel("Tiempo")
plt.ylabel("Tasa de reconexión $R$")
plt.title("Comparación entre simulación y modelo Sweet-Parker")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("comparison_sweet_parker.png")
plt.show()

# ================================
# GRAFICAR PERFILES ESPACIALES
# ================================
plt.figure(figsize=(8, 5))
plt.plot(y, Jz_mean_profile, label=r"$|J_z(y)|$")
plt.axhline(half_max, color='gray', linestyle='--', label="Mitad del máximo")
plt.xlabel("y")
plt.ylabel(r"$|J_z|$")
plt.title("Perfil promedio vertical de $|J_z|$")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("profile_Jz.png")
plt.show()

By_profiles = np.array(By_profiles)
By_mean_profile = np.mean(By_profiles[mask, :], axis=0)

plt.figure(figsize=(8, 5))
plt.plot(y, By_mean_profile, label=r"$B_y(y)$")
plt.xlabel("y")
plt.ylabel(r"$B_y$")
plt.title("Perfil promedio vertical de $B_y$")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("profile_By.png")
plt.show()
