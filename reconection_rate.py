# reconnection_rate_custom.py
import os
import numpy as np
import pyvista as pv
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from natsort import natsorted
import re
from scipy.signal import savgol_filter  # Añade esto al principio con los otros imports

# === CONFIGURACIÓN ===
data_dir = "/home/yo/perturbation2"
quantity = "Bcc"
dx = 2.0 / 512   # de x = [-1, 1]
dy = 0.5 / 256   # de y = [-0.25, 0.25]

NXb, NYb = 64, 64
NXtot, NYtot = 512, 256

vtk_files = natsorted([f for f in os.listdir(data_dir) if f.endswith(".vtk")])
time_steps = sorted({re.search(r"\.(\d{5})\.vtk$", f).group(1) for f in vtk_files})

def filename_to_time(step_str, dt=0.01):
    return int(step_str) * dt

def compute_B_energy_for_step(step):
    files = [f for f in vtk_files if f".{step}.vtk" in f]
    Bsq_full = np.zeros((NYtot, NXtot))

    for fname in files:
        mesh = pv.read(os.path.join(data_dir, fname))
        Bvec = mesh[quantity]  # (4096, 3)
        Bsq = np.sum(Bvec**2, axis=1)
        Bsq = Bsq.reshape((NYb, NXb))

        b = mesh.bounds
        ix = int(round((b[0] + 1.0) / (2.0 / 8)))  # 8 bloques
        iy = int(round((b[2] + 0.25) / (0.5 / 4))) # 4 bloques

        x0, x1 = ix * NXb, (ix + 1) * NXb
        y0, y1 = iy * NYb, (iy + 1) * NYb
        Bsq_full[y0:y1, x0:x1] = Bsq

    # Integración doble sobre el dominio
    Ey = simpson(Bsq_full, dx=dy, axis=0)
    Exy = simpson(Ey, dx=dx)
    return Exy

# === Loop para cada paso de tiempo ===
energies = []
times = []

for step in time_steps:
    t = filename_to_time(step)
    try:
        E_B = compute_B_energy_for_step(step)
        energies.append(E_B)
        times.append(t)
        print(f"t = {t:.2f} → E_B = {E_B:.6f}")
    except Exception as e:
        print(f"[Error] en t={t}: {e}")

energies = np.array(energies)
times = np.array(times)

# === Derivada temporal: dE/dt
def central_diff(y, x):
    dydt = np.zeros_like(y)
    dx = np.diff(x)
    dydt[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
    dydt[0] = (y[1] - y[0]) / dx[0]
    dydt[-1] = (y[-1] - y[-2]) / dx[-1]
    return dydt

reconnection_rate = - central_diff(energies, times)

# === Suavizado con Savitzky-Golay ===
window_size = 21  # debe ser impar y menor que len(times)
poly_order = 5    # grado del polinomio para el suavizado
reconnection_rate_smooth = savgol_filter(reconnection_rate, window_size, poly_order)

# === Plot reconnection rate suavizada
plt.figure(figsize=(8, 4))
plt.plot(times, reconnection_rate_smooth, label=r"$-dE_B/dt$", color='crimson')
plt.xlabel("Time")
#plt.yscale('log')
plt.ylabel("Reconnection Rate")
plt.title("Magnetic Reconnection Rate")
plt.tight_layout()
plt.legend()
plt.savefig("reconnection_rate_phi=0.01$.png", dpi=120)
plt.show()