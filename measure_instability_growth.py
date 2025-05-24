import os
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import re
from natsort import natsorted
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline

# === CONFIGURACIÓN ===
data_dir = "/home/yo/perturbation"
quantity = "Bcc"
y_center = 0.0
nx_total, ny_total = 512, 256
dt = 0.01*5.0

# === FUNCIONES AUXILIARES ===

def extract_step_from_filename(filename):
    match = re.search(r"\.(\d{5})\.vtk$", filename)
    return int(match.group(1)) if match else -1

def get_by_centerline(mesh):
    try:
        coords = mesh.points
        y_coords = coords[:, 1]
        idx_center = np.argmin(np.abs(y_coords - y_center))

        if quantity in mesh.array_names:
            vec = mesh[quantity]
            if vec.ndim == 2 and vec.shape[1] >= 2:
                return vec[idx_center, 1]  # componente By
    except Exception as e:
        return None
    return None

# === PROCESAMIENTO ===

vtk_files = [f for f in os.listdir(data_dir) if f.endswith(".vtk")]
vtk_files = natsorted(vtk_files, key=extract_step_from_filename)

by_max_by_step = {}

for file in vtk_files:
    filepath = os.path.join(data_dir, file)
    try:
        mesh = pv.read(filepath)
        step = extract_step_from_filename(file)
        val = get_by_centerline(mesh)
        if val is not None:
            if step not in by_max_by_step:
                by_max_by_step[step] = []
            by_max_by_step[step].append(abs(val))
    except Exception as e:
        print(f"Error leyendo {file}: {e}")

# === DATOS PROCESADOS ===

steps = sorted(by_max_by_step.keys())
by_max = [np.mean(by_max_by_step[s]) for s in steps]
times = [s * dt for s in steps] 
ln_by = np.log(by_max)

# === SUAVIZADO y EXTRAPOLACIÓN ===

ln_by_smooth = savgol_filter(ln_by, window_length=11, polyorder=5)
spline_fit = UnivariateSpline(times, ln_by_smooth, s=0.1)

# === GRAFICAR ===

plt.figure(figsize=(10, 5))
#plt.plot(times, ln_by, 'o', label="Raw data", markersize=3, alpha=0.5)
plt.plot(times, ln_by_smooth, '-', color='tab:blue', lw=2, label="Smoothed (Savitzky-Golay)")
#plt.plot(times, spline_fit(times), '--', color='darkred', lw=2, label="Spline fit")

plt.xlabel("Time")
plt.ylabel(r"$\ln(\max B_y)$")
plt.title("Growth of tearing instability at $y=0$")
#plt.grid(True, ls='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("Instability_growth_phy=0.1.png", dpi=150)
plt.show()
