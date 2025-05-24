import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from natsort import natsorted
import multiprocessing as mp
from make_video import make_video_from_folder

# =================== CONFIGURACIÓN ===================
data_dir       = "/home/yo/perturbation2"
output_folder  = "frames_profiles2"
video_folder   = "videos"
video_filename = "evolucion_perfiles2_1D.mp4"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(video_folder, exist_ok=True)

quantity_rho   = "rho"
quantity_pres  = "press"
quantity_B     = "Bcc"

x_min_dom, x_max_dom = -1.0, 1.0
y_min_dom, y_max_dom = -0.25, 0.25
NXb, NYb = 64, 64
block_dx = (x_max_dom - x_min_dom) / 8
block_dy = (y_max_dom - y_min_dom) / 4
NXtot = NXb * 8
NYtot = NYb * 4

vtk_files = natsorted([f for f in os.listdir(data_dir) if f.endswith(".vtk")])
time_steps = sorted({re.search(r"\.(\d{5})\.vtk$", f).group(1) for f in vtk_files})

# =================== FUNCIONES ===================
def plot_profiles(step):
    print(f"[PID {os.getpid()}] Paso {step} …")
    step_files = [f for f in vtk_files if f".{step}.vtk" in f]
    if not step_files:
        print(f"[Aviso] No hay bloques para t={step}")
        return

    rho_2d = np.zeros((NYtot, NXtot))
    P_2d   = np.zeros((NYtot, NXtot))
    By_2d  = np.zeros((NYtot, NXtot))

    for fname in step_files:
        mesh = pv.read(os.path.join(data_dir, fname))
        rho = mesh[quantity_rho].reshape((NYb, NXb))
        press = mesh[quantity_pres].reshape((NYb, NXb))
        Bvec = mesh[quantity_B].reshape((NYb, NXb, 3))
        By = Bvec[:, :, 1]

        b = mesh.bounds
        ix = int(round((b[0] - x_min_dom) / block_dx))
        iy = int(round((b[2] - y_min_dom) / block_dy))
        x0, x1 = ix * NXb, (ix + 1) * NXb
        y0, y1 = iy * NYb, (iy + 1) * NYb

        rho_2d[y0:y1, x0:x1] = rho
        P_2d[y0:y1, x0:x1]   = press
        By_2d[y0:y1, x0:x1]  = By

    # Coordenadas y
    y = np.linspace(y_min_dom, y_max_dom, NYtot)
    x = np.linspace(x_min_dom, x_max_dom, NXtot)
    ix0 = np.argmin(np.abs(x - 0.0))  # índice más cercano a x=0

    rho1 = rho_2d[:, ix0] - np.mean(rho_2d[:, ix0][-10:])  # fondo removido
    T1   = ((P_2d[:, ix0] / rho_2d[:, ix0]) - np.mean(P_2d[:, ix0][-10:] / rho_2d[:, ix0][-10:]))*(-1.0)
    By1  = By_2d[:, ix0]  # sin remover fondo

    # =================== PLOTEO ===================
    fig, axs = plt.subplots(3, 1, figsize=(6, 7), sharex=True)

    axs[0].plot(y, rho1, label=r"$\rho_1(y)$")
    axs[0].set_ylabel(r"$\rho_1$")

    axs[1].plot(y, T1, label=r"$T_1(y)$", color='darkgreen')
    axs[1].set_ylabel(r"$T_1$")

    axs[2].plot(y, By1, label=r"$B_{1y}(y)$", color='darkred')
    axs[2].set_ylabel(r"$B_{1y}$")
    axs[2].set_xlabel("y")

    plt.suptitle(f"Perfiles en x=0, paso {step}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"profile_{step}.png"))
    plt.close()

# =================== EJECUCIÓN ===================
if __name__ == "__main__":
    with mp.Pool(4) as pool:
        pool.map(plot_profiles, time_steps)

    make_video_from_folder(output_folder, video_folder, video_filename, fps=1)
