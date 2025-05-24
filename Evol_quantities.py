# plot_energy_from_hst.py

import numpy as np
import matplotlib.pyplot as plt

# === Cambia esto si tu archivo se llama diferente ===
filename = "/home/yo/perturbation/harris_laminar.hst"

# === Leer el archivo, omitir líneas con comentarios ===
data = np.loadtxt(filename, comments="#")

# === Extraer columnas relevantes ===
time    = data[:, 0]  # tiempo
dt      = data[:, 1]  # paso de tiempo
KE_x    = data[:, 6]  # energía cinética en x
KE_y    = data[:, 7]  # energía cinética en y
E_total = data[:, 9]  # energía total
ME_x    = data[:, 10] # energía magnética en x
ME_y    = data[:, 11] # energía magnética en y

# === Cálculos derivados ===
KE_total = KE_x + KE_y
ME_total = ME_x + ME_y  # Energía magnética total (sin Bz)

# === Figura con subplots ===
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Energía total
axs[0].plot(time, E_total, label="Total Energy", lw=2)
axs[0].set_ylabel("Total Energy")
axs[0].legend()

# Energía cinética
axs[1].plot(time, KE_total, label="Kinetic Energy", lw=2, color='tab:green')
axs[1].set_ylabel("Kinetic Energy")
axs[1].legend()

# Energía magnética
axs[2].plot(time, ME_x, label="Magnetic Energy $B_x$", lw=2, color='tab:blue')
axs[2].plot(time, ME_y, label="Magnetic Energy $B_y$", lw=2, color='tab:orange')
axs[2].plot(time, ME_total, label="Magnetic Energy Total", lw=2, color='tab:red', linestyle='--')
axs[2].set_ylabel("Magnetic Energy")
axs[2].set_xlabel("Time")
axs[2].legend()

# Ajustes finales
fig.suptitle("Energy Evolution from Athena++ History File", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("energy_vs_time_subplots.png", dpi=150)
plt.show()
