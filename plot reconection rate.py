#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

# Cargar datos
data = np.loadtxt("reconnection_rate-2.txt")
steps = data[:, 0]
rates = data[:, 1]

# Opcional: convertir pasos a tiempo físico si conoces dt
# ← Usa el dt correcto basado en tus datos
dt = 100 / np.max(steps)
time = steps * dt

# Plot
plt.figure(figsize=(8, 5))
plt.plot(time, rates, marker='o', linestyle='-', color='navy')
plt.xlabel("Tiempo [s]")
plt.ylabel("Tasa de reconexión $R(t)$")
plt.title("Evolución de la tasa de reconexión magnética")
plt.grid(True)
plt.tight_layout()
plt.savefig("reconnection_rate_plot.png")
plt.show()

print("✅ Gráfico guardado como reconnection_rate_plot.png")
