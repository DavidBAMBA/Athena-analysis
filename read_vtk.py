import os
import pyvista as pv
import re

# Configura aquí tu ruta
data_dir = "/home/yo/no_perturbation"

# Obtener archivos VTK ordenados
vtk_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".vtk")])

# Extraer pasos de tiempo únicos
time_steps = set()
for f in vtk_files:
    match = re.search(r"\.(\d{5})\.vtk$", f)
    if match:
        time_steps.add(match.group(1))
time_steps = sorted(list(time_steps))

# Elegimos el primer paso de tiempo para inspección
t = time_steps[0]
block_files = [f for f in vtk_files if f".{t}.vtk" in f]

print(f"Inspeccionando {len(block_files)} bloques para t = {t}\n")

for bf in block_files:
    path = os.path.join(data_dir, bf)
    mesh = pv.read(path)
    dims = mesh.dimensions
    bounds = mesh.bounds
    variable_names = mesh.array_names  # Aquí están los nombres de todas las variables

    print(f"{bf}")
    print(f"  dims: {dims}, bounds: {bounds}")
    print(f"  Variables disponibles: {variable_names}\n")
