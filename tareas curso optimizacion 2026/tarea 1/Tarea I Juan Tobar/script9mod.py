import numpy as np
import matplotlib.pyplot as plt

# Función solicitada
f = lambda x: x**4 - 5*x**2 + 4

# Segunda derivada
fsecond = lambda x: 12*x**2 - 10

# Raíces de f'(x) = 4x^3 - 10x = 0
roots = np.roots([4, 0, -10, 0])
roots = np.sort(roots[np.isreal(roots)].real)

# Dominio de la gráfica
x = np.linspace(-2.6, 2.6, 400)

fig, ax = plt.subplots(figsize=(7, 4.8))
ax.plot(x, f(x), lw=2.5)

# Clasificar los puntos críticos
labels = []

for r in roots:
    curv = fsecond(r)

    if curv > 0:
        kind = "mínimo"
    elif curv < 0:
        kind = "máximo"
    else:
        kind = "no concluyente"

    labels.append((r, f(r), kind))

# Encontrar el menor valor entre los mínimos
minimos = [(r, val) for r, val, k in labels if k == "mínimo"]
valor_global = min(val for r, val in minimos)

# Dibujar puntos críticos
for r, val, kind in labels:

    es_global = (
        kind == "mínimo"
        and np.isclose(val, valor_global)
    )

    if es_global:
        etiqueta = "mínimo global"
    elif kind == "mínimo":
        etiqueta = "mínimo local"
    else:
        etiqueta = "máximo local"

    ax.plot(r, val, "o", ms=9)

    ax.annotate(
        f"{etiqueta}\n({r:.2f}, {val:.2f})",
        (r, val),
        textcoords="offset points",
        xytext=(0, 14 if kind == "máximo" else -35),
        ha="center",
        fontsize=9
    )

ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title(r"$f(x)=x^4-5x^2+4$: mínimos y máximo")
ax.grid(alpha=0.3)

fig.tight_layout()

# Guardar imagen en la misma carpeta del script
fig.savefig("fig09_min_local_global.png", dpi=150)

print("Puntos críticos:")
for r, val, kind in labels:
    print(f"x = {r:.4f}, f(x) = {val:.4f} -> {kind}")

print("\nLa gráfica se guardó como fig09_min_local_global.png")

plt.show()
