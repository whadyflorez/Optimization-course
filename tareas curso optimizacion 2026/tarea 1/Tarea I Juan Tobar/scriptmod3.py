import numpy as np
import matplotlib.pyplot as plt

# Función
f = lambda x, y: np.sin(x) * np.cos(y)

# Gradiente
grad = lambda x, y: (
    np.cos(x) * np.cos(y),
    -np.sin(x) * np.sin(y)
)

# Dominio
x = np.linspace(-2*np.pi, 2*np.pi, 300)
y = np.linspace(-2*np.pi, 2*np.pi, 300)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Malla más gruesa para dibujar los vectores gradiente
xg = np.linspace(-2*np.pi, 2*np.pi, 17)
yg = np.linspace(-2*np.pi, 2*np.pi, 17)

XG, YG = np.meshgrid(xg, yg)

U, V = grad(XG, YG)

# Crear figura
fig, ax = plt.subplots(figsize=(7, 6))

# Curvas de nivel
cs = ax.contour(
    X,
    Y,
    Z,
    levels=12,
    cmap="viridis"
)

ax.clabel(
    cs,
    inline=True,
    fontsize=8,
    fmt="%.1f"
)

# Magnitud del gradiente
mag = np.sqrt(U**2 + V**2)

# Copia para evitar división entre cero
mag_safe = mag.copy()
mag_safe[mag_safe == 0] = 1

# Campo gradiente
ax.quiver(
    XG,
    YG,
    U / mag_safe,
    V / mag_safe,
    mag,
    cmap="autumn",
    scale=22,
    width=0.006
)

# Puntos de silla:
# x = k*pi
# y = pi/2 + m*pi

saddle_x = []
saddle_y = []

for k in range(-2, 3):
    for m in range(-2, 2):

        xs = k * np.pi
        ys = np.pi/2 + m*np.pi

        if (
            -2*np.pi <= xs <= 2*np.pi
            and -2*np.pi <= ys <= 2*np.pi
        ):
            saddle_x.append(xs)
            saddle_y.append(ys)

# Marcar puntos de silla
ax.scatter(
    saddle_x,
    saddle_y,
    marker="x",
    s=60,
    label="Puntos de silla"
)

# Etiquetas
ax.set_xlabel("x")
ax.set_ylabel("y")

ax.set_title(
    r"Curvas de nivel de $f(x,y)=\sin(x)\cos(y)$"
    "\n"
    r"y campo gradiente $\nabla f$"
)

ax.set_aspect("equal")

ax.legend()

fig.tight_layout()

# Guardar la imagen directamente
fig.savefig(
    "fig03_curvas_nivel_gradiente.png",
    dpi=150
)

print("Gráfica guardada como fig03_curvas_nivel_gradiente.png")

# Mostrar gráfica
plt.show()
