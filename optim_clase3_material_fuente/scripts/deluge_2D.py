#deluge en 2D
import numpy as np
import matplotlib.pyplot as plt

def funcion_himmelblau(x):
    """
    Función objetivo de 2 variables: f(x1, x2)
    Mínimos globales en (3, 2), (-2.805, 3.131), (-3.779, -3.283) y (3.584, -1.848)
    donde f(x) = 0.
    """
    x1, x2 = x[0], x[1]
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

def gran_diluvio_2d(f_objetivo, x_inicial, max_iter=10000, max_costo_admisible=200.0):
    # 1. Configuración del estado inicial
    x_actual = np.array(x_inicial, dtype=float)
    costo_actual = f_objetivo(x_actual)
    
    # 2. Definir el nivel de agua inicial (L) por encima del costo inicial
    L = costo_actual + 20.0
    
    # 3. Calcular la tasa de descenso (UP) para llegar al costo deseado al final de las iteraciones
    # UP = (Nivel_Inicial - Nivel_Final) / Total_Iteraciones
    UP = (L - 0.0) / max_iter
    
    # Guardar el mejor resultado global
    x_mejor = np.copy(x_actual)
    costo_mejor = costo_actual

    # Bucle principal
    for i in range(max_iter):
        # Generar un vecino dando un paso aleatorio pequeño en 2D
        paso_2d = np.random.uniform(-0.3, 0.3, size=2)
        x_nuevo = x_actual + paso_2d
        
        costo_nuevo = f_objetivo(x_nuevo)

        # CRITERIO DEL GRAN DILUVIO:
        # Aceptamos la nueva posición 2D si NO supera el nivel de agua L
        if costo_nuevo <= L:
            x_actual = x_nuevo
            costo_actual = costo_nuevo
            
            # Actualizar el óptimo global si es la mejor solución vista
            if costo_nuevo < costo_mejor:
                x_mejor = np.copy(x_nuevo)
                costo_mejor = costo_nuevo

        # El nivel del agua desciende en cada paso
        L -= UP

    return x_mejor, costo_mejor

# --- EJECUCIÓN DEL ALGORITMO ---
# Punto de partida arbitrario en 2D
x0 = [-5.0, -5.0]

x_opt, f_opt = gran_diluvio_2d(
    f_objetivo=funcion_himmelblau, 
    x_inicial=x0, 
    max_iter=15000
)

print(f"Punto inicial x0: {x0}")
print(f"Mejor punto [x1, x2] encontrado: [{x_opt[0]:.4f}, {x_opt[1]:.4f}]")
print(f"Valor de la función f(x1, x2): {f_opt:.6f}")

# --- BLOQUE DE GRAFICACIÓN (Adicionar al final del código) ---
# 1. Crear la malla para graficar el espacio de búsqueda 2D
x1_range = np.linspace(-5, 5, 100)
x2_range = np.linspace(-5, 5, 100)
X1, X2 = np.meshgrid(x1_range, x2_range)
Z = (X1**2 + X2 - 11)**2 + (X1 + X2**2 - 7)**2

# 2. Configurar la figura con 2 subgráficos (Superficie 3D y Mapa de Contornos)
fig = plt.figure(figsize=(14, 6))

# Gráfico 1: Superficie 3D
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax1.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8, edgecolor='none')
ax1.scatter(x_opt[0], x_opt[1], f_opt, color='red', s=100, label='Mínimo Encontrado', zorder=5)
ax1.set_title("Superficie 3D de Himmelblau")
ax1.set_xlabel("x1")
ax1.set_ylabel("x2")
ax1.set_zlabel("f(x1, x2)")
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)

# Gráfico 2: Curvas de Nivel (Contorno 2D)
ax2 = fig.add_subplot(1, 2, 2)
contour = ax2.contour(X1, X2, Z, levels=np.logspace(0, 3, 20), cmap='viridis')
ax2.plot(x0[0], x0[1], 'bo', markersize=8, label='Inicio (x0)')
ax2.plot(x_opt[0], x_opt[1], 'r*', markersize=15, label='Mínimo Encontrado')
ax2.set_title("Curvas de Nivel y Solución")
ax2.set_xlabel("x1")
ax2.set_ylabel("x2")
ax2.grid(True)
ax2.legend()
fig.colorbar(contour, ax=ax2)

plt.tight_layout()
plt.show()