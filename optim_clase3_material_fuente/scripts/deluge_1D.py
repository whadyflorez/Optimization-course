#algoritmo del diluvio 1-D GUnter Duek 1993
import numpy as np

def funcion_objetivo(x):
    """Función a minimizar. Mínimo global en x = 0."""
    return x**2

def gran_diluvio(costo_inicial, x_inicial, pasadas=1000, tasa_descenso=0.01):
    # 1. Estado inicial
    x_actual = x_inicial
    costo_actual = funcion_objetivo(x_actual)
    
    # 2. El nivel de agua inicial L se define por encima del costo inicial
    L = costo_actual + 10.0  
    
    # Tasa constante a la que se reduce el nivel del agua
    UP = tasa_descenso  
    
    x_mejor = x_actual
    costo_mejor = costo_actual

    for _ in range(pasadas):
        # Generar un vecino (paso aleatorio pequeño)
        x_nuevo = x_actual + np.random.uniform(-0.5, 0.5)
        costo_nuevo = funcion_objetivo(x_nuevo)

        # CRITERIO DEL GRAN DILUVIO:
        # Aceptamos si el nuevo costo NO supera el nivel de agua L
        if costo_nuevo <= L:
            x_actual = x_nuevo
            costo_actual = costo_nuevo
            
            # Guardar la mejor solución global encontrada
            if costo_nuevo < costo_mejor:
                x_mejor = x_nuevo
                costo_mejor = costo_nuevo

        # El nivel de agua BAJA en cada iteración
        L = L - UP

    return x_mejor, costo_mejor

# Ejemplo de uso
x_opt, f_opt = gran_diluvio(costo_inicial=100, x_inicial=10.0)
print(f"Mejor x encontrado: {x_opt:.4f}")
print(f"Costo mínimo: {f_opt:.4f}")
