import numpy as np

def funcion_himmelblau(x):
    """Mínimos globales f(x) = 0 en: (3, 2), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)"""
    x1, x2 = x[:, 0], x[:, 1]
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

def gran_diluvio_poblacional(f_objetivo, num_agentes=30, max_iter=8000, bounds=(-5, 5)):
    # 1. Inicializar población aleatoria de agentes en el espacio 2D
    poblacion = np.random.uniform(bounds[0], bounds[1], size=(num_agentes, 2))
    costos = f_objetivo(poblacion)
    
    # 2. Definir nivel de agua inicial (L) por encima del máximo costo actual
    L = np.max(costos) + 10.0
    
    # Tasa de descenso uniforme del agua
    UP = (L - 0.0) / max_iter

    # 3. Bucle evolutivo / paralelo
    for i in range(max_iter):
        # Cada agente genera una propuesta de paso independiente
        pasos = np.random.uniform(-0.3, 0.3, size=(num_agentes, 2))
        propuestas = poblacion + pasos
        
        # Mantener los agentes dentro de los límites de búsqueda
        propuestas = np.clip(propuestas, bounds[0], bounds[1])
        
        # Evaluar toda la población de forma vectorizada
        costos_propuestos = f_objetivo(propuestas)
        
        # CRITERIO DEL DILU VIO VECTORIAL:
        # Cada agente acepta si su nuevo costo es menor o igual al nivel de agua L
        aceptados = costos_propuestos <= L
        
        # Actualizar solo los agentes que superaron la prueba de agua
        poblacion[aceptados] = propuestas[aceptados]
        costos[aceptados] = costos_propuestos[aceptados]

        # El nivel de agua desciende para todos los agentes por igual
        L -= UP

    return poblacion, costos

# --- EJECUCIÓN DEL ALGORITMO POBLACIONAL ---
np.random.seed(42)
agentes_finales, costos_finales = gran_diluvio_poblacional(
    funcion_himmelblau, 
    num_agentes=1000, 
    max_iter=10000
)

# Agrupar las soluciones encontradas por los agentes para filtrar los distintos mínimos
minimos_unicos = []
for p, c in zip(agentes_finales, costos_finales):
    if c < 1e-2:  # Consideramos óptimo válido
        p_redondeado = np.round(p, 3)
        if not any(np.allclose(p_redondeado, m, atol=0.2) for m in minimos_unicos):
            minimos_unicos.append(p_redondeado)

print(f"Total de mínimos globales distintos identificados: {len(minimos_unicos)}")
print("Coordenadas [x1, x2] de los mínimos encontrados:")
for idx, m in enumerate(minimos_unicos, 1):
    print(f" Mínimo {idx}: [{m[0]:.3f}, {m[1]:.3f}]")