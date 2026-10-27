"""
GENERADOR DE DATOS DEL MINIPROYECTO  (SOLO PROFESOR)

Genera, para cada estudiante, un archivo  datos_<id>.csv  (columnas t, y) con
datos sinteticos del modelo biexponencial

        y(t) = a*exp(-b*t) + c*exp(-d*t)  +  ruido gaussiano N(0, sigma^2)

La semilla es el identificador del estudiante (p. ej. la cedula), asi cada
estudiante recibe datos distintos pero con la misma dificultad. Los parametros
verdaderos se obtienen perturbando +-10 % los valores base y se guardan en
'verdad_parametros.csv' (NO se entrega a los estudiantes).

Uso:
    python generar_datos.py 1017123456 1020333444        # varios estudiantes
    python generar_datos.py --lista cedulas.txt          # una cedula por linea
    python generar_datos.py 2026 --sigma 0.1 --n 100     # ejemplo de prueba
"""
import argparse
import csv
import os
import numpy as np

BASE = np.array([2.0, 1.5, 3.0, 0.15])      # a, b, c, d  (b >> d: dos escalas de tiempo)


def modelo(th, t):
    a, b, c, d = th
    return a * np.exp(-b * t) + c * np.exp(-d * t)


def generar(id_est, n=150, sigma=0.05, t_max=20.0):
    rng = np.random.default_rng(int(id_est))
    th_true = BASE * (1 + 0.10 * rng.uniform(-1, 1, 4))
    t = np.sort(rng.uniform(0, t_max, n))
    t[0] = 0.0
    y = modelo(th_true, t) + sigma * rng.standard_normal(n)
    return t, y, th_true


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", help="identificadores (semillas)")
    ap.add_argument("--lista", help="archivo de texto con un id por linea")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--salida", default="datos_estudiantes")
    args = ap.parse_args()

    ids = list(args.ids)
    if args.lista:
        ids += [l.strip() for l in open(args.lista) if l.strip()]
    if not ids:
        ap.error("indica al menos un identificador")
    os.makedirs(args.salida, exist_ok=True)

    filas = []
    for i in ids:
        t, y, th = generar(i, n=args.n, sigma=args.sigma)
        with open(os.path.join(args.salida, f"datos_{i}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t", "y"])
            w.writerows(zip(t, y))
        filas.append([i, *th, args.sigma])
    with open(os.path.join(args.salida, "verdad_parametros.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "a", "b", "c", "d", "sigma"])
        w.writerows(filas)
    print(f"{len(ids)} archivos en '{args.salida}/'  (verdad_parametros.csv: SOLO PROFESOR)")
