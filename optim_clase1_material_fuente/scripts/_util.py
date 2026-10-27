"""Utilidades comunes para los scripts de la Clase 1 (Optimizacion).
Cada script de esta carpeta es independiente y corto; este modulo solo evita
repetir 3 lineas de configuracion de carpeta de salida en cada uno.
"""
import os

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs")
os.makedirs(FIGDIR, exist_ok=True)


def savepath(name):
    return os.path.join(FIGDIR, name)
