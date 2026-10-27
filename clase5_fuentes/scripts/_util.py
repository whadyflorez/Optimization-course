"""Utilidades compartidas: rutas de figuras."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR = os.path.join(HERE, "..", "figs")
os.makedirs(FIGS_DIR, exist_ok=True)


def figpath(name):
    return os.path.join(FIGS_DIR, name)
