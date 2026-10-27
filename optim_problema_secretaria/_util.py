"""Utilidad común: dónde guardar las figuras."""
import os

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs")
os.makedirs(FIGDIR, exist_ok=True)


def savepath(name):
    return os.path.join(FIGDIR, name)
