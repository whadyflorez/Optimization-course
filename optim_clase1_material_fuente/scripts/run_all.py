"""Ejecuta todos los scripts de figuras de la Clase 1 en orden y reporta
si alguno falla. Uso:  python3 run_all.py
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
scripts = sorted(p for p in HERE.glob("*.py")
                  if p.name not in ("run_all.py", "_util.py"))

ok, fail = [], []
for s in scripts:
    print(f"\n=== Ejecutando {s.name} ===")
    r = subprocess.run([sys.executable, s.name], cwd=HERE)
    (ok if r.returncode == 0 else fail).append(s.name)

print("\n\nResumen:")
print("  OK :", ok)
print("  FAIL:", fail)
if fail:
    sys.exit(1)
