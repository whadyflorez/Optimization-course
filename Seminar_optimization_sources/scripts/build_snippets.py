"""Extract the '# --8<-- name ... # --8<-- end' blocks of methods.py (and problems.py) into snippets/."""
import os, re
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "snippets"); os.makedirs(OUT, exist_ok=True)
text = open(os.path.join(HERE, "methods.py")).read()
for m in re.finditer(r"# --8<-- (\w+)\n(.*?)# --8<-- end", text, re.S):
    name, body = m.group(1), m.group(2).rstrip() + "\n"
    open(os.path.join(OUT, name + ".py"), "w").write(body)
    print(name, body.count("\n"), "lines")
