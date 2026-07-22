import ast
import json
from pathlib import Path


path = Path("ITC6110-Project-Spring-2026.ipynb")
notebook = json.loads(path.read_text(encoding="utf-8"))

print("Notebook:", path)
print("Kernel:", notebook.get("metadata", {}).get("kernelspec"))
print("Cells:", len(notebook["cells"]))

print("\nSTORED ERRORS")
found = False
for index, cell in enumerate(notebook["cells"]):
    for output in cell.get("outputs", []):
        if output.get("output_type") == "error":
            found = True
            print(f"CELL {index} execution_count={cell.get('execution_count')}")
            print(output.get("ename"), output.get("evalue"))
            print("\n".join(output.get("traceback", []))[-4000:])
if not found:
    print("None")

print("\nSYNTAX CHECK")
for index, cell in enumerate(notebook["cells"]):
    if cell.get("cell_type") != "code":
        continue
    source = "".join(cell.get("source", []))
    if source.lstrip().startswith(("%%", "%", "!")):
        continue
    try:
        ast.parse(source)
    except SyntaxError as error:
        print(f"CELL {index}: {error.msg} line {error.lineno}: {error.text!r}")

print("\nEXECUTION COUNTS")
counts = [
    (index, cell.get("execution_count"))
    for index, cell in enumerate(notebook["cells"])
    if cell.get("cell_type") == "code"
]
print(counts)
