import json
from io import StringIO

import pandas as pd


with open("ITC6110-Project-Spring-2026.ipynb", encoding="utf-8") as handle:
    notebook = json.load(handle)

for cell_index in [7, 8, 9, 10, 14, 16, 18, 23, 60, 65, 67, 69, 71, 72, 73, 75, 81, 83, 88, 90, 96, 97, 102, 103]:
    print(f"\n### CELL {cell_index}")
    for output in notebook["cells"][cell_index].get("outputs", []):
        if "text" in output:
            print("".join(output["text"])[:10000])
        data = output.get("data", {})
        if "text/plain" in data:
            print("".join(data["text/plain"])[:10000])
        if cell_index == 102 and "text/html" in data:
            table = pd.read_html(StringIO("".join(data["text/html"])))[0]
            print(table.to_string(index=False))
