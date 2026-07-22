import base64
import json
from pathlib import Path


NOTEBOOK = Path("ITC6110-Project-Spring-2026.ipynb")
OUT = Path("tmp/notebook")
OUT.mkdir(parents=True, exist_ok=True)

nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
lines = []

for index, cell in enumerate(nb["cells"]):
    source = "".join(cell.get("source", []))
    lines.append(f"\n===== CELL {index} ({cell['cell_type']}) =====\n{source}\n")
    for output_index, output in enumerate(cell.get("outputs", [])):
        lines.append(f"--- OUTPUT {output_index}: {output.get('output_type')} ---\n")
        if "text" in output:
            text = output["text"]
            lines.append("".join(text) if isinstance(text, list) else str(text))
        data = output.get("data", {})
        for mime in ("text/plain", "text/html"):
            if mime in data:
                value = data[mime]
                lines.append("".join(value) if isinstance(value, list) else str(value))
                lines.append("\n")
        if "image/png" in data:
            image_data = data["image/png"]
            if isinstance(image_data, list):
                image_data = "".join(image_data)
            image_path = OUT / f"cell-{index:03d}-output-{output_index}.png"
            image_path.write_bytes(base64.b64decode(image_data))
            lines.append(f"[IMAGE: {image_path}]\n")

(OUT / "notebook_evidence.txt").write_text("".join(lines), encoding="utf-8")
print(f"Extracted evidence and images to {OUT}")
