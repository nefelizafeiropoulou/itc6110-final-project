from pathlib import Path
from PIL import Image, ImageDraw


render_dir = Path("tmp/report_render")
pages = sorted(render_dir.glob("page-*.png"), key=lambda p: int(p.stem.split("-")[1]))
thumb_w, thumb_h = 360, 466
for sheet_index in range(0, len(pages), 6):
    subset = pages[sheet_index:sheet_index + 6]
    sheet = Image.new("RGB", (thumb_w * 3, (thumb_h + 30) * 2), "#d8dde5")
    draw = ImageDraw.Draw(sheet)
    for local_index, path in enumerate(subset):
        page = Image.open(path).convert("RGB")
        page.thumbnail((thumb_w - 16, thumb_h - 16))
        x = (local_index % 3) * thumb_w + (thumb_w - page.width) // 2
        y = (local_index // 3) * (thumb_h + 30) + 8
        sheet.paste(page, (x, y))
        draw.text((x, y + page.height + 3), path.stem, fill="#111827")
    sheet.save(render_dir / f"contact-{sheet_index // 6 + 1}.png")
