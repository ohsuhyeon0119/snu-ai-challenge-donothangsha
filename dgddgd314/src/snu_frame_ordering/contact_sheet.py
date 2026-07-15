from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def make_contact_sheet(image_paths, labels=None, size=448, label_height=28):
    """Build one labeled 2x2 image for single-image VLMs such as PaliGemma."""
    labels = labels or [f"Image {i}" for i in range(1, 5)]

    cell = size // 2
    sheet = Image.new("RGB", (cell * 2, cell * 2), "white")
    font = ImageFont.load_default()

    for idx, raw_path in enumerate(image_paths):
        path = Path(raw_path)
        with Image.open(path) as img:
            rgb = ImageOps.exif_transpose(img).convert("RGB")
            panel = Image.new("RGB", (cell, cell), "white")
            image_area = (cell, cell - label_height)
            fitted = ImageOps.contain(rgb, image_area, method=Image.Resampling.BICUBIC)
            x = (cell - fitted.width) // 2
            y = label_height + (image_area[1] - fitted.height) // 2
            panel.paste(fitted, (x, y))

        panel_draw = ImageDraw.Draw(panel)
        panel_draw.rectangle((0, 0, cell, label_height), fill="white")
        panel_draw.rectangle((0, 0, cell - 1, cell - 1), outline="black", width=2)
        panel_draw.text((8, 8), labels[idx], fill="black", font=font)

        sx = (idx % 2) * cell
        sy = (idx // 2) * cell
        sheet.paste(panel, (sx, sy))

    draw = ImageDraw.Draw(sheet)
    draw.line((cell, 0, cell, cell * 2), fill="black", width=2)
    draw.line((0, cell, cell * 2, cell), fill="black", width=2)
    return sheet


def save_contact_sheet(image_paths, out_path, labels=None, size=448):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet = make_contact_sheet(image_paths, labels=labels, size=size)
    sheet.save(out_path)
    return out_path
