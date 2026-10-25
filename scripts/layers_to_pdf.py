"""
Aim: Print the layers made by another script.

This script: Put every layer_N.png from a script's output folder into a PDF,
layers_per_page to a page, ready to print.

1. Load the layers, in order.
2. Pick the grid (columns and rows), and whether to turn the layers on their
   side, that fits layers_per_page on a page as large as possible, with
   margin_mm around the page and gap_mm between layers. Every page uses the
   same grid, so the layers all print the same size and still stack.
3. Paste them onto white pages and save them as one PDF.
"""

import math
from pathlib import Path

from PIL import Image

input_folder = Path("data/output/blob_levels3")
layers_per_page = 1
page_mm = (297, 210)  # A4 landscape
margin_mm = 20
gap_mm = 5
dpi = 300

output_folder = Path("data/output/layers_to_pdf")


def main():
    layers = load_layers()
    pages = [
        lay_out(layers[i : i + layers_per_page])
        for i in range(0, len(layers), layers_per_page)
    ]
    output_folder.mkdir(parents=True, exist_ok=True)
    pdf_file = output_folder / f"{input_folder.name}_layers.pdf"
    # The PDF stores each page as a JPEG, so keep it high quality for printing
    pages[0].save(
        pdf_file,
        save_all=True,
        append_images=pages[1:],
        resolution=dpi,
        quality=95,
        subsampling=0,
    )
    print(f"Saved {len(layers)} layers on {len(pages)} pages to {pdf_file}")


def load_layers() -> list[Image.Image]:
    files = sorted(
        input_folder.glob("layer_*.png"), key=lambda f: int(f.stem.split("_")[1])
    )
    assert files, f"No layer_*.png files in {input_folder}"
    return [Image.open(f).convert("RGB") for f in files]


def lay_out(layers: list[Image.Image]) -> Image.Image:
    page_w, page_h = (to_px(mm) for mm in page_mm)
    margin, gap = to_px(margin_mm), to_px(gap_mm)
    best = None
    for rotate in (False, True):
        layer_w, layer_h = layers[0].size[::-1] if rotate else layers[0].size
        for cols in range(1, layers_per_page + 1):
            rows = math.ceil(layers_per_page / cols)
            cell_w = (page_w - 2 * margin - (cols - 1) * gap) / cols
            cell_h = (page_h - 2 * margin - (rows - 1) * gap) / rows
            scale = min(cell_w / layer_w, cell_h / layer_h)
            if best is None or scale > best[0]:
                best = (scale, layer_w, layer_h, cols, rows, rotate)
    scale, layer_w, layer_h, cols, rows, rotate = best
    if rotate:
        layers = [layer.rotate(90, expand=True) for layer in layers]

    tile_w, tile_h = int(layer_w * scale), int(layer_h * scale)
    # Centre the grid on the page
    left = (page_w - cols * tile_w - (cols - 1) * gap) // 2
    top = (page_h - rows * tile_h - (rows - 1) * gap) // 2
    page = Image.new("RGB", (page_w, page_h), "white")
    for i, layer in enumerate(layers):
        row, col = divmod(i, cols)
        tile = layer.resize((tile_w, tile_h), Image.LANCZOS)
        page.paste(tile, (left + col * (tile_w + gap), top + row * (tile_h + gap)))
    return page


def to_px(mm: float) -> int:
    return round(mm / 25.4 * dpi)


if __name__ == "__main__":
    main()
