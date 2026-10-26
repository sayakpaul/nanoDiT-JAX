"""Animate saved sample grids with the original nanoDiT layout and frame rate."""
import argparse
import json
from pathlib import Path

from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--output", type=Path, default=Path("assets/output.gif"))
    p.add_argument("--ema", action="store_true", help="Use EMA grids instead of live weights")
    args = p.parse_args()
    paths = sorted(args.run_dir.glob("samples_ema_*.png" if args.ema else "samples_[0-9]*.png"))
    if not paths:
        p.error("No sample grids found")
    size = json.loads((args.run_dir / "config.json").read_text())["image_size"]
    frames = []
    for path in paths:
        with Image.open(path) as image:
            cols, rows = image.width // size, image.height // size
            grid = Image.new("RGB", (cols * size + (cols + 1) * 2, rows * size + (rows + 1) * 2))
            for row in range(rows):
                for col in range(cols):
                    tile = image.crop((col * size, row * size, (col + 1) * size, (row + 1) * size))
                    grid.paste(tile, (2 + col * (size + 2), 2 + row * (size + 2)))
            frames.append(grid.quantize(colors=256, dither=Image.Dither.NONE))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output, save_all=True, append_images=frames[1:], loop=0,
                   duration=[200] * (len(frames) - 1) + [1000], disposal=2, optimize=False)
    print(f"Saved {args.output}: {len(frames)} frames, {frames[0].size}")


if __name__ == "__main__":
    main()
