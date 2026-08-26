#!/usr/bin/env python3
"""Create all-black masks for a verified instrument-free sequence."""

import argparse
from pathlib import Path

from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    images = sorted(args.images.glob("*.png"))
    if not images:
        parser.error(f"no PNG images found in {args.images}")
    args.output.mkdir(parents=True, exist_ok=True)
    existing = list(args.output.glob("*.png"))
    if existing:
        parser.error(f"refusing to overwrite {len(existing)} masks in {args.output}")

    for source in images:
        with Image.open(source) as image:
            Image.new("L", image.size, color=0).save(args.output / source.name)
    print(f"Created {len(images)} all-black masks in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
