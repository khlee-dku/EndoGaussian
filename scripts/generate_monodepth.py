#!/usr/bin/env python3
"""Generate EndoGaussian-compatible grayscale depth with Depth Anything V2."""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def select_device(torch, requested: str):
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--depth-anything-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder", choices=("vits", "vitb", "vitl", "vitg"), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--limit", type=int, help="process only the first N frames for a smoke test")
    args = parser.parse_args()

    sys.path.insert(0, str(args.depth_anything_root.resolve()))
    import cv2
    import torch
    from depth_anything_v2.dpt import DepthAnythingV2

    configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }
    images = sorted(args.images.glob("*.png"))
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        parser.error(f"no input PNGs found in {args.images}")
    args.output.mkdir(parents=True, exist_ok=True)
    collisions = [path.name for path in images if (args.output / path.name).exists()]
    if collisions:
        parser.error(f"refusing to overwrite existing output: {collisions[0]}")

    device = select_device(torch, args.device)
    model = DepthAnythingV2(**configs[args.encoder])
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model = model.to(device).eval()
    print(f"Depth Anything V2 encoder={args.encoder}, device={device}, frames={len(images)}")

    for index, source in enumerate(images, start=1):
        bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cannot read {source}")
        depth = model.infer_image(bgr, args.input_size)
        span = float(depth.max() - depth.min())
        if span <= 0:
            raise RuntimeError(f"constant depth prediction for {source}")
        depth = ((depth - depth.min()) / span * 255.0).astype(np.uint8)
        depth_rgb = np.repeat(depth[..., None], 3, axis=2)
        Image.fromarray(depth_rgb, mode="RGB").save(args.output / source.name)
        print(f"[{index}/{len(images)}] {source.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
