#!/usr/bin/env python3
"""Validate an EndoNeRF-style monocular dataset without importing CUDA code."""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def png_map(path: Path):
    return {item.name: item for item in sorted(path.glob("*.png"))} if path.is_dir() else {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args()
    root = args.dataset.resolve()
    problems = []
    images = png_map(root / "images")
    depths = png_map(root / "monodepth")
    masks = png_map(root / "masks")

    print(f"RGB/depth/mask counts: {len(images)}/{len(depths)}/{len(masks)}")
    if args.expected_count is not None and len(images) != args.expected_count:
        problems.append(f"RGB count is {len(images)}; expected {args.expected_count}")
    if not images:
        problems.append(f"no RGB PNGs found in {root / 'images'}")
        expected_size = None
    else:
        with Image.open(next(iter(images.values()))) as image:
            expected_size = image.size
        print(f"RGB dimensions: {expected_size[0]}x{expected_size[1]}")

    for label, files in (("monodepth", depths), ("masks", masks)):
        missing = sorted(set(images) - set(files))
        extra = sorted(set(files) - set(images))
        if missing or extra:
            problems.append(f"{label} basenames differ: {len(missing)} missing, {len(extra)} extra")

    for path in depths.values():
        with Image.open(path) as depth:
            array = np.asarray(depth)
            if expected_size and depth.size != expected_size:
                problems.append(f"{path}: size {depth.size}, expected {expected_size}")
            if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
                problems.append(f"{path}: expected 3-channel uint8 PNG, got {array.dtype} {array.shape}")
            elif not (np.array_equal(array[..., 0], array[..., 1]) and np.array_equal(array[..., 1], array[..., 2])):
                problems.append(f"{path}: RGB depth channels are not identical")

    for path in masks.values():
        with Image.open(path) as mask:
            array = np.asarray(mask)
            if expected_size and mask.size != expected_size:
                problems.append(f"{path}: size {mask.size}, expected {expected_size}")
            values = np.unique(array)
            if array.dtype != np.uint8 or not set(values.tolist()).issubset({0, 255}):
                problems.append(f"{path}: mask must be uint8 with values only 0/255; got {values}")

    poses_path = root / "poses_bounds.npy"
    if not poses_path.is_file():
        problems.append(f"missing camera poses/intrinsics: {poses_path}")
    else:
        try:
            poses = np.load(poses_path, allow_pickle=False)
            expected_shape = (len(images), 17)
            if poses.shape != expected_shape:
                problems.append(f"{poses_path}: shape {poses.shape}, expected {expected_shape}")
            elif expected_size:
                blocks = poses[:, :15].reshape(-1, 3, 5)
                if not np.all(blocks[:, 0, 4] == expected_size[1]):
                    problems.append(f"{poses_path}: encoded H does not equal {expected_size[1]}")
                if not np.all(blocks[:, 1, 4] == expected_size[0]):
                    problems.append(f"{poses_path}: encoded W does not equal {expected_size[0]}")
                focal = blocks[:, 2, 4]
                if not np.all(np.isfinite(focal)) or not np.all(focal > 0):
                    problems.append(f"{poses_path}: focal lengths must be finite and positive")
                bounds = poses[:, 15:17]
                if not np.all(np.isfinite(bounds)) or not np.all(bounds[:, 0] < bounds[:, 1]):
                    problems.append(f"{poses_path}: near/far values must be finite with near < far")
                print(f"poses_bounds shape: {poses.shape}")
        except Exception as exc:
            problems.append(f"cannot read {poses_path}: {exc}")

    if problems:
        print("Dataset is not reconstruction-ready:", file=sys.stderr)
        for problem in problems[:30]:
            print(f"  - {problem}", file=sys.stderr)
        if len(problems) > 30:
            print(f"  - ... {len(problems) - 30} more", file=sys.stderr)
        return 3
    print("Dataset structure is valid for the monocular EndoNeRF loader.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
