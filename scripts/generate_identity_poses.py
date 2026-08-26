#!/usr/bin/env python3
"""Generate an explicitly synthetic LLFF identity-pose smoke-test baseline."""

import argparse
from pathlib import Path

import numpy as np


WARNING = (
    "This poses_bounds.npy uses identity camera poses and synthetic 90-degree-FOV "
    "intrinsics solely for pipeline smoke testing. It is NOT calibrated and must "
    "NOT be used to evaluate reconstruction geometry or accuracy."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--height", type=float, required=True)
    parser.add_argument("--width", type=float, required=True)
    parser.add_argument("--focal", type=float, required=True)
    parser.add_argument("--near", type=float, required=True)
    parser.add_argument("--far", type=float, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    if args.count <= 0 or args.focal <= 0 or not 0 <= args.near < args.far:
        parser.error("count/focal must be positive and bounds must satisfy 0 <= near < far")

    pose = np.concatenate(
        [np.eye(3, 4, dtype=np.float64), np.array([[args.height], [args.width], [args.focal]])],
        axis=1,
    )
    row = np.concatenate([pose.reshape(15), [args.near, args.far]])
    poses = np.repeat(row[None, :], args.count, axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, poses)
    args.output.with_suffix(".SMOKE_TEST_ONLY.txt").write_text(WARNING + "\n", encoding="utf-8")
    print(WARNING)
    print(f"Wrote {args.output}: shape={poses.shape}, dtype={poses.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
