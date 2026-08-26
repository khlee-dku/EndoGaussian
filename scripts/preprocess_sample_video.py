#!/usr/bin/env python3
"""Extract every video frame as a lossless, full-resolution PNG."""

import argparse
import json
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    video = args.video.resolve()
    output = args.output.resolve()
    if not video.is_file():
        print(f"ERROR: video not found: {video}", file=sys.stderr)
        return 2
    output.mkdir(parents=True, exist_ok=True)
    existing = sorted(output.glob("*.png"))
    if existing:
        print(
            f"ERROR: {output} already contains {len(existing)} PNG files; "
            "refusing to overwrite or mix frame sets.",
            file=sys.stderr,
        )
        return 2

    reader = imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24")
    metadata = next(reader)
    width, height = metadata["size"]
    count = 0
    try:
        for count, frame_bytes in enumerate(reader, start=1):
            frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(height, width, 3)
            Image.fromarray(frame, "RGB").save(output / f"{count - 1:06d}.png")
    except Exception:
        print(f"ERROR: extraction stopped after {count} frames", file=sys.stderr)
        raise

    manifest = {
        "source": str(video),
        "frame_count": count,
        "width": width,
        "height": height,
        "fps": metadata.get("fps"),
        "duration_seconds": metadata.get("duration"),
        "sampling": "all decoded frames; no temporal sampling",
        "resize": "none",
    }
    (output.parent / "video_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Extracted {count} frames at {width}x{height} into {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
