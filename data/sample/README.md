# Sample video input

Place the unmodified video at:

```text
data/sample/sample.mp4
```

`run_sample.sh` extracts every decoded frame, at its original resolution and
without temporal sampling, into `data/sample/images/`. It also writes
`video_manifest.json`. Existing PNG frames are never overwritten.

EndoGaussian cannot reconstruct from RGB frames alone. Before training, this
directory must also contain:

```text
data/sample/
  images/             # N RGB PNGs, sorted by filename
  monodepth/           # N RGB/RGBA monocular depth PNGs, matching image order
  masks/               # N masks; white = instrument/excluded, black = tissue
  poses_bounds.npy     # shape (N, 17), LLFF/EndoNeRF format
```

Each row of `poses_bounds.npy` is the flattened 3x5 camera block followed by
near and far bounds. The first 3x4 values are a camera-to-world pose; the last
column is `[height, width, focal_length]`. These values must be estimated or
measured for the actual camera/video and must remain consistent with any
resizing. The repository does not estimate them. The original EndoNeRF data
preparation pipeline is the compatible source of this format.

The monocular loader reads the first channel of each depth PNG, so grayscale
PNG files are not accepted without adapting the loader. `run_sample.sh` creates
the loader's `gray_out/` diagnostic-output directory automatically.

The EndoNeRF loader now derives width and height from the first RGB frame. The
extractor preserves the source resolution because resizing before calibration
would invalidate the intrinsics. `H`, `W`, and focal length encoded in
`poses_bounds.npy` must describe that same resolution.

For an instrument-free sequence, create structurally required all-black masks:

```bash
python scripts/generate_black_masks.py data/sample/images data/sample/masks
```

Depth Anything V2 is kept as an external dependency. After cloning its official
repository and downloading the matching checkpoint, run a small test first:

```bash
python scripts/generate_monodepth.py data/sample/images /tmp/sample-depth \
  --depth-anything-root /path/to/Depth-Anything-V2 \
  --checkpoint /path/to/depth_anything_v2_vits.pth --encoder vits --limit 3
```

Omit `--limit 3` and change the output to `data/sample/monodepth` only after the
sample outputs have been checked.

## SMOKE-TEST-ONLY camera baseline

This poses_bounds.npy uses identity camera poses and synthetic 90-degree-FOV
intrinsics solely for pipeline smoke testing. It is NOT calibrated and must NOT
be used to evaluate reconstruction geometry or accuracy.

The baseline uses 598x598 images, focal length 299 px, centered principal point,
identity camera-to-world transforms, and near/far placeholders 0.1/10.0. The
current EndoNeRF loader does not consume the two bounds.
