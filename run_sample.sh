#!/bin/sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python_bin="$repo_dir/.venv/bin/python"
video=${1:-"$repo_dir/data/sample/sample.mp4"}
dataset_dir="$repo_dir/data/sample"

if [ ! -x "$python_bin" ]; then
    echo "ERROR: $repo_dir/.venv is missing. Create it before running this script." >&2
    exit 2
fi
if [ ! -f "$video" ]; then
    echo "ERROR: sample video not found: $video" >&2
    echo "Place it at $repo_dir/data/sample/sample.mp4" >&2
    exit 2
fi

if ! find "$dataset_dir/images" -name '*.png' -type f -print -quit 2>/dev/null | grep -q .; then
    "$python_bin" "$repo_dir/scripts/preprocess_sample_video.py" \
        "$video" --output "$dataset_dir/images"
else
    echo "Using existing extracted frames in $dataset_dir/images"
fi

mkdir -p "$dataset_dir/gray_out"
"$python_bin" "$repo_dir/scripts/validate_sample_dataset.py" \
    "$dataset_dir" --expected-count 2117

if [ "$(uname -s)" = "Darwin" ]; then
    echo "ERROR: dataset validation passed, but EndoGaussian reconstruction cannot run on macOS." >&2
    echo "It requires NVIDIA CUDA and its two custom CUDA extensions; CPU and MPS are unsupported." >&2
    exit 4
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: an NVIDIA CUDA host is required for reconstruction." >&2
    exit 4
fi

cd "$repo_dir"
exec "$python_bin" train.py \
    -s "$dataset_dir" \
    --expname sample \
    --configs arguments/sample_mono.py
