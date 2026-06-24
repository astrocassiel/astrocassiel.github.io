#!/usr/bin/env python3
"""Convert raw Bayer uint16 .bin image files to MATLAB .mat format."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from scipy.io import savemat


def infer_dimensions(num_pixels: int) -> tuple[int, int]:
    """Infer width/height from pixel count using common camera resolutions."""
    candidates: list[tuple[int, int]] = [
        (640, 480),
        (800, 600),
        (1024, 768),
        (1280, 720),
        (1280, 1024),
        (1600, 1200),
        (1920, 1080),
        (2048, 1536),
        (2048, 2048),
        (2448, 2048),
        (2592, 1944),
        (4096, 3000),
    ]

    matches = [(w, h) for w, h in candidates if w * h == num_pixels]
    if len(matches) == 1:
        return matches[0]

    side = int(np.sqrt(num_pixels))
    if side * side == num_pixels:
        return side, side

    for width in range(2, int(np.sqrt(num_pixels)) + 1):
        if num_pixels % width == 0:
            height = num_pixels // width
            if 0.5 <= width / height <= 2.0:
                return width, height

    raise ValueError(
        f"Could not infer image dimensions for {num_pixels:,} pixels. "
        "Specify --width and --height explicitly."
    )


def load_bayer_bin(path: Path, width: int | None, height: int | None) -> np.ndarray:
    """Load a raw Bayer uint16 .bin file into a 2D array."""
    raw = np.fromfile(path, dtype=np.uint16)
    if raw.size == 0:
        raise ValueError(f"{path} is empty.")

    if width is None or height is None:
        inferred_w, inferred_h = infer_dimensions(raw.size)
        width = width or inferred_w
        height = height or inferred_h

    expected = width * height
    if raw.size != expected:
        raise ValueError(
            f"{path.name}: expected {expected:,} pixels ({width}x{height}), "
            f"but file contains {raw.size:,} uint16 values."
        )

    return raw.reshape((height, width))


def sanitize_var_name(path: Path) -> str:
    """Create a valid MATLAB struct field name from a filename."""
    name = re.sub(r"[^0-9a-zA-Z_]", "_", path.stem)
    if not name or name[0].isdigit():
        name = f"img_{name}"
    return name


def convert_folder(
    input_dir: Path,
    output_mat: Path,
    width: int | None,
    height: int | None,
    pattern: str,
) -> None:
    bin_files = sorted(input_dir.glob("*.bin"))
    if not bin_files:
        raise FileNotFoundError(f"No .bin files found in {input_dir}")

    mat_data: dict[str, object] = {
        "bayer_pattern": pattern,
        "dtype": "uint16",
    }

    for index, bin_path in enumerate(bin_files, start=1):
        image = load_bayer_bin(bin_path, width, height)
        var_name = sanitize_var_name(bin_path)
        mat_data[var_name] = image
        mat_data[f"{var_name}_filename"] = bin_path.name
        print(
            f"[{index}/{len(bin_files)}] {bin_path.name}: "
            f"{image.shape[1]}x{image.shape[0]} uint16 Bayer"
        )

    savemat(output_mat, mat_data, do_compression=True)
    print(f"\nSaved {len(bin_files)} image(s) to {output_mat}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw Bayer uint16 .bin files to a MATLAB .mat file."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Folder containing .bin files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .mat path (default: <input_dir>/bayer_images.mat)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Image width in pixels (auto-detected if omitted)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Image height in pixels (auto-detected if omitted)",
    )
    parser.add_argument(
        "--pattern",
        default="RGGB",
        choices=["RGGB", "BGGR", "GRBG", "GBRG"],
        help="Bayer color filter arrangement (default: RGGB)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_mat = (
        args.output.resolve()
        if args.output
        else input_dir / "bayer_images.mat"
    )

    if not input_dir.is_dir():
        raise SystemExit(f"Input folder does not exist: {input_dir}")

    convert_folder(input_dir, output_mat, args.width, args.height, args.pattern)


if __name__ == "__main__":
    main()
