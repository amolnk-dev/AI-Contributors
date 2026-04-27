"""Strip YOLO checkpoint to inference-only weights (~84MB instead of ~352MB).

Keeps only the EMA model (or regular model if no EMA), discards optimizer state.

Usage:
    python strip_model.py runs/cls10/weights/best.pt -o models/best_cls10.pt
    python strip_model.py input.pt  # outputs input_stripped.pt
"""

import argparse
import torch
from pathlib import Path


def strip_checkpoint(input_path, output_path):
    ckpt = torch.load(str(input_path), map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        # Standard YOLO checkpoint — keep model + EMA, drop optimizer
        stripped = {}

        # Use EMA model for inference (it's the better one)
        if "ema" in ckpt and ckpt["ema"] is not None:
            stripped["model"] = ckpt["ema"]
            print(f"Using EMA model for inference")
        else:
            stripped["model"] = ckpt["model"]
            print(f"No EMA found, using regular model")

        # Keep training args for compatibility
        if "train_args" in ckpt:
            stripped["train_args"] = ckpt["train_args"]

        torch.save(stripped, str(output_path))
    else:
        # Already stripped or unknown format — just copy
        torch.save(ckpt, str(output_path))

    in_mb = Path(input_path).stat().st_size / (1024 * 1024)
    out_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Stripped: {in_mb:.1f} MB → {out_mb:.1f} MB ({output_path})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input YOLO checkpoint")
    parser.add_argument("-o", "--output", help="Output path (default: input_stripped.pt)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_stem(input_path.stem + "_stripped")

    strip_checkpoint(input_path, output_path)


if __name__ == "__main__":
    main()
