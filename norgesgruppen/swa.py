"""Stochastic Weight Averaging — merge multiple model checkpoints.

Averages the weights of multiple YOLOv8 models to create a smoother,
more generalizable model. Zero inference cost (single model output).

Usage:
    python swa.py models/best_a.pt models/best_b.pt models/best_c.pt -o models/swa.pt
"""

import argparse
import torch
from pathlib import Path
from collections import OrderedDict


def average_checkpoints(model_paths, output_path):
    """Average state dicts from multiple YOLO checkpoints."""
    print(f"Averaging {len(model_paths)} models:")

    # Load all state dicts
    state_dicts = []
    for path in model_paths:
        print(f"  Loading {path}...")
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        if "model" in ckpt:
            sd = ckpt["model"].float().state_dict()
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            sd = ckpt
        state_dicts.append(sd)
        print(f"    Keys: {len(sd)}")

    # Average weights
    avg_sd = OrderedDict()
    keys = state_dicts[0].keys()
    for key in keys:
        tensors = [sd[key].float() for sd in state_dicts if key in sd]
        if len(tensors) == len(state_dicts):
            avg_sd[key] = torch.stack(tensors).mean(dim=0)
        else:
            # Key missing in some models, use first available
            avg_sd[key] = tensors[0]

    # Save as YOLO checkpoint format
    # Load first checkpoint as template
    template = torch.load(str(model_paths[0]), map_location="cpu", weights_only=False)
    if "model" in template:
        # Use strict=True to catch mismatches
        template["model"].load_state_dict(avg_sd, strict=True)
        # Clear optimizer state (not needed for inference)
        template.pop("optimizer", None)
        # Keep EMA — YOLO uses ema.model for inference, update it with averaged weights
        if "ema" in template and template["ema"] is not None:
            try:
                template["ema"].load_state_dict(avg_sd, strict=True)
                print("Updated EMA with averaged weights")
            except Exception as e:
                # EMA might be a different structure, try setting ema to None
                # so YOLO falls back to model weights
                print(f"EMA update failed ({e}), removing EMA")
                template.pop("ema", None)

    torch.save(template, str(output_path))
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"\nSaved SWA model to {output_path} ({size_mb:.1f} MB)")
    print(f"Averaged {len(state_dicts)} models, {len(avg_sd)} parameters")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", help="Model checkpoint paths")
    parser.add_argument("-o", "--output", default="models/swa.pt",
                        help="Output path for averaged model")
    args = parser.parse_args()

    for m in args.models:
        if not Path(m).exists():
            print(f"ERROR: {m} not found")
            return

    average_checkpoints([Path(m) for m in args.models], Path(args.output))


if __name__ == "__main__":
    main()
