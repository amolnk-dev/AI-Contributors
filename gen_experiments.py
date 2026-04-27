#!/usr/bin/env python3
"""Generate experiment variants of train.py for GCP fleet parallel testing.

Creates modified copies of train.py with specific parameter overrides.
Each experiment file is a standalone train.py that can run independently.

Usage:
    python gen_experiments.py                    # Generate default sweep
    python gen_experiments.py --outdir /tmp/exp  # Custom output directory
    python gen_experiments.py --list             # Show experiments without generating

The generated files go into experiments/ directory, one per VM.
Fleet-experiment.sh deploys these to VMs and collects results.
"""

import argparse
import os
import re
import sys


# Read current train.py as template
TRAIN_PY = os.path.join(os.path.dirname(__file__), "train.py")


def read_template():
    with open(TRAIN_PY) as f:
        return f.read()


def apply_param(source: str, param: str, value: str) -> str:
    """Replace a parameter value in the TUNABLE PARAMETERS section.

    Handles both top-level and nested dict assignments:
      BLEND_WEIGHT = 0.35  ->  BLEND_WEIGHT = 0.40
      "plains": 0.55       ->  "plains": 0.60
    """
    # Try top-level assignment: PARAM = VALUE
    pattern = rf'^(\s*{re.escape(param)}\s*=\s*)(.+)$'
    new_source, n = re.subn(pattern, rf'\g<1>{value}', source, count=1, flags=re.MULTILINE)
    if n > 0:
        return new_source

    # Try dict value: "param": VALUE  or  'param': VALUE
    pattern = rf'(["\']{re.escape(param)}["\']\s*:\s*)([^,\}}\n]+)'
    new_source, n = re.subn(pattern, rf'\g<1>{value}', source, count=1)
    if n > 0:
        return new_source

    # Try numpy array element by index (e.g., L7_STRENGTHS index 0)
    # This is for cases like "L7_STRENGTHS[0] = 1.60"
    raise ValueError(f"Parameter '{param}' not found in train.py")


def apply_params(source: str, overrides: dict) -> str:
    """Apply multiple parameter overrides."""
    for param, value in overrides.items():
        source = apply_param(source, param, str(value))
    return source


# ──────────────────────────────────────────────────────────────
# Experiment definitions
# ──────────────────────────────────────────────────────────────

def get_default_experiments() -> list[dict]:
    """Return the default experiment sweep.

    Each experiment is a dict with:
      - name: short identifier (used as filename and VM assignment)
      - desc: human-readable description
      - params: dict of param_name -> new_value
    """
    return [
        # EMP_BLEND sweep (current=0.50 in evaluate_loro)
        {"name": "emp40", "desc": "EMP_BLEND=0.40", "params": {"EMP_BLEND": "0.40"}},
        {"name": "emp45", "desc": "EMP_BLEND=0.45", "params": {"EMP_BLEND": "0.45"}},
        {"name": "emp55", "desc": "EMP_BLEND=0.55", "params": {"EMP_BLEND": "0.55"}},

        # Per-terrain blend (current: p=1.00, f=1.00, s=1.00 in train.py)
        {"name": "blend_095", "desc": "all blend=0.95",
         "params": {"plains": "0.95", "forest": "0.95", "settl": "0.95"}},
        {"name": "blend_090", "desc": "all blend=0.90",
         "params": {"plains": "0.90", "forest": "0.90", "settl": "0.90"}},

        # L7 strength sweep (current: [1.40, 1.00, 0, 0, 1.50, 0])
        {"name": "l7_strong", "desc": "L7 stronger [1.60, 1.20, 0, 0, 1.70, 0]",
         "params": {"L7_STRENGTHS": "np.array([1.60, 1.20, 0.0, 0.0, 1.70, 0.0])"}},
        {"name": "l7_weak", "desc": "L7 weaker [1.20, 0.80, 0, 0, 1.30, 0]",
         "params": {"L7_STRENGTHS": "np.array([1.20, 0.80, 0.0, 0.0, 1.30, 0.0])"}},
        {"name": "l7_wide", "desc": "L7 wider clamp [0.70-1.40]",
         "params": {"L7_ADJ_MIN": "0.70", "L7_ADJ_MAX": "1.40"}},

        # XGB hyperparameters
        {"name": "xgb_depth6", "desc": "max_depth=6 all terrains",
         "params": {"max_depth": "6"}},
        {"name": "xgb_800", "desc": "n_estimators=800 all terrains",
         "params": {"n_estimators": "800"}},
        {"name": "xgb_lr005", "desc": "learning_rate=0.05 (slower)",
         "params": {"learning_rate": "0.05"}},

        # Baseline sanity check
        {"name": "baseline", "desc": "BASELINE (no changes)", "params": {}},
    ]


def generate(experiments: list[dict], outdir: str):
    """Generate experiment files."""
    os.makedirs(outdir, exist_ok=True)
    template = read_template()

    manifest = []
    for exp in experiments:
        try:
            modified = apply_params(template, exp["params"])
        except ValueError as e:
            print(f"  SKIP {exp['name']}: {e}", file=sys.stderr)
            continue

        fname = f"train_{exp['name']}.py"
        path = os.path.join(outdir, fname)
        with open(path, "w") as f:
            f.write(modified)
        manifest.append({"name": exp["name"], "desc": exp["desc"], "file": fname})
        print(f"  {fname}: {exp['desc']}")

    # Write manifest for fleet-experiment.sh to read
    manifest_path = os.path.join(outdir, "manifest.txt")
    with open(manifest_path, "w") as f:
        for m in manifest:
            f.write(f"{m['name']}|{m['file']}|{m['desc']}\n")

    print(f"\nGenerated {len(manifest)} experiments in {outdir}/")
    print(f"Manifest: {manifest_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Generate fleet experiment variants")
    parser.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "experiments"),
                        help="Output directory for generated files")
    parser.add_argument("--list", action="store_true", help="List experiments without generating")
    parser.add_argument("--custom", nargs="*", help="Custom param overrides: name=val ...")
    args = parser.parse_args()

    experiments = get_default_experiments()

    if args.custom:
        # Parse custom experiment: gen_experiments.py --custom EMP_BLEND=0.42 plains=0.50
        params = {}
        for kv in args.custom:
            k, v = kv.split("=", 1)
            params[k] = v
        desc = " ".join(args.custom)
        experiments.append({"name": "custom", "desc": desc, "params": params})

    if args.list:
        print(f"{'NAME':<15} {'DESCRIPTION'}")
        print("-" * 60)
        for exp in experiments:
            print(f"{exp['name']:<15} {exp['desc']}")
        return

    generate(experiments, args.outdir)


if __name__ == "__main__":
    main()
