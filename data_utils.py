"""Autoresearch: autonomous experiment loop for overnight optimization.

This is a lightweight GCP/PyTorch adapter for the autoresearch protocol.
The full protocol and MLX implementation live in the sibling repo:

    ~/Utvikling/autoresearch-mlx/
    https://github.com/Walgermo/autoresearch-mlx

The protocol (from autoresearch-mlx/program.md):
    1. Edit train.py with an experimental idea
    2. Commit: git add tasks/<task>/train.py && git commit -m "experiment: <desc>"
    3. Run: cd tasks/<task> && uv run train.py > run.log 2>&1
    4. Read: grep "^val_metric:" run.log
    5. If improved → git add results.tsv && git commit --amend --no-edit
    6. If worse → git reset --hard <previous kept commit>
    7. Log to results.tsv (tab-separated: commit, val_metric, memory_gb, diff_lines, status, description, reject_reason)
    8. Repeat indefinitely

For GCP (PyTorch/CUDA) runs during competition, this module provides
helpers to run the same loop on GPU VMs. For local Apple Silicon runs,
use autoresearch-mlx directly.

Task contract (from autoresearch-mlx):
    prepare.py exports: TIME_BUDGET (int), evaluate(...) (returns float)
    train.py prints: val_metric: <float> in output block
    task.md declares: metric_name, metric_direction (lower/higher)
"""

import math
import subprocess
import time
from datetime import datetime
from pathlib import Path


def run_experiment(command: str, task_dir: str, timeout_minutes: float = 15.0) -> dict:
    """Run a training experiment as a subprocess with timeout.

    Args:
        command: Shell command to run (e.g. "python train.py" or "uv run train.py")
        task_dir: Working directory for the command
        timeout_minutes: Max wall-clock time before killing the process

    Returns:
        dict with keys: val_metric, duration_s, success, error, output_tail
    """
    start = time.time()
    log_path = Path(task_dir) / "run.log"

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=task_dir,
            capture_output=True,
            text=True,
            timeout=timeout_minutes * 60,
        )

        # Write output to log file
        log_path.write_text(result.stdout + result.stderr)
        duration = time.time() - start

        # Parse val_metric from output
        val_metric = None
        for line in result.stdout.splitlines():
            if line.startswith("val_metric:"):
                try:
                    val_metric = float(line.split(":")[1].strip())
                except ValueError:
                    pass

        if result.returncode != 0:
            tail = result.stderr[-500:] if result.stderr else result.stdout[-500:]
            return {
                "val_metric": 0.0,
                "duration_s": duration,
                "success": False,
                "error": f"exit code {result.returncode}",
                "output_tail": tail,
            }

        if val_metric is None or not math.isfinite(val_metric):
            return {
                "val_metric": 0.0,
                "duration_s": duration,
                "success": False,
                "error": "no valid val_metric found in output (got: {})".format(val_metric),
                "output_tail": result.stdout[-500:],
            }

        return {
            "val_metric": val_metric,
            "duration_s": duration,
            "success": True,
            "error": None,
            "output_tail": None,
        }

    except subprocess.TimeoutExpired:
        return {
            "val_metric": 0.0,
            "duration_s": time.time() - start,
            "success": False,
            "error": "timeout",
            "output_tail": None,
        }
    except Exception as e:
        return {
            "val_metric": 0.0,
            "duration_s": time.time() - start,
            "success": False,
            "error": str(e),
            "output_tail": None,
        }


def parse_peak_memory(task_dir: str) -> float:
    """Parse peak_vram_mb from run.log, return GB."""
    log_path = Path(task_dir) / "run.log"
    if not log_path.exists():
        return 0.0
    for line in log_path.read_text().splitlines():
        if line.startswith("peak_vram_mb:"):
            try:
                return float(line.split(":")[1].strip()) / 1024
            except ValueError:
                pass
    return 0.0


def log_result(results_path: str, entry: dict):
    """Append a row to results.tsv."""
    tsv = Path(results_path)
    if not tsv.exists():
        tsv.write_text("commit\tval_metric\tmemory_gb\tdiff_lines\tstatus\tdescription\treject_reason\n")

    row = "\t".join(str(entry.get(k, "-")) for k in [
        "commit", "val_metric", "memory_gb", "diff_lines", "status", "description", "reject_reason"
    ])
    with open(tsv, "a") as f:
        f.write(row + "\n")
