"""AI-guided swarm autoresearch with Gemini coordination.

Fleet of GPU VMs coordinated by a shared results file. Each VM:
1. Reads ALL results (from all VMs) via shared results file
2. Asks Gemini to suggest the next config based on the FULL fleet history
3. Runs the experiment, appends result to shared file
4. Repeat

The "swarm intelligence" comes from every VM seeing every other VM's results
when asking Gemini for the next config.

Usage on each VM:
    VM_ID=a0 GPU_TYPE=a100 GOOGLE_API_KEY=... nohup python3 autoresearch_overnight.py > overnight.log 2>&1 &

Environment variables:
    VM_ID          — unique per VM (default: "vm0")
    GPU_TYPE       — "a100" or "l4" (default: "l4")
    GOOGLE_API_KEY — Gemini API key (required for AI mode)
    MODEL_ID       — Gemini model (default: "gemini-2.5-flash")
"""

import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

TASK_DIR = Path(__file__).parent.resolve()
MODELS_DIR = TASK_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

VM_ID = os.environ.get("VM_ID", "vm0")
GPU_TYPE = os.environ.get("GPU_TYPE", "l4").lower()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.environ.get("MODEL_ID", "gemini-3.1-pro-preview")

# Hub VM for swarm coordination — pulls results from all VMs, serves merged file
HUB_IP = os.environ.get("HUB_IP", "10.128.0.13")  # ainm-norgesgruppen-a100
IS_HUB = os.environ.get("IS_HUB", "0") == "1"

TIMEOUT_HOURS = 1.5 if GPU_TYPE == "a100" else 3.0

# Per-VM results file
RESULTS_TSV = TASK_DIR / f"overnight_results_{VM_ID}.tsv"
# Shared fleet results — merged by hub, pulled by workers
SHARED_RESULTS = TASK_DIR / "overnight_results_fleet.tsv"
BEST_MODEL = MODELS_DIR / f"best_overnight_{VM_ID}.pt"

DATA_YAML = TASK_DIR / "data" / "yolo" / "data.yaml"

BASE_MODEL = MODELS_DIR / "best.pt"
if not BASE_MODEL.exists():
    BASE_MODEL = TASK_DIR / "yolov8l.pt"

SEED = int(hashlib.md5(VM_ID.encode()).hexdigest()[:8], 16) % (2**31)

TSV_HEADER = (
    "timestamp\tvm_id\tval_metric\tseed\tcls\tbox\tdfl\tmosaic\tmixup\t"
    "copy_paste\tdegrees\tscale\tepochs\tclose_mosaic\twarmup_epochs\t"
    "freeze\tlabel_smoothing\tlr0\tlrf\tcos_lr\tduration_min\tstatus\tnotes\n"
)

# ─── Search space: NARROWED based on 506 experiments ─────────────────
# Winners: cls=1.1-1.2, close_mosaic=3, freeze=None, box=5-7.5
# Losers: mosaic=0.5 (worst keep rate), freeze=10/15, box=10
SEARCH_SPACE = {
    "epochs":         [20, 30, 30, 40],
    "lr0":            [0.0005, 0.001, 0.001, 0.002],
    "lrf":            [0.001, 0.01, 0.01],
    "cos_lr":         [True, True, False],
    "warmup_epochs":  [1.0, 2.0, 3.0],
    "cls":            [1.0, 1.1, 1.1, 1.2, 1.2, 1.3],
    "box":            [5.0, 7.5, 7.5],
    "dfl":            [1.0, 1.5, 1.5],
    "mosaic":         [0.0, 0.3, 0.8, 1.0],
    "mixup":          [0.0, 0.05, 0.05, 0.1],
    "copy_paste":     [0.0, 0.0, 0.05],
    "degrees":        [0.0, 5.0, 10.0],
    "scale":          [0.3, 0.5],
    "close_mosaic":   [3, 3, 5],
    "freeze":         [None, None, None, None],
    "label_smoothing": [0.0, 0.0, 0.05, 0.05],
}

FIXED = {
    "imgsz":     1280,
    "optimizer": "SGD",
    "batch":     2,
    "patience":  20,
}


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{VM_ID}] {msg}", flush=True)


def init_results():
    for tsv in [RESULTS_TSV, SHARED_RESULTS]:
        if not tsv.exists():
            tsv.write_text(TSV_HEADER)


def make_result_row(config, val_metric, duration_min, status, notes=""):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    freeze_str = str(config.get("freeze", "none"))
    return (
        f"{ts}\t{VM_ID}\t{val_metric:.4f}\t{config['seed']}\t"
        f"{config['cls']}\t{config['box']}\t{config['dfl']}\t"
        f"{config['mosaic']}\t{config['mixup']}\t{config['copy_paste']}\t"
        f"{config['degrees']}\t{config['scale']}\t{config['epochs']}\t"
        f"{config['close_mosaic']}\t{config['warmup_epochs']}\t"
        f"{freeze_str}\t{config['label_smoothing']}\t"
        f"{config['lr0']}\t{config['lrf']}\t{config['cos_lr']}\t"
        f"{duration_min:.1f}\t{status}\t{notes}\n"
    )


def append_result(config, val_metric, duration_min, status, notes=""):
    row = make_result_row(config, val_metric, duration_min, status, notes)
    # Write to both per-VM and shared fleet file
    for tsv in [RESULTS_TSV, SHARED_RESULTS]:
        with open(tsv, "a") as f:
            f.write(row)


def get_best_metric():
    """Best metric from this VM's results."""
    if not RESULTS_TSV.exists():
        return 0.0
    best = 0.0
    for line in RESULTS_TSV.read_text().strip().split("\n")[1:]:
        parts = line.split("\t")
        # Status column: index 21 (new format) or 18 (old format)
        status_idx = 21 if len(parts) >= 22 else 18
        if len(parts) > status_idx and "kept" in parts[status_idx]:
            try:
                best = max(best, float(parts[2]))
            except ValueError:
                pass
    return best


def get_fleet_results_summary():
    """Summarize ALL fleet results for Gemini — swarm intelligence."""
    # Read shared fleet results
    source = SHARED_RESULTS if SHARED_RESULTS.exists() else RESULTS_TSV
    if not source.exists():
        return "No results yet. This is the first run across the fleet."

    lines = source.read_text().strip().split("\n")
    if len(lines) <= 1:
        return "No results yet. This is the first run across the fleet."

    header = lines[0]
    data_lines = lines[1:]

    # Parse and sort
    parsed = []
    for line in data_lines:
        parts = line.split("\t")
        if len(parts) >= 19:
            try:
                metric = float(parts[2])
                status = parts[18]
            except (ValueError, IndexError):
                metric = 0.0
                status = "unknown"
            if status in ("kept", "rejected") and metric > 0:
                parsed.append((metric, line))

    if not parsed:
        return "No successful results yet. All runs failed."

    parsed.sort(key=lambda x: -x[0])
    n_total = len(data_lines)
    n_success = len(parsed)
    n_kept = sum(1 for _, l in parsed if "kept" in l)

    summary = f"Fleet: {n_total} total runs, {n_success} successful, {n_kept} improvements\n"
    summary += f"Best metric: {parsed[0][0]:.4f}\n"
    summary += f"Median metric: {parsed[len(parsed)//2][0]:.4f}\n\n"
    summary += f"HEADER: {header}\n\n"

    # Top 10 + bottom 5 (compact for Gemini token budget)
    summary += "TOP 10 (best configs):\n"
    for metric, line in parsed[:10]:
        parts = line.split("\t")
        if len(parts) >= 22:
            # New format with lr0/lrf/cos_lr
            summary += f"  metric={parts[2]} cls={parts[4]} box={parts[5]} dfl={parts[6]} mos={parts[7]} mix={parts[8]} ep={parts[12]} cm={parts[13]} frz={parts[15]} ls={parts[16]} lr0={parts[17]} lrf={parts[18]} cos={parts[19]}\n"
        elif len(parts) >= 17:
            # Old format without lr0/lrf/cos_lr
            summary += f"  metric={parts[2]} cls={parts[4]} box={parts[5]} dfl={parts[6]} mos={parts[7]} mix={parts[8]} ep={parts[12]} cm={parts[13]} frz={parts[15]} ls={parts[16]}\n"

    if len(parsed) > 10:
        summary += "\nBOTTOM 5 (avoid):\n"
        for metric, line in parsed[-5:]:
            parts = line.split("\t")
            if len(parts) >= 22:
                summary += f"  metric={parts[2]} cls={parts[4]} box={parts[5]} dfl={parts[6]} mos={parts[7]} mix={parts[8]} ep={parts[12]} cm={parts[13]} frz={parts[15]} ls={parts[16]} lr0={parts[17]} lrf={parts[18]} cos={parts[19]}\n"
            elif len(parts) >= 17:
                summary += f"  metric={parts[2]} cls={parts[4]} box={parts[5]} dfl={parts[6]} mos={parts[7]} mix={parts[8]} ep={parts[12]} cm={parts[13]} frz={parts[15]} ls={parts[16]}\n"

    return summary


def _clean_json(text):
    """Strip markdown fences and whitespace from Gemini response."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def sample_config_random(rng):
    config = {}
    for key, values in SEARCH_SPACE.items():
        config[key] = rng.choice(values)
    config["seed"] = rng.randint(0, 9999)
    return config


def sample_config_gemini(rng):
    """Ask Gemini for next config based on FLEET-WIDE results."""
    if not GOOGLE_API_KEY:
        return None

    results_summary = get_fleet_results_summary()

    prompt = f"""HPO for YOLOv8l fine-tuning. 356 grocery classes, 248 imgs, 1280px. Suggest next config.

{results_summary}

Return ONLY JSON with these keys (no markdown, no explanation):
epochs(int) lr0(float) lrf(float) cos_lr(bool) warmup_epochs(float) cls(float) box(float) dfl(float) mosaic(float) mixup(float) copy_paste(float) degrees(float) scale(float) close_mosaic(int) freeze(null or int) label_smoothing(float) seed(int 0-9999) reasoning(string)

Exploit what works in top results. Avoid patterns from bottom results."""

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GOOGLE_API_KEY}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.8,
                "maxOutputTokens": 8192,
            }
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode())

        text = body["candidates"][0]["content"]["parts"][0]["text"].strip()

        # Try parsing: raw first, then cleaned, then regex extraction
        raw = None
        for attempt_text in [text, _clean_json(text)]:
            try:
                raw = json.loads(attempt_text)
                break
            except (json.JSONDecodeError, ValueError):
                continue

        # Last resort: find JSON between first { and last }
        if raw is None:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    raw = json.loads(text[start:end + 1])
                except (json.JSONDecodeError, ValueError):
                    pass

        if raw is None:
            log(f"Gemini returned unparseable: {text[:200]}")
            return None

        # Log Gemini's reasoning
        reasoning = raw.pop("reasoning", "no reasoning given")
        log(f"Gemini reasoning: {reasoning}")

        # Validate and coerce
        config = {}
        for key, values in SEARCH_SPACE.items():
            if key not in raw:
                config[key] = rng.choice(values)
                continue
            val = raw[key]
            if key == "freeze":
                config[key] = None if val is None or str(val).lower() in ("null", "none") else int(val)
            elif key == "cos_lr":
                config[key] = val if isinstance(val, bool) else str(val).lower() not in ("false", "0", "no")
            elif key in ("epochs", "close_mosaic"):
                config[key] = int(val)
            else:
                config[key] = float(val)

        config["seed"] = int(raw.get("seed", rng.randint(0, 9999)))
        return config

    except Exception as e:
        log(f"Gemini error: {e}")
        return None


def sample_config(rng, use_ai=True):
    if use_ai and GOOGLE_API_KEY:
        config = sample_config_gemini(rng)
        if config is not None:
            log("Config source: GEMINI AI (fleet-aware)")
            return config
    config = sample_config_random(rng)
    log("Config source: random (fallback)")
    return config


def config_to_name(config, run_idx):
    return f"overnight_{VM_ID}_r{run_idx}"


def get_current_best_model():
    if BEST_MODEL.exists():
        return str(BEST_MODEL)
    return str(BASE_MODEL)


def generate_train_script(config, exp_name):
    model_path = get_current_best_model()

    freeze_line = ""
    if config["freeze"] is not None:
        freeze_line = f"    freeze={config['freeze']},"

    label_smoothing_line = ""
    if config["label_smoothing"] > 0:
        label_smoothing_line = f"    label_smoothing={config['label_smoothing']},"

    return f'''
import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
_orig = torch.load
def _patched(*a, **kw):
    if "weights_only" not in kw: kw["weights_only"] = False
    return _orig(*a, **kw)
torch.load = _patched

from ultralytics import YOLO
from pathlib import Path

model = YOLO("{model_path}")
results = model.train(
    data="{DATA_YAML}",
    imgsz={FIXED['imgsz']},
    epochs={config['epochs']},
    batch={FIXED['batch']},
    patience={FIXED['patience']},
    device=0,
    project="runs",
    name="{exp_name}",
    exist_ok=True,
    seed={config['seed']},
    optimizer="{FIXED['optimizer']}",
    cos_lr={config['cos_lr']},
    lr0={config['lr0']},
    lrf={config['lrf']},
    warmup_epochs={config['warmup_epochs']},
    close_mosaic={config['close_mosaic']},
    box={config['box']},
    cls={config['cls']},
    dfl={config['dfl']},
    mosaic={config['mosaic']},
    mixup={config['mixup']},
    copy_paste={config['copy_paste']},
    degrees={config['degrees']},
    scale={config['scale']},
    fliplr=0.5,
    flipud=0.0,
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.3,
{freeze_line}
{label_smoothing_line}
    save=True,
    save_period=-1,
    plots=False,
    verbose=True,
)

metrics = results.results_dict
map50 = metrics.get("metrics/mAP50(B)", 0.0)
print(f"val_metric: {{map50}}")

best_pt = Path("runs") / "{exp_name}" / "weights" / "best.pt"
if best_pt.exists():
    size_mb = best_pt.stat().st_size / (1024 * 1024)
    print(f"model_size_mb: {{size_mb:.1f}}")

try:
    peak = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"peak_vram_mb: {{peak:.0f}}")
except: pass
'''


def run_experiment(config, run_idx):
    exp_name = config_to_name(config, run_idx)
    model_path = get_current_best_model()
    log(f"--- Run {run_idx}: {exp_name} ---")
    log(f"Fine-tuning from: {Path(model_path).name}")
    log(f"Config: seed={config['seed']} ep={config['epochs']} lr0={config['lr0']} "
        f"lrf={config['lrf']} cos={config['cos_lr']} cls={config['cls']} "
        f"box={config['box']} dfl={config['dfl']} mosaic={config['mosaic']} "
        f"mixup={config['mixup']} cp={config['copy_paste']} deg={config['degrees']} "
        f"scale={config['scale']} cm={config['close_mosaic']} wu={config['warmup_epochs']} "
        f"freeze={config['freeze']} ls={config['label_smoothing']}")

    start = time.time()
    train_script_path = TASK_DIR / f"_tmp_{exp_name}.py"
    train_script_path.write_text(generate_train_script(config, exp_name))

    env = os.environ.copy()
    env["WANDB_DISABLED"] = "true"
    env["WANDB_MODE"] = "disabled"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    try:
        result = subprocess.run(
            ["python3", str(train_script_path)],
            cwd=str(TASK_DIR),
            capture_output=True,
            text=True,
            timeout=TIMEOUT_HOURS * 3600,
            env=env,
        )

        duration_min = (time.time() - start) / 60
        train_script_path.unlink(missing_ok=True)

        log_file = TASK_DIR / f"{exp_name}.log"
        log_file.write_text(result.stdout[-50000:] + "\n--- STDERR ---\n" + result.stderr[-10000:])

        if result.returncode != 0:
            err_lines = [l for l in result.stderr.split("\n") if l.strip()][-5:]
            log(f"FAILED ({duration_min:.1f}min): {'; '.join(err_lines)}")
            append_result(config, 0.0, duration_min, "failed", f"exit {result.returncode}")
            cleanup_run(exp_name)
            return None

        val_metric = None
        for line in result.stdout.split("\n"):
            if line.startswith("val_metric:"):
                try:
                    val_metric = float(line.split(":")[1].strip())
                except ValueError:
                    pass

        if val_metric is None or not math.isfinite(val_metric):
            log(f"NO METRIC ({duration_min:.1f}min)")
            append_result(config, 0.0, duration_min, "failed", "no val_metric")
            cleanup_run(exp_name)
            return None

        current_best = get_best_metric()
        is_best = val_metric > current_best
        if is_best:
            log(f"NEW BEST: {val_metric:.4f} > {current_best:.4f} ({duration_min:.1f}min)")
            best_pt_path = TASK_DIR / "runs" / exp_name / "weights" / "best.pt"
            if best_pt_path.exists():
                shutil.copy2(best_pt_path, BEST_MODEL)
            append_result(config, val_metric, duration_min, "kept",
                         f"beat {current_best:.4f}")
        else:
            log(f"NO GAIN: {val_metric:.4f} <= {current_best:.4f} ({duration_min:.1f}min)")
            append_result(config, val_metric, duration_min, "rejected",
                         f"below {current_best:.4f}")

        cleanup_run(exp_name, keep_log=is_best)
        return val_metric

    except subprocess.TimeoutExpired:
        train_script_path.unlink(missing_ok=True)
        duration_min = (time.time() - start) / 60
        log(f"TIMEOUT ({duration_min:.0f}min)")
        append_result(config, 0.0, duration_min, "timeout", "")
        cleanup_run(exp_name)
        return None

    except Exception as e:
        train_script_path.unlink(missing_ok=True)
        duration_min = (time.time() - start) / 60
        log(f"CRASH: {e}")
        append_result(config, 0.0, duration_min, "crash", str(e)[:100])
        cleanup_run(exp_name)
        return None


def cleanup_run(exp_name, keep_log=False):
    run_dir = TASK_DIR / "runs" / exp_name
    if run_dir.exists():
        try:
            shutil.rmtree(run_dir)
        except Exception as e:
            log(f"Warning: cleanup failed for {run_dir}: {e}")
    if not keep_log:
        log_file = TASK_DIR / f"{exp_name}.log"
        log_file.unlink(missing_ok=True)


def wait_for_gpu():
    while True:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                log("GPU busy, waiting 60s...")
                time.sleep(60)
            else:
                return
        except Exception:
            return


def get_completed_runs():
    if not RESULTS_TSV.exists():
        return 0
    lines = RESULTS_TSV.read_text().strip().split("\n")
    return max(0, len(lines) - 1)


# ─── Swarm sync ──────────────────────────────────────────────────────

# All VM internal IPs for hub to pull from
FLEET_IPS = os.environ.get("FLEET_IPS", "").split(",") if os.environ.get("FLEET_IPS") else []


def sync_fleet_results():
    """Sync results across the fleet.

    Hub: pulls overnight_results_*.tsv from all VMs, merges into fleet file.
    Workers: pull the merged fleet file from hub.
    """
    if IS_HUB:
        _hub_collect_and_merge()
    else:
        _worker_pull_fleet()


def _hub_collect_and_merge():
    """Hub: pull per-VM results from all VMs, merge into fleet file."""
    if not FLEET_IPS:
        # No fleet IPs configured — just merge local files
        _merge_local_results()
        return

    # Pull results from each VM (best-effort, don't block on failures)
    for ip in FLEET_IPS:
        ip = ip.strip()
        if not ip:
            continue
        try:
            subprocess.run(
                ["scp", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
                 f"root@{ip}:/root/task/overnight_results_*.tsv",
                 str(TASK_DIR) + "/"],
                capture_output=True, timeout=15
            )
        except Exception:
            pass

    _merge_local_results()
    log(f"Hub merged fleet results: {_count_fleet_results()} total")


def _merge_local_results():
    """Merge all local overnight_results_*.tsv into fleet file."""
    all_rows = set()
    for f in TASK_DIR.glob("overnight_results_*.tsv"):
        if f.name == "overnight_results_fleet.tsv":
            continue
        for line in f.read_text().strip().split("\n")[1:]:
            if line.strip():
                all_rows.add(line.strip())

    with open(SHARED_RESULTS, "w") as f:
        f.write(TSV_HEADER)
        for row in sorted(all_rows):
            f.write(row + "\n")


def _worker_pull_fleet():
    """Worker: pull merged fleet file from hub."""
    try:
        result = subprocess.run(
            ["scp", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
             f"root@{HUB_IP}:/root/task/overnight_results_fleet.tsv",
             str(SHARED_RESULTS)],
            capture_output=True, timeout=15
        )
        if result.returncode != 0:
            log(f"Sync pull failed (rc={result.returncode}), using local results only")
    except Exception as e:
        log(f"Sync pull error: {e}, using local results only")


def _push_results_to_hub():
    """Worker: push this VM's results to the hub after each experiment."""
    try:
        result = subprocess.run(
            ["scp", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
             str(RESULTS_TSV),
             f"root@{HUB_IP}:/root/task/overnight_results_{VM_ID}.tsv"],
            capture_output=True, timeout=15
        )
        if result.returncode != 0:
            log(f"Sync push failed (rc={result.returncode})")
    except Exception:
        pass


def _count_fleet_results():
    if not SHARED_RESULTS.exists():
        return 0
    return max(0, len(SHARED_RESULTS.read_text().strip().split("\n")) - 1)


def main():
    rng = random.Random(SEED)

    ai_mode = "GEMINI SWARM" if GOOGLE_API_KEY else "RANDOM (no API key)"
    log(f"Autoresearch starting — mode: {ai_mode}")
    log(f"VM_ID={VM_ID}, GPU_TYPE={GPU_TYPE}, timeout={TIMEOUT_HOURS}h")
    log(f"Gemini model: {GEMINI_MODEL}")
    log(f"Data: {DATA_YAML}")
    log(f"Base model: {BASE_MODEL}")
    log(f"Shared results: {SHARED_RESULTS}")

    init_results()

    completed = get_completed_runs()
    if completed > 0:
        log(f"Resuming after {completed} previous runs (best: {get_best_metric():.4f})")
        for _ in range(completed):
            sample_config_random(rng)

    wait_for_gpu()

    # First run random (warmup), then Gemini-guided
    # Fleet already has hundreds of results, no long warmup needed
    run_idx = completed
    while True:
        run_idx += 1
        use_ai = run_idx > (completed + 1)

        # Swarm sync: pull fleet results before asking Gemini
        if use_ai:
            try:
                sync_fleet_results()
            except Exception as e:
                log(f"Sync warning: {e}")

        config = sample_config(rng, use_ai=use_ai)

        log(f"\n{'='*60}")
        log(f"Experiment {run_idx} (fleet: {_count_fleet_results()} results)")
        log(f"Best so far: {get_best_metric():.4f}")
        log(f"{'='*60}")

        try:
            run_experiment(config, run_idx)
        except Exception as e:
            log(f"Unexpected error in run {run_idx}: {e}")

        # Push results to hub after each experiment
        try:
            _push_results_to_hub()
        except Exception:
            pass

        time.sleep(10)


if __name__ == "__main__":
    main()
