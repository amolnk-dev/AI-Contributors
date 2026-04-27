"""Cross-task ML utilities: metrics, encoding, timing, ensembles."""

import base64
import io
import time
import functools
import numpy as np
from PIL import Image


# --- Image encoding/decoding ---

def encode_image_b64(image: Image.Image, fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def decode_image_b64(b64_string: str) -> Image.Image:
    image_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_bytes))


# --- Metrics ---

def dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Dice coefficient for binary segmentation."""
    pred_flat = pred.flatten().astype(bool)
    target_flat = target.flatten().astype(bool)
    intersection = (pred_flat & target_flat).sum()
    return float((2 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))


def accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    return float((pred == target).mean())


def f1_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Binary F1 score."""
    tp = ((pred == 1) & (target == 1)).sum()
    fp = ((pred == 1) & (target == 0)).sum()
    fn = ((pred == 0) & (target == 1)).sum()
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    return float(2 * precision * recall / (precision + recall + 1e-9))


# --- Timing ---

def log_if_slow(max_seconds: float):
    """Decorator to warn if a function exceeds a time limit (does NOT kill it)."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            if elapsed > max_seconds:
                print(f"WARNING: {func.__name__} took {elapsed:.2f}s (limit: {max_seconds}s)")
            return result
        return wrapper
    return decorator


def timer(func):
    """Simple timing decorator."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__}: {elapsed:.3f}s")
        return result
    return wrapper


# --- Threshold optimization ---

def optimize_threshold(probs: np.ndarray, targets: np.ndarray,
                       metric_fn=accuracy, n_steps: int = 100) -> tuple[float, float]:
    """Search for optimal classification threshold."""
    best_threshold = 0.5
    best_score = 0.0
    for t in np.linspace(0, 1, n_steps):
        preds = (probs >= t).astype(int)
        score = metric_fn(preds, targets)
        if score > best_score:
            best_score = score
            best_threshold = t
    return best_threshold, best_score


# --- Ensemble ---

def weighted_average(predictions: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Weighted average of predictions."""
    if weights is None:
        weights = [1.0 / len(predictions)] * len(predictions)
    result = np.zeros_like(predictions[0], dtype=float)
    for pred, w in zip(predictions, weights):
        result += w * pred
    return result


def majority_vote(predictions: list[np.ndarray]) -> np.ndarray:
    """Majority vote across predictions."""
    stacked = np.stack(predictions, axis=0)
    from scipy import stats
    mode_result = stats.mode(stacked, axis=0, keepdims=False)
    return mode_result.mode
