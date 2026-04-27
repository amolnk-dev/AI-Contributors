"""Data utilities: loading, splitting, cross-validation."""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path


def load_json(path: str) -> list | dict:
    with open(path) as f:
        return json.load(f)


def save_json(data: list | dict, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def train_val_split(data, val_ratio: float = 0.2, seed: int = 42, stratify=None):
    """Split data into train/val sets."""
    from sklearn.model_selection import train_test_split
    return train_test_split(data, test_size=val_ratio, random_state=seed, stratify=stratify)


def kfold_split(n_samples: int, k: int = 5, seed: int = 42):
    """Generate k-fold indices."""
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    return list(kf.split(range(n_samples)))


def stratified_kfold_split(labels: np.ndarray, k: int = 5, seed: int = 42):
    """Generate stratified k-fold indices."""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(skf.split(range(len(labels)), labels))


def list_files(directory: str, extensions: list[str] | None = None) -> list[str]:
    """List files in directory, optionally filtered by extension."""
    path = Path(directory)
    files = []
    for f in sorted(path.rglob("*")):
        if f.is_file():
            if extensions is None or f.suffix.lower() in extensions:
                files.append(str(f))
    return files
