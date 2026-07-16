"""Loaders for the SCANIA Component X dataset.

Source : Research Data Sweden, https://doi.org/10.5878/jvb5-d390 (CC BY 4.0)
         Kharazian et al. (2025), Scientific Data 12:493.
Layout : data/scania/raw/{train,validation,test}_operational_readouts.csv
         {train,validation,test}_specifications.csv
         train_tte.csv, {validation,test}_labels.csv

Operational readouts: irregular snapshots per vehicle (anonymised time_step),
107 columns = 1 numerical counter block + 6 histogram feature families.
All feature names and values are anonymised by the data owner.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import SCANIA_RAW

SPLITS = ("train", "validation", "test")

# Anonymised feature families as documented in the accompanying paper:
# 6 histogram variables (10-36 bins each) and 8 numerical counters.
HISTOGRAM_PREFIXES = ("167", "272", "291", "158", "459", "397")


def load_readouts(split: str = "train", root: Path = SCANIA_RAW) -> pd.DataFrame:
    """Operational readouts with memory-efficient dtypes (~1.1 GB raw CSV)."""
    path = root / f"{split}_operational_readouts.csv"
    header = pd.read_csv(path, nrows=0).columns
    dtypes = {c: np.float32 for c in header}
    dtypes["vehicle_id"] = np.int32
    return pd.read_csv(path, dtype=dtypes)


def load_specifications(split: str = "train", root: Path = SCANIA_RAW) -> pd.DataFrame:
    """Categorical vehicle specifications (anonymised, Spec_0..Spec_7)."""
    return pd.read_csv(root / f"{split}_specifications.csv")


def load_tte(root: Path = SCANIA_RAW) -> pd.DataFrame:
    """Train time-to-event table: study length and repair indicator."""
    return pd.read_csv(root / "train_tte.csv")


def load_labels(split: str, root: Path = SCANIA_RAW) -> pd.DataFrame:
    """Validation/test class labels (5 classes: time windows to failure)."""
    if split not in ("validation", "test"):
        raise ValueError("labels exist only for 'validation' and 'test'")
    return pd.read_csv(root / f"{split}_labels.csv")


def feature_families(columns) -> pd.Series:
    """Map readout columns to their anonymised feature family (e.g. '167')."""
    cols = [c for c in columns if c not in ("vehicle_id", "time_step")]
    return pd.Series({c: c.split("_")[0] for c in cols}, name="family")
