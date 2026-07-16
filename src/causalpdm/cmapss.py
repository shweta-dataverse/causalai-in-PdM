"""Loaders for the NASA C-MAPSS turbofan degradation dataset.

Source : NASA Ames Prognostics Center of Excellence, data set #6
         (Saxena & Goebel 2008), public domain.
Layout : data/cmapss/raw/{train,test}_FD00[1-4].txt, RUL_FD00[1-4].txt

Each row is one operational cycle of one engine: 26 space-separated columns
(unit, cycle, 3 operational settings, 21 sensor measurements).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import CMAPSS_RAW

SUBSETS = ("FD001", "FD002", "FD003", "FD004")

SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *SETTING_COLS, *SENSOR_COLS]

# Physical meaning of each sensor channel (Saxena et al. 2008, Table 1).
SENSOR_DESCRIPTIONS = {
    "s_1": "Total temperature at fan inlet (T2, degR)",
    "s_2": "Total temperature at LPC outlet (T24, degR)",
    "s_3": "Total temperature at HPC outlet (T30, degR)",
    "s_4": "Total temperature at LPT outlet (T50, degR)",
    "s_5": "Pressure at fan inlet (P2, psia)",
    "s_6": "Total pressure in bypass-duct (P15, psia)",
    "s_7": "Total pressure at HPC outlet (P30, psia)",
    "s_8": "Physical fan speed (Nf, rpm)",
    "s_9": "Physical core speed (Nc, rpm)",
    "s_10": "Engine pressure ratio (epr, P50/P2)",
    "s_11": "Static pressure at HPC outlet (Ps30, psia)",
    "s_12": "Ratio of fuel flow to Ps30 (phi, pps/psi)",
    "s_13": "Corrected fan speed (NRf, rpm)",
    "s_14": "Corrected core speed (NRc, rpm)",
    "s_15": "Bypass ratio (BPR)",
    "s_16": "Burner fuel-air ratio (farB)",
    "s_17": "Bleed enthalpy (htBleed)",
    "s_18": "Demanded fan speed (Nf_dmd, rpm)",
    "s_19": "Demanded corrected fan speed (PCNfR_dmd, rpm)",
    "s_20": "HPT coolant bleed (W31, lbm/s)",
    "s_21": "LPT coolant bleed (W32, lbm/s)",
}


def _read_txt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df.columns = COLUMNS
    return df


def load_train(subset: str = "FD001", root: Path = CMAPSS_RAW) -> pd.DataFrame:
    """Run-to-failure trajectories with a computed RUL column per row."""
    df = _read_txt(root / f"train_{subset}.txt")
    df["rul"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    return df


def load_test(subset: str = "FD001", root: Path = CMAPSS_RAW) -> pd.DataFrame:
    """Truncated trajectories (end before failure); labels via load_rul."""
    return _read_txt(root / f"test_{subset}.txt")


def load_rul(subset: str = "FD001", root: Path = CMAPSS_RAW) -> pd.Series:
    """True RUL at the last observed cycle of each test engine."""
    rul = pd.read_csv(root / f"RUL_{subset}.txt", header=None)[0]
    rul.index = rul.index + 1  # unit numbers are 1-based
    rul.name = "rul"
    return rul
