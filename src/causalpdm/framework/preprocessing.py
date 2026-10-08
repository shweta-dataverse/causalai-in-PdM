"""Module 1 — Data understanding & preprocessing.

Turns the raw C-MAPSS text files into a clean multivariate time series with
per-unit RUL labels, ready for feature engineering and causal discovery.

Design notes
------------
* Constant sensors carry no information and are removed: they would produce
  undefined correlations and degenerate conditional-independence tests.
* Normalisation is done **per unit** (z-score). Engines start with different
  unknown initial wear; per-unit centring removes that between-engine offset,
  which would otherwise appear as a spurious common cause.
* RUL is derived from the run-to-failure trajectory length. It is kept as a
  *label*, deliberately outside the causal variable set (see `features`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import cmapss

# Columns that are never causal variables: identifiers and labels.
NON_FEATURE_COLS = ("unit", "cycle", "rul")


def load_raw(subset: str = "FD001") -> pd.DataFrame:
    """Load a C-MAPSS training subset with per-row RUL already attached."""
    return cmapss.load_train(subset)


def drop_constant_channels(df: pd.DataFrame, tol: float = 1e-8) -> tuple[pd.DataFrame, list[str]]:
    """Remove sensor/setting channels with (near-)zero variance.

    Returns the reduced frame and the list of dropped column names.
    """
    candidates = [c for c in df.columns if c not in NON_FEATURE_COLS]
    stds = df[candidates].std()
    dropped = stds[stds <= tol].index.tolist()
    return df.drop(columns=dropped), dropped


def missing_value_report(df: pd.DataFrame) -> pd.Series:
    """Share of missing values per column (C-MAPSS is expected to be complete)."""
    return df.isna().mean().sort_values(ascending=False)


def normalise_per_unit(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Z-score each channel within each engine.

    Per-unit rather than fleet-wide standardisation removes engine-specific
    offsets caused by unknown initial wear, so downstream causal analysis sees
    within-engine dynamics instead of between-engine level differences.
    """
    out = df.copy()
    grouped = out.groupby("unit")[cols]
    out[cols] = (out[cols] - grouped.transform("mean")) / grouped.transform("std").replace(0, np.nan)
    out[cols] = out[cols].fillna(0.0)
    return out


def detrend_per_unit(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Remove a linear trend per channel per engine.

    Degradation is a slowly-moving *latent* common driver of every sensor. Left
    in place it violates both stationarity and causal sufficiency (the two core
    PCMCI assumptions) and floods the discovered graph with trend-induced
    edges. Removing the linear component isolates the sensor-to-sensor
    structure; the degradation signal itself is handled separately by the
    health indicator in Module 2.
    """
    out = df.copy()

    def _detrend(block: pd.DataFrame) -> pd.DataFrame:
        t = np.arange(len(block), dtype=float)
        t_centred = t - t.mean()
        denom = (t_centred**2).sum()
        if denom == 0:
            return block
        for c in cols:
            y = block[c].to_numpy(dtype=float)
            slope = (t_centred * (y - y.mean())).sum() / denom
            block[c] = y - slope * t_centred
        return block

    return out.groupby("unit", group_keys=False)[out.columns.tolist()].apply(_detrend)


def make_windows(
    df: pd.DataFrame, feature_cols: list[str], window: int = 30, horizon: int = 1
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build sliding windows per engine for sequence models.

    Returns
    -------
    X : array (n_windows, window, n_features)
    y : array (n_windows,) - RUL at the end of each window + horizon offset
    meta : frame with the unit and end-cycle of every window
    """
    xs, ys, meta = [], [], []
    for unit, block in df.groupby("unit"):
        values = block[feature_cols].to_numpy(dtype=np.float32)
        rul = block["rul"].to_numpy(dtype=np.float32)
        cycles = block["cycle"].to_numpy()
        for end in range(window, len(block) - horizon + 1):
            xs.append(values[end - window : end])
            ys.append(rul[end + horizon - 1])
            meta.append({"unit": unit, "end_cycle": cycles[end - 1]})
    return np.asarray(xs), np.asarray(ys), pd.DataFrame(meta)


def run(subset: str = "FD001", detrend: bool = True) -> dict:
    """Execute Module 1 end to end.

    Produces **two** representations of the same data, used at different stages,
    because they serve incompatible purposes:

    ``normalised``
        Per-unit z-scored, degradation trend intact. Used for the health
        indicator, SCM mechanism fitting, RCA deviation scoring and all
        simulation. The trend must be present here - it is the degradation
        signal that the pipeline is ultimately explaining.
    ``detrended``
        Per-unit z-scored *and* linearly detrended. Used only for causal
        structure learning, where the shared trend would otherwise act as an
        unobserved common cause and flood the graph with spurious edges.

    Separating the two keeps discovery statistically sound without making the
    downstream health readout meaningless, and every stage documents which
    representation it consumes.
    """
    raw = load_raw(subset)
    clean, dropped = drop_constant_channels(raw)
    feature_cols = [c for c in clean.columns if c not in NON_FEATURE_COLS]

    normalised = normalise_per_unit(clean, feature_cols)
    detrended = detrend_per_unit(normalised, feature_cols) if detrend else normalised

    return {
        "raw": raw,
        "clean": clean,
        "normalised": normalised,
        "detrended": detrended,
        "feature_cols": feature_cols,
        "dropped_constant": dropped,
        "n_units": int(clean["unit"].nunique()),
        "n_rows": int(len(clean)),
        "missing_share": float(raw.isna().mean().mean()),
        "detrend_applied": detrend,
        "subset": subset,
    }
