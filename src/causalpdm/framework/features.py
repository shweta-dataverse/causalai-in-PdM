"""Module 2 — Feature engineering & health indicator generation.

Two kinds of derived variable are produced here, and the distinction is
methodologically load-bearing:

**Causal-safe features** (rolling mean/std, slope, derivatives) are lagged or
windowed transforms of a single channel. They can enter the causal variable set
because each is a legitimate measurement-level quantity.

**Learned composite indicators** (the health indicator) are functions of *many*
channels at once. They must NOT enter the causal graph: a variable that is a
deterministic function of other nodes violates the faithfulness assumption that
PCMCI and every other constraint-based discovery algorithm relies on, and if
the composite is fitted against RUL it also leaks the label into the structure.

The health indicator is therefore used only for (a) defining the failure event
that Module 6 explains and (b) the dashboard health view.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def add_rolling_features(
    df: pd.DataFrame, cols: list[str], window: int = 15
) -> tuple[pd.DataFrame, list[str]]:
    """Append rolling mean and rolling standard deviation per engine."""
    out = df.copy()
    new_cols: list[str] = []
    grouped = out.groupby("unit")
    for c in cols:
        mean_col, std_col = f"{c}_rmean{window}", f"{c}_rstd{window}"
        out[mean_col] = grouped[c].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        out[std_col] = grouped[c].transform(
            lambda s: s.rolling(window, min_periods=2).std()
        ).fillna(0.0)
        new_cols += [mean_col, std_col]
    return out, new_cols


def add_trend_features(
    df: pd.DataFrame, cols: list[str], window: int = 15
) -> tuple[pd.DataFrame, list[str]]:
    """Append a rolling ordinary-least-squares slope per engine.

    The slope of a channel is the classic degradation-rate feature: it captures
    *where the signal is heading* rather than where it currently is.
    """
    out = df.copy()
    new_cols: list[str] = []

    def _slope(series: pd.Series) -> pd.Series:
        def one(vals: np.ndarray) -> float:
            if len(vals) < 2:
                return 0.0
            t = np.arange(len(vals), dtype=float)
            tc = t - t.mean()
            denom = (tc**2).sum()
            return float((tc * (vals - vals.mean())).sum() / denom) if denom else 0.0

        return series.rolling(window, min_periods=2).apply(one, raw=True).fillna(0.0)

    grouped = out.groupby("unit")
    for c in cols:
        col = f"{c}_slope{window}"
        out[col] = grouped[c].transform(_slope)
        new_cols.append(col)
    return out, new_cols


def add_derivatives(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Append first and second discrete derivatives per engine."""
    out = df.copy()
    new_cols: list[str] = []
    grouped = out.groupby("unit")
    for c in cols:
        d1, d2 = f"{c}_d1", f"{c}_d2"
        out[d1] = grouped[c].transform(lambda s: s.diff()).fillna(0.0)
        out[d2] = out.groupby("unit")[d1].transform(lambda s: s.diff()).fillna(0.0)
        new_cols += [d1, d2]
    return out, new_cols


class HealthReadout:
    """Deterministic readout mapping sensor values to a health-loss score.

    Architecturally this is a *measurement function*, not a causal node: it is
    fitted once on observational data and thereafter applied to whatever sensor
    trajectory it is given - factual, interventional or counterfactual. That is
    what lets Modules 7 and 8 report "effect on machine health" while keeping
    the composite out of the causal graph entirely.
    """

    def __init__(self, cols: list[str], smooth: int = 15):
        self.cols = list(cols)
        self.smooth = smooth
        self._pca = PCA(n_components=1, random_state=0)
        self._sign = 1.0
        self._lo = 0.0
        self._hi = 1.0
        self.explained_variance_ratio_ = float("nan")

    def fit(self, df: pd.DataFrame) -> "HealthReadout":
        comp = self._pca.fit_transform(df[self.cols].to_numpy(dtype=float))[:, 0]
        self.explained_variance_ratio_ = float(self._pca.explained_variance_ratio_[0])
        # Orient so higher = more degraded. RUL is used only to fix the sign,
        # never as a regression target, so no label information enters the score.
        if "rul" in df.columns:
            corr = np.corrcoef(comp, df["rul"].to_numpy(dtype=float))[0, 1]
            self._sign = -1.0 if corr > 0 else 1.0
        raw = self._smooth(pd.Series(self._sign * comp, index=df.index), df.get("unit"))
        self._lo, self._hi = float(raw.min()), float(raw.max())
        return self

    def _smooth(self, series: pd.Series, units) -> pd.Series:
        if units is None:
            return series.rolling(self.smooth, min_periods=1).mean()
        return series.groupby(units).transform(
            lambda s: s.rolling(self.smooth, min_periods=1).mean()
        )

    def transform(self, df: pd.DataFrame, units=None) -> pd.Series:
        comp = self._pca.transform(df[self.cols].to_numpy(dtype=float))[:, 0]
        raw = self._smooth(pd.Series(self._sign * comp, index=df.index), units)
        span = self._hi - self._lo
        scaled = (raw - self._lo) / span if span > 0 else raw * 0.0
        return scaled.rename("health_indicator")

    def fit_transform(self, df: pd.DataFrame) -> pd.Series:
        return self.fit(df).transform(df, units=df.get("unit"))

    @property
    def meta(self) -> dict:
        return {
            "explained_variance_ratio": self.explained_variance_ratio_,
            "n_input_channels": len(self.cols),
            "smoothing_window": self.smooth,
            "excluded_from_causal_graph": True,
        }


def build_health_indicator(
    df: pd.DataFrame, cols: list[str], smooth: int = 15
) -> tuple[pd.Series, HealthReadout]:
    """Fit the health readout and return the indicator for the observed data."""
    readout = HealthReadout(cols, smooth=smooth)
    return readout.fit_transform(df), readout


def define_failure_event(
    df: pd.DataFrame, hi: pd.Series, quantile: float = 0.9
) -> tuple[pd.DataFrame, float]:
    """Flag the onset of the failure event that Module 6 will explain.

    Root cause analysis needs an explicit *event to explain*. C-MAPSS has no
    per-engine fault label, so the event is defined as the first cycle at which
    the health indicator crosses a fleet-level threshold. Making this definition
    explicit matters: RCA results are conditional on what counts as the anomaly.
    """
    threshold = float(hi.quantile(quantile))
    out = df.copy()
    out["health_indicator"] = hi
    out["is_degraded"] = (hi >= threshold).astype(int)
    out["onset"] = 0
    for _, block in out.groupby("unit"):
        degraded = block.index[block["is_degraded"] == 1]
        if len(degraded):
            out.loc[degraded[0], "onset"] = 1
    return out, threshold


def run(prepared: pd.DataFrame, base_cols: list[str], window: int = 15) -> dict:
    """Execute Module 2 end to end."""
    df, rolling_cols = add_rolling_features(prepared, base_cols, window)
    df, trend_cols = add_trend_features(df, base_cols, window)
    df, deriv_cols = add_derivatives(df, base_cols)

    hi, readout = build_health_indicator(df, base_cols)
    df, threshold = define_failure_event(df, hi)

    return {
        "featured": df,
        "base_cols": base_cols,
        "engineered_cols": rolling_cols + trend_cols + deriv_cols,
        "health_readout": readout,
        "health_indicator_meta": readout.meta,
        "failure_threshold": threshold,
        "n_engines_with_onset": int(df.groupby("unit")["onset"].max().sum()),
    }
