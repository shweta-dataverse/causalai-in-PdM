"""Loaders for the causRCA dataset (CNC vertical lathe, Fraunhofer IWU).

Source : https://doi.org/10.5281/zenodo.15876410 (Apache-2.0)
Layout : data/causrca/raw/
           real_op/                     170 normal-operation event logs
           dig_twin/exp_{group}/exp_*/  fault experiments (run_*/ per run)
           expert_graph/                ground-truth causal graph (GML + CSV)

Every recording is an *event log* in long format with columns
(time_s, node, value, type) - a row is written only when a variable changes.
`to_wide` resamples an event log onto a regular grid so that standard
time-series methods (e.g. PCMCI) can be applied.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from .config import CAUSRCA_RAW

FAULT_GROUPS = ("coolant", "hydraulics", "probe")


# ---------------------------------------------------------------------------
# Event logs
# ---------------------------------------------------------------------------

def list_real_op_files(root: Path = CAUSRCA_RAW) -> list[Path]:
    """Return all normal-operation recordings, sorted by name."""
    return sorted((root / "real_op").glob("dataset_real_op_*.csv"))


def list_fault_runs(root: Path = CAUSRCA_RAW) -> pd.DataFrame:
    """Index every fault run with its experiment metadata.

    Returns one row per run with the experiment id, fault group, the
    intervened (manipulated) variables, expected alarms and the labelled
    diagnosis - i.e. the ground truth needed for RCA evaluation.
    """
    records = []
    for group in FAULT_GROUPS:
        for exp_dir in sorted((root / "dig_twin" / f"exp_{group}").glob("exp_*")):
            if not exp_dir.is_dir():
                continue
            meta = json.loads((exp_dir / f"{exp_dir.name}_description.json").read_text())
            for run_dir in sorted(exp_dir.glob("run_*")):
                csv_files = list(run_dir.glob("*.csv"))
                if not csv_files:
                    continue
                causes = json.loads((run_dir / "causes.json").read_text())
                records.append(
                    {
                        "group": group,
                        "exp_id": meta["exp_id"],
                        "run": run_dir.name,
                        "csv_path": csv_files[0],
                        "manipulated_vars": meta["manipulatedVars"],
                        "alarms": meta["alarms"],
                        "diagnosis": meta["diagnoses"][0]["name"],
                        "cause_start_s": causes["cause_start_at"],
                        "alarms_detected_s": causes.get("alarms_detected_at"),
                        "diagnosis_s": causes.get("diagnosis_at"),
                        "cause_end_s": causes.get("cause_end_at"),
                    }
                )
    return pd.DataFrame.from_records(records)


def load_event_log(path: Path) -> pd.DataFrame:
    """Load one recording in long (event) format."""
    df = pd.read_csv(path)
    df["time_s"] = df["time_s"].astype(float)
    return df


def to_wide(events: pd.DataFrame, freq_s: float = 1.0) -> pd.DataFrame:
    """Resample an event log onto a regular grid (one column per variable).

    Values are forward-filled between change events, which matches the OPC UA
    semantics of the recording (a variable keeps its value until the next
    published change). Booleans are mapped to {0, 1}; categorical values are
    label-encoded per column so the result is fully numeric.
    """
    events = events.copy()
    events["value"] = events["value"].map(
        lambda v: {"True": 1, "False": 0}.get(str(v), v)
    )

    wide = events.pivot_table(
        index="time_s", columns="node", values="value", aggfunc="last"
    )

    grid = np.arange(0.0, events["time_s"].max() + freq_s, freq_s)
    wide = (
        wide.reindex(wide.index.union(grid))
        .ffill()
        .infer_objects(copy=False)
        .loc[grid]
    )

    for col in wide.columns:
        if wide[col].dtype == object:
            wide[col] = pd.Categorical(wide[col]).codes.astype(float)
        else:
            wide[col] = pd.to_numeric(wide[col], errors="coerce")
    wide.index.name = "time_s"
    return wide


# ---------------------------------------------------------------------------
# Ground-truth causal graph
# ---------------------------------------------------------------------------

def load_expert_graph(root: Path = CAUSRCA_RAW) -> nx.DiGraph:
    """Load the expert-derived causal graph (ground truth, 92 nodes)."""
    return nx.read_gml(root / "expert_graph" / "expert_graph.gml")


def load_node_table(root: Path = CAUSRCA_RAW) -> pd.DataFrame:
    """Node metadata: subsystem path, type (Alarm/Variable), dtype, comment."""
    return pd.read_csv(root / "expert_graph" / "all_nodes.csv")


def load_edge_table(root: Path = CAUSRCA_RAW) -> pd.DataFrame:
    """Edge list with human-readable mechanism descriptions."""
    return pd.read_csv(root / "expert_graph" / "all_edges.csv")
