"""
Loading the causRCA dataset.

causRCA records an industrial CNC vertical lathe. It contains three things we
need:

  1. Normal recordings  - 170 captures of the machine running healthily.
  2. Fault recordings   - 100 captures where a fault was deliberately injected
                          into a digital twin, so we know exactly which
                          variable was tampered with and when.
  3. An expert graph    - a causal graph drawn by engineers who know the
                          machine. This is our ground truth.

Every recording is a CHANGE LOG, not a regular time series: a row is written
only at the moment a variable changes value. Turning that into a table is the
job of preprocess.py.

Source: Mehling, Pieper and Lueke (2025), causRCA, Zenodo 10.5281/zenodo.15876410
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import paths

# The subsystems causRCA splits the machine into, and how many variables each
# one covers according to the dataset documentation.
SUBSYSTEMS = ("coolant", "hydraulics", "probe", "full")


@dataclass
class FaultRun:
    """One recording where a fault was injected, with everything we know about it."""

    experiment: str                 # e.g. "exp_22"
    run: str                        # e.g. "run_2"
    path: Path                      # the CSV holding the change log
    caused_by: list[str]            # variables the digital twin tampered with
    forced_states: dict             # what each of them was forced to, as labelled
    expected_alarms: list[str]      # alarms the engineers expected to fire
    diagnosis: str                  # plain-English fault description
    fault_started_at: float         # seconds into the recording
    alarms_fired_at: float | None
    diagnosed_at: float | None
    fault_ended_at: float | None

    @property
    def name(self) -> str:
        return f"{self.experiment}/{self.run}"


@dataclass
class ExpertGraph:
    """The engineers' causal graph for one subsystem - our ground truth."""

    variables: pd.DataFrame         # one row per variable, with description
    edges: pd.DataFrame             # one row per causal link, with mechanism
    subsystem: str

    @property
    def variable_names(self) -> list[str]:
        return self.variables["label"].tolist()

    @property
    def edge_pairs(self) -> set[tuple[str, str]]:
        """Edges as (cause, effect) pairs, which is what scoring compares."""
        return set(zip(self.edges["source_label"], self.edges["target_label"]))

    @property
    def alarm_names(self) -> list[str]:
        return self.variables.loc[self.variables["type"] == "Alarm", "label"].tolist()

    def describe(self, variable: str) -> str:
        """The engineers' own description of a variable, for the dashboard."""
        match = self.variables.loc[self.variables["label"] == variable, "comment"]
        return match.iloc[0] if len(match) else variable


@dataclass
class Dataset:
    """Everything needed to run the pipeline on one subsystem."""

    subsystem: str
    expert_graph: ExpertGraph
    fault_runs: list[FaultRun]
    normal_recordings: list[Path] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "subsystem": self.subsystem,
            "variables": len(self.expert_graph.variable_names),
            "expert_edges": len(self.expert_graph.edges),
            "fault_runs": len(self.fault_runs),
            "fault_scenarios": len({r.experiment for r in self.fault_runs}),
            "distinct_diagnoses": len({r.diagnosis for r in self.fault_runs}),
            "causing_variables": len({v for r in self.fault_runs for v in r.caused_by}),
            "normal_recordings": len(self.normal_recordings),
        }


def load_expert_graph(subsystem: str) -> ExpertGraph:
    """Read the engineers' causal graph for one subsystem."""
    folder = paths.subsystem_dir(subsystem)
    if subsystem == "full":
        variables = pd.read_csv(folder / "all_nodes.csv")
        edges = pd.read_csv(folder / "all_edges.csv")
    else:
        variables = pd.read_csv(folder / f"{subsystem}_nodes.csv")
        edges = pd.read_csv(folder / f"{subsystem}_edges.csv")
    return ExpertGraph(variables=variables, edges=edges, subsystem=subsystem)


def forced_states(description: dict) -> dict:
    """
    What each tampered variable was forced to, as the dataset labels it.

    The states live inside the diagnoses, not next to `manipulatedVars`, and a
    few scenarios list two diagnoses that each force a different variable
    (exp_15, exp_18). So we merge them. Values are kept exactly as written:
    True/False for switches, a number, or a condition such as "< 6600" for
    hydraulic pressure.
    """
    states = {}
    for diagnosis in description.get("diagnoses", []):
        states.update(diagnosis.get("variablesStates", {}))
    return {name: states[name] for name in description["manipulatedVars"] if name in states}


def load_fault_runs(subsystem: str) -> list[FaultRun]:
    """Read every fault recording for a subsystem, with its ground-truth labels."""
    if subsystem == "full":
        folders = [paths.FAULT_RECORDINGS_DIR / f"exp_{s}"
                   for s in ("coolant", "hydraulics", "probe")]
    else:
        folders = [paths.subsystem_dir(subsystem)]

    runs: list[FaultRun] = []
    for folder in folders:
        for experiment_dir in sorted(folder.glob("exp_*")):
            if not experiment_dir.is_dir():
                continue
            description_file = experiment_dir / f"{experiment_dir.name}_description.json"
            if not description_file.exists():
                continue
            description = json.loads(description_file.read_text())

            for run_dir in sorted(experiment_dir.glob("run_*")):
                csv_files = list(run_dir.glob("*.csv"))
                timing_file = run_dir / "causes.json"
                if not csv_files or not timing_file.exists():
                    continue
                timing = json.loads(timing_file.read_text())

                runs.append(FaultRun(
                    experiment=experiment_dir.name,
                    run=run_dir.name,
                    path=csv_files[0],
                    caused_by=description["manipulatedVars"],
                    forced_states=forced_states(description),
                    expected_alarms=description.get("alarms", []),
                    diagnosis=description["diagnoses"][0]["name"],
                    fault_started_at=timing["cause_start_at"],
                    alarms_fired_at=timing.get("alarms_detected_at"),
                    diagnosed_at=timing.get("diagnosis_at"),
                    fault_ended_at=timing.get("cause_end_at"),
                ))
    return runs


def load_normal_recordings(limit: int | None = None) -> list[Path]:
    """Paths to the healthy-operation recordings, oldest capture id first."""
    recordings = sorted(paths.NORMAL_RECORDINGS_DIR.glob("dataset_real_op_*.csv"))
    return recordings[:limit] if limit else recordings


def read_change_log(path: Path) -> pd.DataFrame:
    """
    Read one recording as it is stored: a list of value changes.

    Columns are time_s, node, value, type. One row per change, so a variable
    that never changed during the recording simply does not appear.
    """
    log = pd.read_csv(path)
    log["time_s"] = log["time_s"].astype(float)
    return log


def load(subsystem: str = "coolant", normal_recordings: int | None = None) -> Dataset:
    """Load everything the pipeline needs for one subsystem."""
    if subsystem not in SUBSYSTEMS:
        raise ValueError(f"Unknown subsystem '{subsystem}'. Choose from {SUBSYSTEMS}.")
    return Dataset(
        subsystem=subsystem,
        expert_graph=load_expert_graph(subsystem),
        fault_runs=load_fault_runs(subsystem),
        normal_recordings=load_normal_recordings(normal_recordings),
    )
