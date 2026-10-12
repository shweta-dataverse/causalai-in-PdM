#!/usr/bin/env python3
"""
Causal discovery on causRCA, end to end, in one file.

    python scripts/causal_discovery_end_to_end.py

Steps: install -> settings -> download -> load -> preprocess -> causal
discovery -> evaluation against the engineers' graph.

There is no "training with epochs" here, on purpose. PCMCI+ is not a neural
network: it has no weights and no loss. It learns the causal graph in one pass
by running statistical tests ("are A and B still related once C is known?").
So instead of epochs, the progress shown is the number of tests run. There is
also no train/test split for the graph: it is learned from the recordings and
then evaluated against the expert graph, which is the ground truth.

Self-contained (does not import the project package) and uses the same
settings as the frozen results, so coolant gives graph F1 0.737.
"""

# =============================================================================
# 0. INSTALL
# =============================================================================
# Run once in a terminal:
#     pip install numpy pandas scikit-learn networkx tigramite
#
# The check below stops with a clear message if anything is missing.

import importlib.util
import sys

for package in ("numpy", "pandas", "sklearn", "networkx", "tigramite"):       # every library this file needs
    if importlib.util.find_spec(package) is None:                             # not installed?
        sys.exit(f"Missing library '{package}'. Run: pip install numpy pandas "
                 f"scikit-learn networkx tigramite")

import io                                                                     # in-memory download
import json                                                                   # fault descriptions are JSON
import time                                                                   # timing the discovery
import urllib.request                                                         # downloading the dataset
import warnings                                                               # silencing library chatter
import zipfile                                                                # the dataset comes zipped
from collections import Counter, defaultdict                                  # counting values, grouping
from pathlib import Path                                                      # file paths

import numpy as np                                                            # arrays
import pandas as pd                                                           # tables
from tigramite import data_processing as pp                                   # tigramite's data container
from tigramite.independence_tests.gsquared import Gsquared                    # G-test for on/off data
from tigramite.pcmci import PCMCI                                             # the discovery algorithm

warnings.filterwarnings("ignore")                                             # keep the output readable


# =============================================================================
# 1. SETTINGS - the same values as settings.py (frozen results)
# =============================================================================

SUBSYSTEM = "coolant"            # "coolant", "hydraulics" or "probe"
HEALTHY_RECORDINGS = 60          # how many normal-operation recordings to use
SECONDS_PER_ROW = 2.5            # resampling grid
MIN_ROWS = 20                    # skip recordings shorter than this
MAX_LAG = 3                      # causes may act up to 3 rows (7.5 s) later
SIGNIFICANCE = 0.001             # how much evidence a link needs
SAME_MOMENT_PARTNERS = 5         # screening: same-step links only to the 5 strongest partners
CAUSES_CONDITIONED = 3           # final tests condition on at most 3 causes of each end
MAX_CONDITIONS = 3               # first phase: at most 3 explaining signals per test
MAX_COMBINATIONS = 2             # first phase: at most 2 combinations per size

PROJECT_ROOT = Path(__file__).resolve().parents[1]                            # repository folder
DATA = PROJECT_ROOT / "data" / "causrca" / "raw"                              # where the dataset lives
OUTPUT = PROJECT_ROOT / "results" / "end_to_end"                              # where this script writes
DATASET_URL = "https://zenodo.org/api/records/15876410/files/dataset_causRCA.zip/content"

# Published graph F1 (Mehling et al. 2026, Table 3), best algorithm per subsystem
PUBLISHED_BEST = {"coolant": 0.29, "hydraulics": 0.40, "probe": 0.37}


# =============================================================================
# 2. DOWNLOAD - only if the data is not there yet
# =============================================================================

def download_dataset() -> None:
    if (DATA / "expert_graph").exists():                                      # already downloaded
        return
    print("Downloading causRCA (about 3.6 MB)...")
    DATA.mkdir(parents=True, exist_ok=True)                                   # make the folder
    with urllib.request.urlopen(DATASET_URL) as response:                     # fetch the zip
        zipfile.ZipFile(io.BytesIO(response.read())).extractall(DATA)         # unpack it in place


# =============================================================================
# 3. LOAD - expert graph, fault recordings, healthy recordings
# =============================================================================

def load_expert_graph(subsystem: str):
    folder = DATA / "dig_twin" / f"exp_{subsystem}"                           # subsystem folder
    nodes = pd.read_csv(folder / f"{subsystem}_nodes.csv")                    # one row per signal
    edges = pd.read_csv(folder / f"{subsystem}_edges.csv")                    # one row per expert link
    signals = nodes["label"].tolist()                                         # signal names
    alarms = nodes.loc[nodes["type"] == "Alarm", "label"].tolist()            # which of them are alarms
    expert_links = set(zip(edges["source_label"], edges["target_label"]))     # (cause, effect) pairs
    return signals, alarms, expert_links


def load_fault_runs(subsystem: str) -> list[dict]:
    runs = []
    for experiment in sorted((DATA / "dig_twin" / f"exp_{subsystem}").glob("exp_*")):  # each fault scenario
        description = experiment / f"{experiment.name}_description.json"
        if not description.exists():                                         # skip anything that is not a scenario
            continue
        for run in sorted(experiment.glob("run_*")):                         # each repeat of the scenario
            timing = json.loads((run / "causes.json").read_text())           # when the fault started
            runs.append({"name": f"{experiment.name}/{run.name}",
                         "path": next(run.glob("*.csv")),                     # the recording itself
                         "fault_started_at": timing["cause_start_at"]})
    return runs


def load_healthy_recordings(limit: int) -> list[Path]:
    return sorted((DATA / "real_op").glob("dataset_real_op_*.csv"))[:limit]  # oldest first


def read_change_log(path: Path) -> pd.DataFrame:
    log = pd.read_csv(path)                                                   # columns: time_s, node, value, type
    log["time_s"] = log["time_s"].astype(float)                               # time as a number
    return log


# =============================================================================
# 4. PREPROCESS - change logs -> regular tables
# =============================================================================
# A recording only has a row when a signal CHANGES. Discovery needs one row
# every 2.5 s with every signal in it.

def parse_value(raw) -> float:
    text = str(raw)
    if text in ("True", "true", "1", "1.0"):                                  # on
        return 1.0
    if text in ("False", "false", "0", "0.0"):                                # off
        return 0.0
    try:
        return float(text)                                                    # a number (e.g. pressure)
    except ValueError:
        return np.nan                                                         # text we cannot use


def learn_healthy_values(healthy: list[Path], fault_runs: list[dict],
                         signals: list[str]) -> dict[str, float]:
    seen = {name: Counter() for name in signals}                              # value counts per signal

    def count(log: pd.DataFrame, names: list[str]) -> None:
        for name, group in log[log["node"].isin(names)].groupby("node"):      # one signal at a time
            for raw in group["value"]:
                value = parse_value(raw)
                if not np.isnan(value):
                    seen[name][value] += 1                                    # tally its values

    for path in healthy:                                                      # normal operation first
        count(read_change_log(path), signals)

    # Some signals (hydraulic pressure) never change when healthy, so they are
    # never logged. Learn those from the part of each fault run BEFORE the fault.
    missing = [name for name in signals if not seen[name]]
    for run in fault_runs if missing else []:
        log = read_change_log(run["path"])
        count(log[log["time_s"] < run["fault_started_at"]], missing)

    return {name: (seen[name].most_common(1)[0][0] if seen[name] else 0.0)   # most common value = healthy
            for name in signals}


def to_table(log: pd.DataFrame, signals: list[str], healthy: dict) -> pd.DataFrame:
    grid = np.arange(0.0, float(log["time_s"].max()) + SECONDS_PER_ROW, SECONDS_PER_ROW)  # one row every 2.5 s, whole recording
    log = log[log["node"].isin(signals)].copy()                               # only this subsystem's signals
    log["parsed"] = log["value"].map(parse_value)                             # text -> numbers
    table = pd.DataFrame(index=grid, columns=signals, dtype=float)
    for name in signals:
        changes = log[log["node"] == name]
        if changes.empty:                                                     # never changed in this recording
            table[name] = healthy[name]                                       # -> it stayed healthy
            continue
        series = pd.Series(changes["parsed"].to_numpy(), index=changes["time_s"].to_numpy())
        series = series[~series.index.duplicated(keep="last")]               # two changes at one instant: keep the last
        placed = series.reindex(series.index.union(grid)).ffill().reindex(grid)  # carry each value forward
        table[name] = placed.fillna(healthy[name]).to_numpy()                 # before the first change: healthy
    return table


def learn_cut_points(tables: list[pd.DataFrame], healthy: dict, levels: int = 4) -> dict:
    pooled = pd.concat(tables, ignore_index=True)                             # all recordings together
    cuts = {}
    for name in pooled.columns:
        values = pooled[name].dropna()
        if values.nunique() <= levels:                                        # already a few states: leave it
            continue
        if (values == healthy.get(name)).mean() > 0.5:                        # sits at one value most of the time
            cuts[name] = np.array([healthy[name], np.nextafter(healthy[name], np.inf)])  # below / at / above
        else:                                                                 # varies normally
            cuts[name] = np.unique(np.quantile(values, np.linspace(0, 1, levels + 1)[1:-1]))  # equal-sized bins
    return cuts


def discretise(table: pd.DataFrame, cuts: dict) -> pd.DataFrame:
    out = table.copy()
    for name, edges in cuts.items():
        out[name] = np.searchsorted(edges, out[name].to_numpy(), side="right").astype(float)  # value -> state number
    return out


def build_tables(healthy_paths, fault_runs, signals):
    healthy = learn_healthy_values(healthy_paths, fault_runs, signals)        # what "normal" looks like
    paths = list(healthy_paths) + [run["path"] for run in fault_runs]         # healthy AND fault recordings
    raw = [to_table(read_change_log(path), signals, healthy) for path in paths]
    raw = [table for table in raw if len(table) >= MIN_ROWS]                  # drop very short recordings
    cuts = learn_cut_points(raw, healthy)                                     # cut points learned once, for all
    return [discretise(table, cuts) for table in raw]


# =============================================================================
# 5. CAUSAL DISCOVERY - PCMCI+
# =============================================================================

class CountingGTest(Gsquared):
    """The G-test, with a progress line every 500 tests (the "epochs" of this method)."""
    tests = 0

    def run_test(self, X, Y, Z=None, *args, **kwargs):
        CountingGTest.tests += 1                                              # one more test
        if CountingGTest.tests % 500 == 0:
            print(f"    {CountingGTest.tests:>6} independence tests run", flush=True)
        return super().run_test(X, Y, Z, *args, **kwargs)


def screen_same_moment_links(test, signals: list[str]) -> set[frozenset]:
    """Keep, per signal, only its 5 strongest same-moment partners (Sparse Candidate screening)."""
    related = defaultdict(list)
    for i in range(len(signals)):
        for j in range(i + 1, len(signals)):
            value, p_value = test.run_test(X=[(i, 0)], Y=[(j, 0)], Z=[])[:2]  # plain dependence, same step
            if p_value <= SIGNIFICANCE:                                       # only real dependence counts
                related[i].append((abs(value), j))
                related[j].append((abs(value), i))
    kept = set()
    for i, found in related.items():
        for _, j in sorted(found, reverse=True)[:SAME_MOMENT_PARTNERS]:      # strongest first
            kept.add(frozenset((signals[i], signals[j])))
    return kept


def allowed_links(signals: list[str], alarms: list[str], same_moment: set[frozenset]) -> dict:
    """Tell PCMCI+ which links it may test. Alarms are effects only; same-step links are screened."""
    alarm_set = set(alarms)
    allowed = {}
    for j, effect in enumerate(signals):
        links = {}
        for i, cause in enumerate(signals):
            for lag in range(0, MAX_LAG + 1):
                if i == j and lag == 0:                                       # nothing causes itself instantly
                    continue
                if lag > 0:                                                   # links across time steps
                    if cause not in alarm_set or i == j:                      # never out of an alarm
                        links[(i, -lag)] = "-?>"                              # may exist, points forward in time
                    continue
                if frozenset((cause, effect)) not in same_moment:             # screened out
                    continue
                if cause in alarm_set and effect in alarm_set:                # alarm <-> alarm: never
                    continue
                if effect in alarm_set:
                    links[(i, 0)] = "-?>"                                     # signal -> alarm only
                elif cause in alarm_set:
                    links[(i, 0)] = "<?-"                                     # the same link, seen from the signal
                else:
                    links[(i, 0)] = "o?o"                                     # direction left to the data
        allowed[j] = links
    return allowed


def discover(tables: list[pd.DataFrame], signals: list[str], alarms: list[str]) -> set[tuple[str, str]]:
    frame = pp.DataFrame({i: t[signals].to_numpy(dtype=float) for i, t in enumerate(tables)},
                         analysis_mode="multiple", var_names=signals)         # recordings kept separate
    pcmci = PCMCI(dataframe=frame, cond_ind_test=CountingGTest(), verbosity=0)
    same_moment = screen_same_moment_links(pcmci.cond_ind_test, signals)     # step 1: screening
    print(f"    screening kept {len(same_moment)} same-moment pairs", flush=True)
    result = pcmci.run_pcmciplus(                                             # step 2: PCMCI+ itself
        tau_min=0, tau_max=MAX_LAG, pc_alpha=SIGNIFICANCE,
        max_conds_dim=MAX_CONDITIONS, max_combinations=MAX_COMBINATIONS,
        max_conds_py=CAUSES_CONDITIONED, max_conds_px=CAUSES_CONDITIONED,
        fdr_method="fdr_bh",                                                  # correct for many tests
        link_assumptions=allowed_links(signals, alarms, same_moment))
    graph = result["graph"]                                                   # link marks, [cause, effect, lag]
    links = set()
    for i, cause in enumerate(signals):
        for j, effect in enumerate(signals):
            if i != j and any(graph[i, j, lag] == "-->" for lag in range(MAX_LAG + 1)):  # only oriented links
                links.add((cause, effect))
    return links


# =============================================================================
# 6. EVALUATE - against the engineers' graph
# =============================================================================

def score(found: set, expert: set) -> dict:
    correct = found & expert                                                  # links we got right
    precision = len(correct) / len(found) if found else 0.0                  # of ours, how many are real
    recall = len(correct) / len(expert) if expert else 0.0                   # of the real ones, how many we found
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"found": len(found), "expert": len(expert), "correct": len(correct),
            "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
            "shd": len(found - expert) + len(expert - found)}                 # edits to reach the expert graph


def main() -> int:
    started = time.time()
    download_dataset()

    print(f"\n[1] Loading the {SUBSYSTEM} subsystem")
    signals, alarms, expert = load_expert_graph(SUBSYSTEM)
    fault_runs = load_fault_runs(SUBSYSTEM)
    healthy = load_healthy_recordings(HEALTHY_RECORDINGS)
    print(f"    {len(signals)} signals ({len(alarms)} alarms), {len(expert)} expert links, "
          f"{len(fault_runs)} fault runs, {len(healthy)} healthy recordings")

    print("\n[2] Preprocessing: change logs -> tables")
    tables = build_tables(healthy, fault_runs, signals)
    print(f"    {len(tables)} tables, {sum(len(t) for t in tables):,} rows, one every {SECONDS_PER_ROW} s")

    print("\n[3] Causal discovery started (PCMCI+)")
    t = time.time()
    found = discover(tables, signals, alarms)
    print(f"    finished: {len(found)} causal links, {CountingGTest.tests} tests, "
          f"{time.time() - t:.1f} s")

    print("\n[4] Evaluation against the engineers' graph")
    result = score(found, expert)
    undirected = score({tuple(sorted(e)) for e in found}, {tuple(sorted(e)) for e in expert})
    print(f"    precision {result['precision']}  recall {result['recall']}  "
          f"F1 {result['f1']}  SHD {result['shd']}")
    print(f"    ignoring direction: F1 {undirected['f1']}")
    print(f"    published best graph F1 on {SUBSYSTEM}: {PUBLISHED_BEST[SUBSYSTEM]}")

    verdicts = ([{"cause": c, "effect": e, "verdict": "found"} for c, e in sorted(expert & found)]
                + [{"cause": c, "effect": e, "verdict": "missed"} for c, e in sorted(expert - found)]
                + [{"cause": c, "effect": e, "verdict": "false alarm"} for c, e in sorted(found - expert)])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(verdicts).to_csv(OUTPUT / f"{SUBSYSTEM}_edges.csv", index=False)  # link-by-link verdict
    (OUTPUT / f"{SUBSYSTEM}_scores.json").write_text(json.dumps(
        {"directed": result, "undirected": undirected}, indent=2))           # the scores
    print(f"\n    link-by-link verdicts and scores written to {OUTPUT.relative_to(PROJECT_ROOT)}/")
    print(f"    total time {time.time() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
