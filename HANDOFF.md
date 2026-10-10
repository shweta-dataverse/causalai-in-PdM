# Project handoff

Everything a new session needs to continue this work. Written 2026-10-10.

---

## 1. What this project is

Master's thesis: **"Toward Actionable Root Cause Analysis: Integrating Causal
Discovery and Counterfactual XAI for Predictive Maintenance"**

A nine-stage pipeline that learns how a machine's signals cause one another,
finds which signal caused a fault, and says what would have prevented it -
evaluated against ground truth at every stage.

**Branch: `causal-rca-pipeline`** (an orphan branch - no history from `main`).
`main` holds an earlier C-MAPSS prototype which is superseded. Do not merge them.

---

## 2. The dataset

**causRCA** - real CNC vertical lathe (Fraunhofer IWU), recorded over OPC-UA.
Zenodo 10.5281/zenodo.15876410, Apache-2.0. Already downloaded to
`data/causrca/raw/` (24 MB, gitignored).

- 170 healthy recordings, 100 fault recordings
- Faults were **injected** into a hardware-in-the-loop digital twin, so the
  true cause of every fault is recorded (`manipulatedVars` in each
  `exp_*_description.json`)
- Ships an **expert-drawn causal graph** (92 signals, 104 links)

Subsystems: coolant (15 signals / 10 links / 25 faults), hydraulics
(17 / 18 / 41), probe (11 / 15 / 34), full (92 / 104 / 100).

### Dataset facts that drove design decisions

- **Recordings are change logs, not tables.** A row exists only when a signal
  changes. A signal that never changed is absent from the file entirely.
- **Almost everything is boolean.** coolant 15/15, probe 11/11, hydraulics
  16/17, full 78/92. The only analog signals are `Hyd_Pressure` and 13 spindle
  or program values in the full graph. **This is why there are no sliders in
  the dashboard** - you cannot set a boolean to 0.37.
- **Alarms never fire in healthy recordings** (0/20 sampled). Healthy and fault
  recordings must be pooled, or alarm signals carry no information.
- **Measured alarm delays are bimodal**: 31 events fire at ~0 s, 18 at ~5 s.
  This matches the documented rule "when X is not OK for at least 5 seconds,
  raise alarm". It is the single most important fact for tuning.

---

## 3. Repository layout

```
settings.py                     every tunable value, with explanations
src/causalpdm/pipeline/
    paths.py                    where files live
    data.py                     loading causRCA
    preprocess.py               change logs -> tables, binning, healthy baseline
    discovery.py                PCMCI+
    evaluate.py                 all metrics
    tuning.py                   search settings against the expert graph
    scm.py                      equations, noise recovery, simulation
    rca.py                      root cause ranking
    explain.py                  interventions and counterfactuals
    runner.py                   runs all stages, caches results
scripts/
    download_data.sh            fetch causRCA
    explore_data.py             EDA report -> reports/eda/
    run_pipeline.py             main entry point
    tune_settings.py            hyperparameter search
dashboard/
    app.py                      11-page Streamlit dashboard
    charts.py                   Plotly charts
    explain.py                  all explanatory text
tests/                          39 tests
results/                        outputs and cache (gitignored)
```

### How to run

```bash
pip install -r requirements.txt
python scripts/explore_data.py        # EDA
python scripts/run_pipeline.py        # full pipeline (~35 s on coolant)
streamlit run dashboard/app.py        # dashboard
pytest tests/                         # 39 tests, ~35 s
```

Settings live in `settings.py`. There are no command-line flags.

---

## 4. Results obtained

| Step | Metric | Coolant | Hydraulics |
|---|---|---|---|
| Graph validation | F1 | **0.71** | 0.46 |
| | precision / recall | 0.86 / 0.60 | 0.75 / 0.33 |
| | SHD | 5 | 14 |
| Root cause | MAP@3 | 0.45 | 0.69 |
| | top-1 hit rate | **0.92** | 0.88 |
| | MRR | 0.96 | - |
| Intervention | F1 | **0.83** | 0.64 |
| | precision | 1.00 | - |
| Counterfactual | prevention rate | **0.76** | **0.02** |

Settings used: 2.5 s grid, max lag 3, significance 0.001, discrete G-test,
FDR correction, 60 healthy recordings.

### Published baselines (verified from the paper's Table 3)

Mehling et al., *Procedia CIRP* 139 (2026) 114-120, Table 3. Extracted text is
at `/tmp/.../causrca_paper.txt` in the old session; re-fetch from
https://publica-rest.fraunhofer.de/server/api/core/bitstreams/8388679a-7fc2-4042-b6dc-133b78b4d59a/content

| Algorithm | Coolant F1/SHD | Hydraulics F1/SHD | Probe F1/SHD | Full F1/SHD |
|---|---|---|---|---|
| PC | 0.29 / 14 | 0.40 / 13 | 0.37 / 10 | 0.15 / 99 |
| FCI | 0.29 / 14 | 0.40 / 13 | 0.37 / 10 | 0.15 / 99 |
| FGES | 0.13 / 30 | 0.27 / 33 | 0.25 / 17 | 0.20 / 157 |
| PCMCI | 0.26 / 14 | 0.16 / 37 | 0.29 / 26 | 0.20 / 264 |

Unsupervised RCA, MAP@3, "Sub" column (Table 6): coolant - TimeRank 0.85,
Baro 0.88, CausTR 1.00, PageRank 1.00, RandWalk 0.14.

**So: we beat their causal discovery substantially (0.71 vs 0.29 best on
coolant), and we are behind on RCA (0.45 vs 0.85-1.00 on coolant).** Both
must be stated.

A leaderboard site (sota2.com) lists "44.7 F1" for causRCA causal discovery.
**That number does not appear in the paper** - its best full-graph F1 is 0.20.
Cite Table 3, not the leaderboard.

---

## 5. Design decisions, and why

**Two models per signal in the SCM.** Alarms fire ~2% of the time. A model
fitted on that refuses to switch the alarm on even when its cause has failed -
P(alarm | pump failed) came out at 0.4849, just under the 0.5 threshold, and
intervention accuracy was 0.0. Fix: keep an `honest_model` (true probabilities,
used for measuring surprise in RCA) and a `decision_model` (class-balanced,
used for deciding whether something switches on in simulation). After the fix
the same case gives 0.9652 and intervention F1 went 0.0 -> 0.83.

**Resampling grid matters more than max lag.** The alarm rules act at 0 s and
5 s, so the search window must reach 5 s back. Widening it by making each step
longer is far cheaper than adding steps. Measured: 1 s grid / lag 2 = F1 0.50
in 7 s; 2.5 s grid / lag 3 = F1 0.71 in 6 s; 1 s grid reaching 5 s back did not
finish in 9 minutes.

**Discrete G-test, not ParCorr.** Every coolant signal is boolean. The paper
notes their PCMCI hit a false discovery rate above 0.8 on hydraulics; our
precision is 0.86 (FDR 0.14). The fixes that got us there: discrete test,
FDR correction, strict significance (0.001).

**Healthy and fault recordings pooled.** Alarms never vary in healthy data.

**Absent signals filled from a learned healthy baseline**, not hardcoded. The
system works out on its own that `*_Ok` signals sit on and alarms sit off.

---

## 6. Evaluation - and what it is honestly worth

Three tiers, not one:

| Tier | Stages | What it is |
|---|---|---|
| Direct ground-truth comparison | graph validation, RCA | Our output vs a value recorded in the dataset |
| Interventional validation | intervention analysis | Our prediction under do() vs what the real machine did under the same do() |
| Consistency check | counterfactual | Necessary condition implied by how each fault was constructed. **No recorded counterfactual exists for any dataset** |

The counterfactual stage is **not** validated against ground truth, and the
thesis must say so. The claim to make is: *"intervention predictions are
validated against ground-truth injected interventions; counterfactual
reasoning is consistency-checked against the known construction of each
fault."*

**The standard CF-XAI metrics (validity, proximity, sparsity, plausibility)
are NOT computed here** - they were in the old C-MAPSS prototype. They
degenerate on boolean data: proximity and plausibility are meaningless when a
signal can only flip, and validity is what we report as prevention rate.

---

## 7. Known open issues

**The full 92-signal graph does not run yet.** The discrete G-test builds one
NumPy array dimension per conditioned signal, and NumPy refuses arrays past 32
dimensions. Unbounded, the full graph reached 39 and crashed. A conditioning
cap (`max_parents_conditioned=6`, passed as tigramite's `max_conds_py` /
`max_conds_px`) was added in commit *"bin analog signals and cap conditioning
set size for large graphs"* and a run was started, but **it had not finished or
been verified when this handoff was written**. Check
`results/cache/pipeline_result_full.pkl`. If it still fails, lower the cap, or
drop `max_lag` to 1, or use a coarser grid.

**Root cause analysis is behind the published baselines** (MAP@3 0.45 vs
0.85-1.00 on coolant). Diagnosis: top-1 is 0.92 but recall@3 is 0.47 - we find
the primary cause reliably and under-report secondary causes. Most causRCA
faults force 2-3 signals at once. This is the clearest thing to improve.

**Four of ten coolant links are missed**, including
`LT_Level_Ok -> LT_A_700317`. That single missing link is why 6 of 25
counterfactuals fail. Signals that vary in only a handful of recordings are
hard to recover.

**Only coolant and hydraulics have been run.** Probe is untouched.

**The tuning table was lost** when `results/` was cleared. The Settings page of
the dashboard needs `results/tuning_results.csv`; re-run
`python scripts/tune_settings.py` (~25 min) to restore it.

**Step 7 compares against a healthy recording**, which is a different run from
the fault recording. Using the fault recording's own pre-fault period as the
baseline would be a stronger comparison.

---

## 8. Contributions - as they should be claimed

**C1. Validated interventions.** Intervention predictions checked against
ground-truth injected faults (F1 0.83, precision 1.00). Rare - most work cannot
do this. **Do not claim validated counterfactuals** (see section 6).

**C2. Error propagation, extended.** Graph recall 0.60 -> 76% counterfactual
prevention; recall 0.33 -> 2.4%. Same method, two subsystems.
**Caveat: the causRCA paper already studied graph quality -> RCA performance**
(their Section 5, Table 7, Figure 3). Our extension is graph quality ->
*counterfactual* outcome, which they did not do. Claim it as an extension,
not as new.

**C3. Tuned causal discovery.** 0.71 vs 0.29 best published baseline on
coolant, 2.7x their PCMCI. The paper itself states it used *"default
hyperparameters without extensive tuning"* and identifies PCMCI's weakness as
a false discovery rate above 0.8 - which is the specific failure we fixed.

---

## 9. Working preferences

- **Advise before coding.** Pasted text, diagrams or plans mean "give me your
  assessment", not "start building". Wait for an explicit go-ahead.
- **Commit messages: short, lowercase, human.** No long bodies.
- **No AI attribution anywhere** in code, comments, or commit history. This is
  a thesis and authorship matters.
- **Code must be plainly readable**: clear names, flat structure, comments that
  explain *why* rather than *what*.
- Verify claims against the data rather than asserting them.
