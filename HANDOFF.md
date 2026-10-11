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

Updated 2026-10-11 after day 3 (see `IMPLEMENTATION_PLAN.md` status).
Regenerate with `python scripts/run_all_subsystems.py`, then
`scripts/held_out_scenarios.py`, `scripts/protocol_comparison.py`,
`scripts/minimal_fixes.py`.

| Step | Metric | Coolant | Hydraulics | Probe |
|---|---|---|---|---|
| Graph validation | F1 | **0.74** | **0.62** | 0.32 |
| | correct / found | 7 / 9 | 8 / 8 | 3 / 4 |
| | SHD | 5 | 10 | 13 |
| Root cause | MAP@3 | **1.00** | **1.00** | 0.69 |
| | top-1 hit rate | **1.00** | **1.00** | 0.68 |
| Intervention | F1 | **1.00** | **1.00** | 0.17 |
| | scenarios | 4 | 9 | 6 |
| Counterfactual | P(our own fix prevents the fault) | **0.94** | **0.89** | 0.76 |
| | P(fixing the true causes prevents it) | 0.94 | 0.89 | 0.48 |
| Minimal fix | found (at most 3 signals) | 0.92 | 0.90 | 0.35 |
| | fix is exactly the injected cause | **0.92** | **0.90** | 0.18 |

"Fixing the true causes" is NOT an upper bound: holding a downstream signal
healthy can also silence an alarm (probe: 0.76 vs 0.48). Read prevention
together with root cause accuracy.

Held out, one fault scenario at a time - the model never saw that fault type:

| | Coolant | Hydraulics | Probe |
|---|---|---|---|
| Graph F1, mean ± sd over folds | 0.69 ± 0.03 | 0.61 ± 0.02 | 0.29 ± 0.08 |
| Root cause MAP@3 / top-1 | 0.66 / 1.00 | **0.96 / 1.00** | 0.47 / 0.35 |
| Intervention F1 | 0.73 | **0.96** | 0.00 |
| P(our fix prevents the fault) | 0.23 | **0.77** | 0.30 |

Where the gain comes from (`results/protocol_comparison.csv`, graph F1):

| | Coolant | Hydraulics | Probe |
|---|---|---|---|
| Paper's best, any algorithm | 0.29 | 0.40 | 0.37 |
| Ours, untuned (1 s grid, lag 1, alpha 0.05) | 0.56 | 0.53 | 0.42 |
| Ours, tuned on coolant | 0.74 | 0.62 | 0.32 |
| Ours, label-free (alpha by held-out prediction) | 0.64 | 0.52 | 0.25 |
| Their protocol (each recording votes) at our settings | 0.00 | 0.00 | 0.00 |

Even untuned, the method beats the published best on every subsystem.

Root cause analysis against the published methods, run here on the same
faults (`results/root_cause_comparison.csv`, MAP@3, Sub setting):

| Method | Coolant | Hydraulics | Probe |
|---|---|---|---|
| TimeRank (no graph) | 0.85 | 0.47 | 0.47 |
| Baro (no graph) | 0.88 | 0.75 | 0.63 |
| PageRank (expert graph) | 1.00 | 1.00 | 0.27 |
| CausTR (expert graph) | 1.00 | 0.74 | 0.58 |
| CausTR (our learned graph) | 1.00 | 0.74 | **0.78** |
| **Ours** (learned graph, no labels) | **1.00** | **1.00** | 0.69 |

The ported baselines reproduce the published values exactly, except PageRank
on hydraulics (1.00 vs 0.98, networkx vs scikit-network). Our learned graph is
as useful as the expert graph for CausTR, and better on probe.

Ablation (`results/ablation.csv`) - one decision switched off at a time:

| Switched off | Coolant | Hydraulics | Probe |
|---|---|---|---|
| | graph F1 / MAP@3 / intervention F1 / prevented | | |
| nothing (full method) | 0.74 / 1.00 / 1.00 / 0.94 | 0.62 / 1.00 / 1.00 / 0.89 | 0.32 / 0.69 / 0.17 / 0.76 |
| alarm rule | 0.71 / 0.88 / 0.92 / 0.72 | 0.39 / 1.00 / 0.86 / 0.18 | 0.11 / 0.46 / 0.00 / 0.43 |
| FDR correction | no change | no change | no change |
| discrete test (ParCorr instead) | 0.46 / 0.78 / 0.90 / 0.97 | 0.30 / 0.67 / 1.00 / 0.85 | 0.40 / 0.60 / 0.17 / 0.51 |
| window from the 5 s alarm rule (untuned) | 0.56 / 0.80 / 0.71 / 0.27 | 0.53 / 0.69 / 0.61 / 0.65 | 0.42 / 0.56 / 0.17 / 0.33 |
| suspect restriction | 0.74 / 0.45 / 1.00 / 0.00 | 0.62 / 0.58 / 1.00 / 0.45 | 0.32 / 0.43 / 0.17 / 0.33 |
| second (balanced) model | 0.74 / 1.00 / 0.00 / 0.94 | 0.62 / 1.00 / 0.53 / 0.89 | 0.32 / 0.69 / 0.00 / 0.76 |

Every decision except FDR correction earns part of the result. On probe a
less accurate-looking graph (continuous test, untuned) scores higher graph F1
but diagnoses worse - graph F1 alone is not the right target.

Settings used: 2.5 s grid, max lag 3, significance 0.001, discrete G-test,
FDR correction, alarms only as effects, 5 same-moment partners, 3 conditioned
causes per end, suspects = learned causes of the fired alarms, 60 healthy
recordings. Tuned on coolant only; hydraulics and probe use them unchanged.

What changed on day 1, and why the numbers moved:

- Intervention checks now force each variable to its labelled value instead
  of 0 for everything. Coolant 0.83 -> 0.92, hydraulics 0.64 -> 0.86.
- The dataset labels the temperature switches in exp_10 and exp_15 as forced
  True (healthy); every recording shows them going False, as the expert
  graph's alarm rule requires. The recording is trusted there. These are the
  only labels of 100 runs that contradict their recordings.
- Hydraulic pressure is no longer filled with 0 (it is never logged in a
  healthy recording); it is learned as 10000 from the pre-fault period, and
  analog signals are binned once over all recordings. Hydraulics discovery now
  takes seconds instead of ~36 minutes.
- The counterfactual headline is now what the pipeline achieves with its own
  suspects. On coolant that is 0%: root cause analysis names one true cause
  first but fills places 2-3 with pump states (`LP_Pump_On`, `HP_Pump_isOff`)
  that are consequences, so one alarm per fault is never silenced. That is
  what T2 must fix.
- Probe is weak everywhere (graph F1 0.18, intervention F1 0.0). The coolant
  settings do not transfer to it.

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

**All three subsystems now run** (`scripts/run_all_subsystems.py`, under
two minutes in total). Probe is the weakest - see section 4.

**The tuning table was regenerated** on 2026-10-11 and is kept in
`results/final/tuning_results.csv` (tracked in git). It confirms the chosen
settings. `python scripts/tune_settings.py` takes about 5 minutes.

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
