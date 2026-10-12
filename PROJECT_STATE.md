# Project state - the complete record

Written 2026-10-12. Branch `causal-rca-pipeline`, code frozen at tag
`code-freeze` (commit `f4de3f3`); the Fault advisor and the full-machine
screening were added after it without changing any frozen number.

Master's thesis: **Toward Actionable Root Cause Analysis: Integrating Causal
Discovery and Counterfactual XAI for Predictive Maintenance.**

This file records exactly what the project does, step by step, what was
changed and why, what every number is, and what is still open - including a
full account of why the full 92-signal machine does not finish.

Every number here is in `results/final/` and is regenerated exactly by
`python scripts/reproduce_all.py` (about 25 minutes). 64 tests pass
(`pytest tests/`).

---

## The project at a glance

| | Coolant | Hydraulics | Probe |
|---|---|---|---|
| Signals / expert links / fault runs / fault scenarios | 15 / 10 / 25 / 4 | 17 / 18 / 41 / 9 | 11 / 15 / 34 / 6 |
| Causal map accuracy (graph F1) | **0.737** | **0.615** | 0.316 |
| Root cause named first (top-1) | **1.00** | **1.00** | 0.68 |
| Root cause MAP@3 | **1.00** | **1.00** | 0.69 |
| Intervention prediction (F1, vs real injected faults) | **1.00** | **1.00** | 0.17 |
| Chance our own fix would have prevented the fault | **0.94** | **0.89** | 0.76 |
| Smallest fix is exactly the injected cause | **0.92** | **0.90** | 0.18 |
| On fault types never seen in training: MAP@3 / top-1 | 0.66 / 1.00 | 0.96 / 1.00 | 0.47 / 0.35 |
| Published best graph F1 (any algorithm) | 0.29 | 0.40 | 0.37 |
| Published best root cause MAP@3 (needs the expert graph) | 1.00 | 1.00 | 0.63 |

The full 92-signal machine: **no result yet** - see "The full machine" below.

---

## The 11-step workflow

```
 1 Load the data           ->  2 Build tables          ->  3 Learn the causal graph
 4 Check the graph         ->  5 Build the causal model ->  6 Find the root cause
 7 Predict interventions   ->  8 Counterfactuals        ->  9 Smallest fix
10 Evaluate everything     -> 11 Fault advisor (dashboard for the engineer)
```

---

### Step 1 - Load the data

**What it does.** Reads the causRCA dataset: a real CNC vertical lathe
(Fraunhofer IWU), recorded over OPC UA.

- 170 healthy recordings from normal factory operation.
- 100 fault recordings from a hardware-in-the-loop digital twin, where a
  fault was injected by forcing known signals to known values. The forced
  signals are the ground truth for every root cause.
- An expert causal graph drawn by the machine's engineers: 92 signals,
  104 rows = **102 distinct links** (two links are listed twice).
- Three subsystems with their own subgraphs: coolant, hydraulics, probe.

**How.** `src/causalpdm/pipeline/data.py`. Each fault run carries its forced
signals (`caused_by`), the forced values (`forced_states`, merged across all of
a scenario's diagnoses), the expected alarms and the timing labels.

**What we found and fixed.**
- The forced *values* live inside the diagnoses, not next to the forced
  signals. They are now read per signal (before: everything was assumed
  forced to 0).
- **A dataset label error:** scenarios exp_10 and exp_15 label the two
  hydraulic temperature switches as forced to True (their healthy state).
  Every recording shows them going False, and the expert graph's own alarm
  rule requires False. Checked across all 100 runs: these are the only labels
  that contradict their recordings. Where a switch changed during the fault
  and never took its labelled value, the recording is trusted.

**Measured facts that drive later design.**
- Recordings are change logs: a row exists only when a signal changes.
- Almost everything is on/off: coolant 15 of 15, probe 11 of 11, hydraulics
  16 of 17, full machine 78 of 92 (plus 8 decimal, 3 whole-number and 3 text
  signals; after preprocessing, 11 signals have more than two states).
- Alarms never fire in healthy recordings.
- Alarm delays are bimodal: about 0 s or about 5 s (the documented rule
  "not OK for 5 seconds -> alarm").
- Warning time in the signals (cause changes -> first alarm): median 1.3 s
  coolant, 0.5 s hydraulics, 4.5 s probe; only 11 of 100 runs give 10 s or
  more. The ~1 minute between the `cause_start_at` and `alarms_detected_at`
  labels is labelling delay, not usable warning. **causRCA cannot show early
  warning.**

**Status:** done.

---

### Step 2 - Turn change logs into tables

**What it does.** Turns each change log into a regular table: one row every
2.5 seconds, one column per signal, each value carried forward until it
changes.

**How.** `src/causalpdm/pipeline/preprocess.py`.

- A signal that never changed in a recording is absent from the file. It is
  filled with its **healthy value**, learned from the healthy recordings.
- Signals never logged in any healthy recording are learned from the part of
  each fault recording *before* the fault was injected.
- Analog signals are cut into a few states, with cut points learned **once
  over all recordings**:
  - signals that sit at one value in normal operation (hydraulic pressure is
    at 10000 in 95% of rows) become below / at / above healthy;
  - signals that vary normally (spindle speeds, positions) get four
    equal-sized bins.
- Both healthy and fault recordings are used together, because alarms only
  vary in fault recordings.

**What we found and fixed.**
- Hydraulic pressure is never logged in a healthy recording, so it was filled
  with **0**, while the real machine sits at 10000. Now learned as 10000.
- Analog signals were binned per recording, so "state 3" meant a different
  pressure in every recording. Now one set of cut points for all.
- Side effect: hydraulics discovery went from ~36 minutes to seconds.

**Status:** done.

---

### Step 3 - Learn the causal graph

**What it does.** Learns which signals cause which, and with what delay,
using PCMCI+ (Runge 2020, tigramite 5.2). Recordings are passed as separate
realisations, never glued end to end.

**How.** `src/causalpdm/pipeline/discovery.py`. Settings in `settings.py`:

| Setting | Value | Why |
|---|---|---|
| Grid | 2.5 s | With lag 3 the window reaches 7.5 s, covering the 5 s alarm rule |
| Maximum lag | 3 | Tuned on coolant only |
| Significance | 0.001 | Tuned on coolant only |
| Independence test | G-test (discrete) | The signals are on/off |
| Multiple-test correction | Benjamini-Hochberg | Thousands of candidate links |
| Alarms only as effects | on | Domain knowledge: an alarm reports, it never causes |
| Same-moment partners | 5 | Screening (see below) |
| Causes conditioned per end | 3 | Keeps the final tests small |
| Recordings | 100 fault + 60 healthy | |

**What we changed, and why.**

1. **Alarms only as effects** (new). Links out of alarms are never tested.
   Graph F1: coolant 0.71 -> 0.74, hydraulics 0.43 -> 0.64, probe 0.18 -> 0.30.
   Cost: interlocks (an alarm switching a pump off or setting the error
   state) cannot be learned - 5 of 18 hydraulics links, 19 of 104 on the full
   machine. Allowing alarms to act one step later recovered none of them and
   hurt probe, so it was dropped. In a plant, interlocks are documented in
   the controller's alarm configuration and would be added as known links.
2. **Same-moment screening** (`SAME_MOMENT_PARTNERS = 5`). Each signal may
   only be linked within the same time step to its 5 strongest partners.
   Precedent: Sparse Candidate (Friedman, Nachman and Pe'er 1999), sure
   independence screening (Fan and Lv 2008). Identical graphs on all three
   subsystems.
3. **Conditioning cap** (`MAX_PARENTS_CONDITIONED = 3`, was 6). Coolant
   unchanged; hydraulics 0.643 -> 0.615 (precision 0.90 -> 1.00); probe
   0.300 -> 0.316.
4. **Voting protocols** (`discover_by_vote`), kept for comparison:
   - per recording (the causRCA authors' protocol): finds **nothing** at our
     settings on any subsystem - no alarm link can win half the votes;
   - replicate voting (each vote = one run of every scenario): precision
     1.00 but loses recall (hydraulics 0.36 vs pooled 0.64).
   Pooled stays the method.
5. **Earlier-cause screening** (`FULL_MACHINE_EARLIER_CAUSES = 10`), full
   machine only. Identical graphs on coolant and hydraulics; probe
   0.316 -> 0.250. Not used for the subsystems.

**Results.**

| | Coolant | Hydraulics | Probe |
|---|---|---|---|
| Graph F1 | 0.737 | 0.615 | 0.316 |
| Correct / found links | 7 / 9 | 8 / 8 | 3 / 4 |
| SHD | 5 | 10 | 13 |

**Status:** done for the subsystems. Full machine: open (see below).

---

### Step 4 - Check the graph against the engineers' graph

**What it does.** Scores the learned graph against the expert graph:
precision, recall, F1 and structural Hamming distance (`evaluate.py`).

**The ruler is validated** (`scripts/validate_baseline.py`,
`results/final/baseline_validation.json`):
- Scored with our code, the 16 graphs the causRCA authors published
  (`references/causrca_published/`) give back their published F1 exactly.
- Two findings: their **coolant** evaluation included two lubrication
  signals (17 signals, 11 links); and the full expert graph lists two links
  twice, which their scorer counts as found whether predicted or not.
  **Scored correctly, the best published full-graph F1 is 0.190 (PCMCI), not
  0.20.**

**Where the gain comes from** (`scripts/protocol_comparison.py`):

| Graph F1 | Coolant | Hydraulics | Probe |
|---|---|---|---|
| Published best, any algorithm | 0.29 | 0.40 | 0.37 |
| Ours, untuned (1 s grid, lag 1, alpha 0.05) | **0.56** | **0.53** | **0.42** |
| Ours, tuned on coolant | 0.74 | 0.62 | 0.32 |
| Ours, label-free (alpha chosen by predicting unseen recordings) | 0.64 | 0.52 | 0.25 |
| Their protocol (each recording votes) at our settings | 0.00 | 0.00 | 0.00 |

Even untuned, the method beats the published best on every subsystem: the
gain is the method, not tuning on the answer. Choosing settings by prediction
picks looser levels and worse graphs - a model that predicts well is not the
right causal model.

**Status:** done.

---

### Step 5 - Build the structural causal model

**What it does.** Turns the graph into one equation per signal: a logistic
model of "is this signal on", given the state of its causes a moment earlier
(`scm.py`).

**Two models per signal.** Alarms are on about 2% of the time. A model fitted
on that never switches an alarm on in simulation, even when its cause has
failed. So each signal keeps:
- an **honest** model (real probabilities) - used to measure surprise and to
  replay a recording;
- a **balanced** model (rare states weighted up) - used to decide what
  switches on in a simulated intervention.

Ablation: with a single model, intervention F1 falls to 0.00 (coolant),
0.53 (hydraulics), 0.00 (probe).

**Known limitation.** Every signal is treated as on/off, so the 3-state
hydraulic pressure is reduced to "above or below 0.5".

**Status:** done.

---

### Step 6 - Find the root cause

**What it does.** Ranks every signal by how likely it is to be the root cause
of an alarm event (`rca.py`), using no labels.

1. Find the alarms that fired in the recording.
2. **Prime suspects** = their direct causes in the *learned* graph (the idea
   of CausTR, the best published method, which uses the expert graph).
3. Rank suspects by **unexplained behaviour**: how surprising each signal's
   behaviour is given its own causes (inspired by Budhathoki et al. 2022). A
   symptom is explained by its causes; a root cause is not.
4. Everything else follows; alarms are never suspects.

**What we changed.** Before the suspect restriction, places 2-3 went to pump
states that switch off *because* of the fault. MAP@3: coolant 0.45 -> 1.00,
hydraulics 0.58 -> 1.00, probe 0.43 -> 0.69. "Direct causes" and "all
upstream" tie on coolant; direct causes is simpler.

**Compared with the published methods**, run here on the same faults with the
same scorer (`scripts/compare_root_cause.py`; the ports reproduce the
published values, except PageRank on hydraulics, 1.00 vs 0.98):

| MAP@3 | Coolant | Hydraulics | Probe |
|---|---|---|---|
| TimeRank (no graph) | 0.85 | 0.47 | 0.47 |
| Baro (no graph) | 0.88 | 0.75 | 0.63 |
| PageRank (expert graph) | 1.00 | 1.00 | 0.27 |
| CausTR (expert graph) | 1.00 | 0.74 | 0.58 |
| CausTR (our learned graph) | 1.00 | 0.74 | 0.78 |
| **Ours** (learned graph, no labels) | **1.00** | **1.00** | 0.69 |

Our learned graph is as useful for diagnosis as the experts' graph (better on
probe). Our scoring adds value on hydraulics. On probe, CausTR on our graph
beats our scoring.

**Status:** done.

---

### Step 7 - Predict interventions, checked against real injected faults

**What it does.** Forces the injected signals to their real forced values in
the model (do-operation) on a healthy recording, and compares the alarms the
model predicts with the alarms the real machine raised
(`explain.validate_intervention`). One check per fault scenario.

**What we fixed.** The check used to force every signal to 0. Coolant has a
fault that forces a signal *on*; hydraulics has pressure faults labelled
"< 6600" and "!= 10000" (resolved from what the recording shows); probe
forces half its signals on.

**Results** (`results/final/*/intervention_by_scenario.csv`): F1
**1.00** coolant (4 scenarios), **1.00** hydraulics (9), 0.17 probe (6).
On unseen fault types: 0.73 / 0.96 / 0.00.

This is the **validated** part of the what-if reasoning: predictions under
real interventions checked against what really happened.

**Status:** done.

---

### Step 8 - Counterfactuals: would the fault have been prevented?

**What it does.** For one specific fault run: had these signals stayed
healthy from this moment on, would any alarm have fired?

**How** (`scm.sample_hidden_randomness`, `explain.prevention_probability`).
1. **Abduction:** recover the run's hidden randomness. For an on/off signal a
   single observation only gives a range, and counterfactuals are not pinned
   down by the data (Tian and Pearl 2000). So we **draw it 200 times** within
   that range instead of picking the midpoint.
2. **Action:** hold the chosen signals healthy from the chosen step.
3. **Prediction:** replay every draw.

Every draw, with nothing changed, **reproduces the recording exactly** (tested).
The result is a probability, plus a per-step alarm-risk curve.

**Results.** Chance our own fix (top-3 suspects) prevents the fault:
**0.94** coolant, **0.89** hydraulics, 0.76 probe. Fixing the true causes
instead: 0.94 / 0.89 / 0.48.

**Important.** Fixing the true causes is **not an upper bound**: holding a
downstream signal healthy can also silence an alarm (probe: 0.76 vs 0.48).
Prevention must always be read together with root cause accuracy. These
counterfactuals are **consistency-checked**, not validated - no dataset can
record a counterfactual.

**Status:** done.

---

### Step 9 - The smallest fix (counterfactual explanation)

**What it does.** For each fault, finds the smallest set of signals that, had
they stayed healthy, would have prevented the fault with probability at
least 0.9 (`explain.minimal_fix`, `scripts/minimal_fixes.py`). Tries one,
then two, then three signals from the 8 top-ranked suspects. Alarms are never
levers. This is a counterfactual explanation in the sense of algorithmic
recourse (Karimi, Schölkopf and Valera 2021).

**Results** (`results/final/minimal_fixes.csv`):

| | Coolant | Hydraulics | Probe |
|---|---|---|---|
| Fix of at most 3 signals found | 92% | 90% | 35% |
| Signals needed (average) | 2.2 | 1.6 | 1.1 |
| Fix is exactly the injected cause | **92%** | **90%** | 18% |

On coolant and hydraulics, whenever a fix is found it names exactly the
signals the test rig forced - computed without labels.

**Status:** done.

---

### Step 10 - Evaluate everything

| What | Script | Result file |
|---|---|---|
| Our scoring reproduces the published numbers | `validate_baseline.py` | `baseline_validation.json` |
| Main results, every subsystem | `run_all_subsystems.py` | `all_subsystems.csv`, `<subsystem>/` |
| Fault types never seen in training (leave one scenario out) | `held_out_scenarios.py` | `held_out_scenarios.csv`, `held_out_by_scenario.csv` |
| Tuned vs untuned vs label-free, pooled vs voted | `protocol_comparison.py` | `protocol_comparison.csv` |
| Against the published root cause methods | `compare_root_cause.py` | `root_cause_comparison.csv` |
| Smallest fixes | `minimal_fixes.py` | `minimal_fixes*.csv` |
| What each design decision is worth | `ablation.py` | `ablation.csv` |
| Settings search (coolant only) | `tune_settings.py` | `tuning_results.csv` |
| Everything above, in order | `reproduce_all.py` | all of the above |

**Held out - the model never saw that fault type:**

| | Coolant | Hydraulics | Probe |
|---|---|---|---|
| Graph F1, mean ± sd over folds | 0.69 ± 0.03 | 0.61 ± 0.02 | 0.29 ± 0.08 |
| Root cause MAP@3 / top-1 | 0.66 / 1.00 | 0.96 / 1.00 | 0.47 / 0.35 |
| Intervention F1 | 0.73 | 0.96 | 0.00 |
| P(our fix prevents the fault) | 0.23 | 0.77 | 0.30 |

Hydraulics generalises. Coolant drops where an alarm only ever fired in the
held-out scenario: the model cannot learn an alarm it has never seen fire.
Deployment requirement: every alarm must appear in the training history.

**Ablation** - one decision switched off at a time (graph F1 / MAP@3 /
intervention F1 / prevention):

| Switched off | Coolant | Hydraulics | Probe |
|---|---|---|---|
| nothing | 0.74 / 1.00 / 1.00 / 0.94 | 0.62 / 1.00 / 1.00 / 0.89 | 0.32 / 0.69 / 0.17 / 0.76 |
| alarm rule | 0.71 / 0.88 / 0.92 / 0.72 | 0.39 / 1.00 / 0.86 / 0.18 | 0.11 / 0.46 / 0.00 / 0.43 |
| FDR correction | no change | no change | no change |
| discrete test | 0.46 / 0.78 / 0.90 / 0.97 | 0.30 / 0.67 / 1.00 / 0.85 | 0.40 / 0.60 / 0.17 / 0.51 |
| window from the 5 s alarm rule | 0.56 / 0.80 / 0.71 / 0.27 | 0.53 / 0.69 / 0.61 / 0.65 | 0.42 / 0.56 / 0.17 / 0.33 |
| suspect restriction | 0.74 / 0.45 / 1.00 / 0.00 | 0.62 / 0.58 / 1.00 / 0.45 | 0.32 / 0.43 / 0.17 / 0.33 |
| balanced second model | 0.74 / 1.00 / 0.00 / 0.94 | 0.62 / 1.00 / 0.53 / 0.89 | 0.32 / 0.69 / 0.00 / 0.76 |

Every decision except FDR correction earns part of the result. On probe a
higher graph F1 can come with worse diagnosis - graph F1 alone is not the
right target.

**Status:** done; frozen in `results/final/`.

---

### Step 11 - The Fault advisor (decision support for the engineer)

**What it does.** The first page of the dashboard (`dashboard/advisor.py`),
written for an engineer, not for a data scientist:

1. **What to inspect** - prime suspects in plain words, each marked
   "confirmed by the engineers' map" or "learned from data only"; and
   "don't chase" - the alarms and the signals that only reacted.
2. **What would have prevented it** - choose signals to keep healthy and from
   which second; the chance of prevention and the alarm-risk curve update
   live (200 draws, about 0.2 s). Moving the fix after the alarm drops the
   chance from 90% to 4% on coolant event exp_22 - the deadline, visible.
3. **Preventing this type of fault** - the recurring smallest fix across all
   recorded runs of the same fault.
4. **How far to trust this** - accuracy overall and on unseen fault types; a
   warning at the top where a subsystem is unreliable (probe); a check
   against what the test rig really did.

The other 11 pages explain each pipeline stage. Run with
`python3 -m streamlit run dashboard/app.py`. Checked in the browser on all
three subsystems; all pages load without errors.

**Status:** done. Optional: a short usability check with 3-5 people.

---

## The full machine (92 signals) - why it does not finish

This is the one open item. Everything above is for the three subsystems.

### Every attempt

| When | What ran | How long | Outcome |
|---|---|---|---|
| Oct 9-11 | Pooled, old code (no alarm rule, no screening, old binning), from an earlier session (`/tmp/run_full.py`) | 19.5 h | Never finished. Believed killed after 7 h; was still running. Stopped Oct 11 - it would have overwritten the new result file. |
| Oct 11 | Three runs side by side: pooled with alarm rule; replicate voting; coarse 5 s grid, lag 1 | ~2.5 h | None finished. Diagnosed (below) and stopped. |
| Oct 11 | + same-moment screening (5 partners) | ~1 h | Instrumented copy: first phase 72,000 tests in 6 min; final phase fewer than 5,000 tests in 18 min (over 0.2 s per test). Stopped. |
| Oct 11 | + conditioning cap 3, 40-minute hard stop | 40 min | Stopped at the cap after 98,357 tests; final phase down to ~5 tests/s. |
| Oct 11-12 | Same, no time cap | ~5 h | Stopped by hand. 120,000 tests at 155 min, under 2 tests/s. Log: `results/final/full_graph_scaling.log`. |
| Oct 12 | + earlier-cause screening (10 per signal) | ~35 min | Screening and first phase done in 3.6 min. Final phase reached 40,000 tests at 12.9 min, then fewer than 5,000 more. Stopped by hand. |

### Why - in plain words

The computer is not the problem. One discovery run uses **one** CPU core
(tigramite's algorithm is sequential - each test depends on what earlier
tests removed), so a faster or bigger machine changes little. The problem is
how the amount of work grows on this particular machine's data. Four measured
reasons:

1. **The whole machine switches together.** When a machining cycle starts or
   stops, dozens of signals change in the same 2.5 s step. Pooled over 22,648
   rows, **2,863 signal pairs** move together, a median of **71** per signal
   (coolant: 8). Almost everything looks related to almost everything.

2. **PCMCI+'s final phase cannot be limited the way the first can.** It tries
   to explain each link away with combinations of the other related signals.
   Tigramite deliberately lifts our combination limit there, because the
   method's correctness guarantee needs it. With 71 neighbours that is about
   2.3 million combinations. Screening (attempts 3-6) removed most of them -
   and the first phase now takes minutes - but the final phase stays slow.

3. **Each test gets exponentially more expensive with what it conditions on.**
   The discrete test counts every combination of values. Each on/off signal
   doubles the table; each 4-state signal multiplies it by four. Measured on
   this data: a test conditioning on 6 on/off signals takes 7.5 ms; on 6
   multi-state signals, 35 ms; the final phase conditions on up to about 9,
   which with multi-state signals is around seconds per test.

4. **The full machine has 11 multi-state signals** that move with the
   operating cycle and so turn up in many tests: `Hyd_Pressure`,
   `M_PowerOnDuration`, `Prog_CuttingTime`, `Prog_CycleTime`, `Prog_LineNo`,
   `Spd_ActPos_C/X/Z`, `Spd_ActSpeed_C/X/Z`. The subsystems have none of
   these (except pressure), which is why they run in seconds.

There is also a statistical limit, not only a time limit: with 22,648 rows, a
test conditioning on 15 on/off signals has ~130,000 value combinations, most
with no data at all. Unrestricted PCMCI+ on the full machine would spend most
of its time on tests that cannot mean anything. Every large-scale causal
discovery method restricts its search for this reason.

### What is in place

- The measured scaling is recorded as evidence
  (`results/final/full_graph_scaling.log`) and can be written up as a finding.
- The "Full" root cause setting of the paper (T5) was cut, because it needs
  the full-machine graph.

### Options not yet tried

| Option | Expected effect | Cost |
|---|---|---|
| Reduce the 11 multi-state signals to two states on the full machine | Tables double instead of quadruple; tests back to ~10-30 ms | A preprocessing setting for the full machine only; only pressure is a fault cause among them |
| Learn per machine module (by tag prefix), then link modules | Each module runs in seconds | Measured: 40 of 102 expert links cross the 23 prefix modules |
| Report the scaling as the result | No new run | Already documented; the comparison with the paper's full-graph 0.190 is then absent |

---

## Every correction made along the way

| Problem | Effect before | Fix |
|---|---|---|
| Intervention check forced every signal to 0 | coolant F1 0.83, hydraulics 0.64 | real forced value per signal -> 1.00 / 1.00 (with the alarm rule) |
| Inverted temperature labels in exp_10 / exp_15 | replayed a fault that did not happen | trust the recording where it contradicts the label |
| Hydraulic pressure filled with 0 | wrong healthy state; 36-minute discovery | learned 10000 from pre-fault periods |
| Analog signals binned per recording | same state meant different values | cut points learned once |
| Counterfactual headline used the true causes | 76% looked like the pipeline's result | headline is the pipeline's own fix |
| "Fixing true causes" called an upper bound | wrong statement | renamed; read with root cause accuracy |
| Counterfactual noise at the midpoint | one arbitrary choice | 200 sampled draws -> probabilities |
| Graph cache name ignored preprocessing and the test | could reuse stale graphs | cache name includes a fingerprint of the tables and every setting |
| Results of one subsystem overwrote another's | dashboard showed the wrong subsystem | one folder per subsystem |
| `settings.py` comments quoted old numbers | contradicted the README | regenerated tuning table, comments corrected |
| A 19.5-hour stale process from an earlier session | would have overwritten the full result | stopped |

---

## Where things are

| Path | What |
|---|---|
| `settings.py` | every setting, with the reason and the measured effect |
| `src/causalpdm/pipeline/` | the pipeline: `data`, `preprocess`, `discovery`, `evaluate`, `scm`, `rca`, `explain`, `baselines`, `tuning`, `runner` |
| `scripts/` | entry points; `reproduce_all.py` regenerates every table |
| `dashboard/` | `app.py`, `advisor.py` (Fault advisor), `charts.py`, `explain.py` |
| `tests/` | 64 tests |
| `results/final/` | every frozen table (tracked in git) |
| `references/` | the published protocol and the authors' released graphs and results |
| `HANDOFF.md` | detailed numbers and design decisions |
| `IMPLEMENTATION_PLAN.md` | the plan and a day-by-day status |
| `THESIS_WRITING_PLAN.md` | chapters, sources for every table, schedule |

---

## What comes next

1. **Decide the full machine** (see the options above).
2. **Write the thesis** following `THESIS_WRITING_PLAN.md` (submission around
   2026-10-24).
3. Optional: a short usability check of the Fault advisor.
