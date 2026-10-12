# Implementation plan - frozen scope

Written 2026-10-10, revised 2026-10-11 after a full review of the code, the
data and the literature. Submission around 2026-10-24.

Read `HANDOFF.md` first for project state, then `references/causrca_benchmark.md`
for the published protocol we must match.

**Budget: 4 days of fixes, 2 days of dashboard, then freeze and write.**

**This scope is frozen.** The direction, dataset and method were questioned and
confirmed on 2026-10-11. Do not reopen them. If doubt comes back, re-read the
section below.

## Status

**Day 1 done (2026-10-11):** A1-A4 and the T1 gate. All subsystems re-run;
current numbers in `HANDOFF.md` section 4.

- **T1 passed.** TimeRank reproduces the published MAP@3 exactly in all three
  Sub settings (coolant 0.853, hydraulics 0.472, probe 0.471). Full settings
  fall inside the range that arbitrary tie order produces (coolant Full spans
  0.353-0.383 from ties alone). Our AP@k formula is also identical to theirs
  line for line. `results/final/baseline_validation.json`.
- **The authors' code is public:** github.com/causalgraph/causRCA (Apache-2.0),
  including every RCA baseline and their learned graphs
  (`eval/cd/results/*.gml`). T4 is a port, not a reconstruction. Credit it.
- **Dataset label error found:** exp_10 and exp_15 label the temperature
  switches as forced True; every recording shows them going False. The only
  contradiction in 100 runs. Worth one sentence in the thesis and a note to
  the authors.
- **Pipeline's own counterfactual is 0% on coolant.** RCA puts pump-state
  consequences (`LP_Pump_On`, `HP_Pump_isOff`) in places 2-3. T2 is now the
  most important remaining task for the headline.
- **Probe does not work with coolant settings** (graph F1 0.18, intervention
  F1 0.0). Report it as a held-out failure; diagnose on day 3 only if time.
- **Hydraulics discovery now takes seconds**, not 36 minutes.

**Day 2 (2026-10-11):** the guiding rule from here on is *the best method we
can build, measured with the same ruler as the paper, and never chosen by
looking at the answer.* Method choices are made on coolant; hydraulics, probe
and full are held out.

- **Graph scoring ruler validated too.** Our scorer reproduces every
  published subsystem F1 from their own released graphs
  (`references/causrca_published`). Two findings: their coolant evaluation
  included two lubrication signals; and the full expert graph lists two links
  twice, which their scorer counts as found whether predicted or not. Scored
  correctly, their best full-graph F1 is **0.190 (PCMCI), not 0.20**.
- **New: alarms only as effects** (domain knowledge, `ALARMS_ONLY_AS_EFFECTS`).
  Graph F1 coolant 0.71 -> 0.74, hydraulics 0.43 -> 0.64, probe 0.18 -> 0.30;
  intervention F1 coolant and hydraulics both 1.00. Cost: interlock links
  (alarm -> pump off / error state) cannot be learned - 5 of 18 in
  hydraulics, 19 of 104 on the full machine. Allowing delayed interlocks
  recovered none and hurt probe, so it was dropped.
- **Voting compared (T8).** Per-recording voting (their protocol) finds
  nothing with strict settings - no alarm link can reach half the votes.
  Replicate voting (each vote = one run of every scenario) has precision 1.00
  but loses recall on held-out subsystems (hydraulics 0.36 vs pooled 0.64).
  Pooled stays the main method; voting is an ablation row.
- **Held out scenarios (B2) done.** Hydraulics generalises to unseen faults
  (intervention F1 0.96). Coolant drops because each alarm fires in only one
  or two scenarios - the model cannot learn an alarm it has never seen fire.
  Data requirement for deployment: every alarm must appear in the history.
- **Root cause analysis is now the bottleneck.** Coolant: the model prevents
  100% of faults given the right causes, 0% with our suspects. T2 first on
  day 3.
- **Why the full machine would not finish, and the fix.** On the full machine
  a cycle starting or stopping switches dozens of signals in the same step:
  2,863 signal pairs move together, a median of 71 same-moment neighbours per
  signal (coolant: 8). PCMCI+'s same-moment phase deliberately ignores the
  combination limit, so it faced ~2.3 million tests (13-45 hours per run).
  Fix: **screen, then discover** (`SAME_MOMENT_PARTNERS = 5`) - each signal
  may be linked within the same step only to its 5 strongest same-moment
  partners; lagged links are unscreened. For the thesis:
  - Precedent: Sparse Candidate algorithm (Friedman, Nachman, Pe'er, UAI
    1999); Sure Independence Screening (Fan and Lv, JRSS-B 2008); MMHC
    (Tsamardinos et al., Machine Learning 2006).
  - Evidence it does not distort: identical graphs with and without the
    screen on coolant, hydraulics and probe.
  - What it can cost, full machine: of 83 learnable expert links, 34 fall
    outside the screened pairs, but 22 of those show no same-moment
    dependence in the data at all (the full search drops them at its first
    test). Screening alone costs at most 12, and those can still be found
    with a delay. Ten partners instead of five would recover only 2 of them.
- **Full machine: still not finished.** Screening made the same-moment phase
  tractable, but the final MCI phase conditions on the causes of both ends;
  at 6 per end each test took over 0.2 s (hours). Capped at 3 per end
  (`MAX_PARENTS_CONDITIONED`, coolant unchanged, hydraulics 0.64 -> 0.62,
  probe 0.30 -> 0.32) it still slows to ~5 tests/s late in the phase, because
  the full machine's analog signals have 3-4 states. Running uncapped in the
  background (option A). If not done by the freeze, report the measured
  scaling as a finding.
- **Full machine - final outcome.** Stopped by hand after ~5 hours, still in
  the final phase: 72,000 tests in the first 6 minutes, then the rate fell
  from ~190 to under 2 tests per second as tests conditioned on up to 8
  signals, some with 3-4 states. Fewer than 125,000 tests in 5 hours, no end
  in sight. Evidence: `results/final/full_graph_scaling.log`. Reported as a
  finding: plant-wide pooled PCMCI+ on co-switching PLC signals exceeds a
  single machine's compute budget even with screening and conditioning caps.
  **T5 (Full RCA setting) is cut** - it needs the full-machine graph.

**Day 3 (2026-10-11):**

- **T2 done - root cause analysis fixed.** Suspects are the learned-graph
  causes of the alarms that fired (CausTR's idea, with our learned graph
  instead of the expert one; no labels). MAP@3 coolant 0.45 -> 1.00,
  hydraulics 0.58 -> 1.00, probe 0.43 -> 0.69. Published best Sub: 1.00 /
  0.98 / 0.63, all with the expert graph. Held out: hydraulics 0.96, coolant
  0.66 (top-1 still 1.00), probe 0.47.
- **"Upper bound" was wrong.** Fixing suspects can prevent more faults than
  fixing the true causes (probe 0.91 vs 0.59), because holding a downstream
  signal healthy also silences an alarm. Renamed everywhere; prevention is
  always read with root cause accuracy.
- **B1 done** (`scripts/protocol_comparison.py`). Untuned settings already
  beat the published best on every subsystem (0.56 / 0.53 / 0.42 vs 0.29 /
  0.40 / 0.37): the gain is the method, not tuning. Tuning helps coolant and
  hydraulics and hurts probe. Label-free choice of significance by held-out
  prediction picks looser levels and lower F1 - good prediction is not the
  right causes. Tigramite has no automatic level for the discrete test.
- **C1 done.** Hidden randomness is sampled (200 draws) instead of taking the
  midpoint; every draw replays the recording exactly. Counterfactuals are now
  probabilities, and per-step alarm risk curves are kept for the dashboard.
  P(our fix prevents the fault): 0.94 / 0.89 / 0.76.
- **C2 done** (`scripts/minimal_fixes.py`). Smallest fix (<= 3 signals,
  P >= 0.9) found for 92% / 90% / 35% of faults, and it is exactly the
  injected cause in 92% / 90% / 18% - a counterfactual explanation checked
  against ground truth.
- Known limitation spotted: the SCM treats every variable as on/off, so the
  3-state hydraulic pressure is reduced to "above or below 0.5". Say so.
**Day 4 (2026-10-12):**

- **Full machine: stopped after ~5 h** without finishing; reported as a
  measured finding (`results/final/full_graph_scaling.log`). T5 cut.
- **T4 done** (`scripts/compare_root_cause.py`). Baro, CausTR, PageRank ported
  from the authors' code; 8 of 9 reproduce their published Sub MAP@3
  exactly (PageRank hydraulics 1.00 vs 0.98: networkx vs scikit-network).
  CausTR on our learned graph matches CausTR on the expert graph (better on
  probe, 0.78 vs 0.58); ours is best or tied on coolant and hydraulics.
- **T6 done** (`scripts/ablation.py`). Every decision except FDR correction
  earns part of the result; table in `HANDOFF.md`.
- **T7 covered** by held-out folds: graph F1 sd 0.03 / 0.02 / 0.08 across
  scenarios (`results/final/held_out_by_scenario.csv`).
- **Freeze**: `scripts/reproduce_all.py` regenerates every table into
  `results/final/`; tagged `code-freeze`.
**Day 5 (2026-10-12) - Part D, Fault advisor:**

- New first dashboard page (`dashboard/advisor.py`); the 11 stage pages stay
  as the "how it works" section. Reads the frozen results; changes no number.
- Sections: (1) inspect first / don't chase, with an evidence badge per
  suspect (confirmed by the engineers' map, or learned from data only);
  (2) live what-if - choose signals to keep healthy and when, the chance of
  prevention and the alarm-risk curve update in about 0.2 s (200 draws);
  (3) the recurring smallest fix for this fault type; (4) how far to trust it,
  from the held-out results, plus a test-rig check for the defence.
- A warning at the top when a subsystem is known to be unreliable (probe).
- Plain-language names for every hydraulics and probe signal.
- Checked in the browser on all three subsystems. Moving the fix after the
  alarm drops the chance from 90% to 4% - the deadline effect, live.
- Restart Streamlit after editing dashboard modules (no file watcher).
- **A stale process from an earlier session** (`/tmp/run_full.py`, old code)
  had been running for 19.5 hours and would have overwritten
  `results/cache/pipeline_result_full.pkl` on finishing. Stopped.

---

## What the thesis is, and what it claims

### Positioning

Maintenance analytics is usually described in four levels. This thesis builds
and validates the last two.

| Level | Question | Who answers it |
|---|---|---|
| Descriptive | What happened? | Dashboards, alarm logs |
| Predictive | Will it fail, and when? | Anomaly detection, RUL models |
| **Diagnostic** | **Why - which signal is the root cause?** | **Causal discovery + root cause analysis** |
| **Prescriptive** | **What should we do, and what happens if we do it?** | **Interventions + counterfactuals** |

Most prescriptive maintenance work recommends actions by optimising over RUL
predictions, with no causal model of what the action actually does. Ours says
"doing X prevents alarms Y and Z" and checks that against real injected
interventions. That is the contribution.

The system is **event-triggered**, not watched by a person. It runs
automatically when the first anomaly or alarm appears, in about a second, and
produces a recommendation. Reacting within seconds is the PLC's job, not a
person's.

It reduces downtime in two ways:

- **Preventing stops**, when a detector gives lead time. Depends on the fault
  developing slowly.
- **Shortening stops** (lower MTTR), by naming the root alarm among many so
  repair starts on the right component. Works for every fault, including
  abrupt ones.

### What each dataset proves

**causRCA validates correctness**: is the diagnosis right, and does the
recommended action have the predicted effect. No other public PdM dataset can
check either.

**causRCA cannot show early warning.** Measured on all 100 fault recordings,
time from the cause signal first changing to the first alarm:

| | Coolant | Hydraulics | Probe |
|---|---|---|---|
| Median warning in the signals | 1.3 s | 0.5 s | 4.5 s |

Only 11 of 100 runs give 10 s or more. The roughly one-minute gap between
`cause_start_at` and `alarms_detected_at` is labelling delay, not usable
warning. The digital twin injects faults as sudden switches.

**C-MAPSS shows lead time.** The prototype on `main` already demonstrates
"act N cycles before failure and health stays below the line" on gradually
degrading engines. Use its existing results in one thesis section. No new
code. State that it is not causally validated (C-MAPSS has no ground truth).

### Three levels of claim

| Level | What we say | Backed by |
|---|---|---|
| Proven | The workflow finds the true root cause and correctly predicts what an intervention does | causRCA ground truth |
| Argued, with sources | Unplanned downtime is very expensive and wrong diagnosis is a large part of repair time, so a validated diagnosis layer targets a costly problem | Industry surveys + our results |
| Future work | Actual money saved | Needs an industrial pilot |

### Do not claim

- That the workflow saves money. Only a pilot can show that.
- Validated counterfactuals. They are consistency-checked, not validated.
- Early warning or RUL on causRCA.
- State-of-the-art RCA against methods given the expert graph (see E5).
- That tuned discovery beats the published baseline, until B1 is done.

---

## The governing constraint

Every number we publish must be produced under the same protocol as
Mehling et al., *Procedia CIRP* 139 (2026) 114-120, or the difference must be
stated explicitly.

| Gap | Theirs | Ours | Action |
|---|---|---|---|
| Discovery protocol | Per-dataset runs, edges by majority vote | All recordings pooled in one run | T8 |
| Hyperparameters | Defaults, untuned | Tuned by scoring against the expert graph we report against | **B1** |
| RCA setting | Both **Full** and **Sub** | **Sub** only | T5 |
| MAP@3 definition | Cited to [26], formula not printed | `running_precision / min(len(true), k)` | **T1 (gate)** |

---

# Part A - Correctness fixes  **[must, day 1]**

## A1 - Intervention check forces the wrong values

`runner.py` passes `0.0` as the forced value for every scenario. The dataset
records the real forced state in `diagnoses[0].variablesStates`:

- coolant: 1 of 9 states is `True` (`CLT_Level_lt_Min` - forcing it to 0 is its
  healthy value, so that check is not an intervention at all)
- hydraulics: `True`, and numeric conditions `6600`, `< 6600`, `!= 10000`
- probe: 12 of 24 states are `True`

**Do**
- Read the forced value per variable from `variablesStates`.
- For numeric conditions, pick a value that satisfies the condition and
  document the rule (e.g. `< 6600` -> 6000).
- Report the number of scenarios next to every intervention score. Coolant F1
  0.83 currently rests on 4 scenarios.

**Acceptance** `results/intervention_by_scenario.csv`, one row per scenario.

## A2 - Hydraulic pressure is filled with 0

`Hyd_Pressure` is never logged in any of the 170 healthy recordings, so
`learn_healthy_values` falls back to 0.0. Every healthy table shows pressure 0,
and every fault table shows 0 before its first logged change, while the real
value is around 10000.

`discretise` also bins each recording separately, so "bin 3" means a different
pressure in each recording.

**Do**
- Fill `Hyd_Pressure` from the documented set point (10000), or from the first
  logged value in each fault run.
- Compute bin edges once over all recordings, then apply them to every table.
- Re-run hydraulics. Expect the F1 of 0.46 to change.

## A3 - Counterfactual headline is an oracle

The 76% "faults prevented" comes from `fix_true_causes`, which feeds in the
ground-truth causes rather than the pipeline's suspects.

**Do**
- Headline number: the pipeline's own counterfactual (fixing its top-k
  suspects) - share of faults fully prevented.
- Report 76% as an upper bound: "what the model achieves when given the true
  causes".
- Fix the README table and the dashboard text to match.

## A4 - Reproducibility

- `settings.py` comments say lag 2 was "chosen" with F1 0.64; `MAX_LAG = 3` and
  the README says 0.71. Make the comments match the regenerated tuning table.
- Regenerate `results/tuning_results.csv`.
- `results/` is gitignored. Commit final tables to `results/final/`.
- Tag the freeze commit.

---

# Part B - Comparison protocol

## T1 - Validate MAP@3 by reproducing TimeRank  **[GATE]**

Unchanged. If our MAP@3 is computed differently from theirs, every comparison
is invalid.

- Implement **TimeRank** in `src/causalpdm/pipeline/baselines.py`: *"ranks
  variables by their proximity to alarm activations, prioritizing those with
  the most recent value changes before the alarm."*
- Run on coolant, Sub, all 25 faults, score with `evaluate.score_rankings`.
- Published value 0.85. Land within ±0.07, or stop and diagnose: (a) MAP@k
  denominator, (b) ranking variables vs alarms, (c) labelled
  `alarms_detected_at` vs first alarm in data, (d) Sub variable set.

**Deliverable** `results/baseline_validation.json`.

## T8 - Majority-voting discovery and the full 92-signal graph  **[P0]**

**Not implemented yet.** The commit "promote majority-voting discovery to
required" only edited this plan.

A pooled run on the full graph was killed after seven hours. One recording on
the full graph takes 3.1 s (tau_max 1, 5 s grid, max_conditions 2), so about
14 minutes for 270 recordings, trivially parallel.

**Do**
- In `discovery.py`, add `discover_by_majority_vote(tables, variables, ...)`:
  run `discover()` per table, keep an edge when it appears in at least
  `vote_threshold` of runs.
- **`vote_threshold` is fixed at 0.5 in advance**, as in the paper. Tuning it
  against the expert graph repeats the B1 problem.
- Record per-edge vote share for edge confidence.
- Keep pooled `discover()` unchanged for the ablation.

**Acceptance**
- Full graph finishes in under an hour.
- F1 and SHD against the 104-link expert graph, comparable with Table 3 Full
  (PC 0.15 / 99, FGES 0.20 / 157, PCMCI 0.20 / 264).
- Target: **beat 0.20**. Not near 0.71 - a single recording has ~106 rows
  against 92 variables and alone finds almost nothing (3 edges, F1 0.0).
- State that full-graph settings differ from coolant settings.
- Also run on all three subsystems.

## B1 - Held-out tuning and the 2x2 table  **[must]**

`tuning.py` picks grid, lag and alpha by F1 against the same expert graph we
report against. The paper used defaults. As it stands, 0.71 vs 0.29 is not a
fair comparison.

**Do**
- Tune on **coolant only**. Freeze those settings. Report hydraulics, probe and
  full as held out.
- Add one label-free row: `significance=None`, so PCMCI+ picks alpha itself.
- Produce the 2x2 table on every subsystem:

| | Default settings | Tuned settings |
|---|---|---|
| Pooled | | |
| Majority vote | | |

This is what answers "does the gain come from tuning or from pooling".

**Acceptance** `results/protocol_2x2.csv`.

## B2 - Leave-one-scenario-out

Discovery, SCM and diagnosis are currently all fitted on the fault runs being
diagnosed.

**Do**
- For each fault scenario: drop that scenario's runs, refit, diagnose the
  held-out runs. With majority voting this is nearly free - drop that
  scenario's votes.
- Refit the SCM per fold.
- Coolant's 25 runs are only 4 scenarios. Compute error bars **over
  scenarios**, not runs.

**Acceptance** `results/loso.csv`. A drop from the in-sample numbers is an
honest result, not a failure.

## T2 - Graph-based candidate restriction in RCA

MAP@3 0.45 vs top-1 0.92: we find one cause of two or three, and places 2-3
go to pump states that are consequences. This also drives the 0% pipeline
counterfactual on coolant.

Their best method (CausTR) restricts candidates to the **direct parents** of
the alarms active at diagnosis time, not all ancestors
(`CausalPrioTimeRecencyRCA` in their code). Try parents first, ancestors as
the variant.

- Add a candidate restriction parameter to `rca.analyse()`, parents by default.
- Keep the unrestricted path for the ablation.
- Targets: MAP@3 >= 0.75, recall@3 >= 0.70, top-1 not below 0.85.
- Add a test that known ancestors appear in the candidate set.
- Missing graph edges will hide true causes. That is the error-propagation
  result - do not patch around it.

## T3 - Run all three subsystems

Unchanged. `scripts/run_all_subsystems.py` -> `results/all_subsystems.csv`:
graph F1, SHD, RCA MAP@3, top-1, intervention F1, counterfactual prevention
(pipeline and oracle), with scenario counts.

## T4 - Re-implement the published baselines

Unchanged. TimeRank, Baro, CausTR (expert graph and learned graph), PageRank.
Any method off by more than ±0.10 from its published value is reported with
the discrepancy. `results/baseline_comparison.csv`.

## T5 - "Full" evaluation setting

Unchanged. **Cut first if short of time**, and say so in the thesis.

## T6 - Ablation study

As before, plus one row:

| Row | Change from the full method |
|---|---|
| Full method | - |
| No FDR correction | `MULTIPLE_TEST_CORRECTION = "none"` |
| Continuous test | ParCorr on boolean data |
| Untuned settings | 1.0 s grid, lag 1, alpha 0.05 |
| Pooled instead of voted | T8 off |
| No candidate restriction | T2 flag off |
| Single model in SCM | honest model used for simulation too |

`results/ablation.csv`.

## T7 - Variance

Changed: variance **over scenarios** (from B2), not random 80% subsets.
Headline numbers reported as mean ± sd. `results/variance.csv`.

---

# Part C - Counterfactual XAI

## C1 - Sample the hidden noise  **[must]**

`scm.recover_hidden_randomness` takes the midpoint of each noise interval. That
choice is arbitrary: counterfactuals in binary SCMs are not point-identified
(Tian and Pearl 2000).

**Do**
- Sample the noise uniformly within its interval (e.g. 200 draws per fault).
- Report the **probability** that each alarm fires, per step, and the
  probability the fault is prevented.
- State the assumption in the thesis (monotonicity, or report as a distribution
  over consistent models).
- Keep the replay test: with no intervention, every draw reproduces the
  recording.

This also gives the smooth risk curve the dashboard needs.

## C2 - Minimal-fix search (recourse)

There is currently no counterfactual *explanation*: we test fixed candidates
but never search. Follow Karimi, Schölkopf and Valera (FAccT 2021): recourse as
a minimal set of interventions.

**Do**
- Find the smallest set (size <= 3) of non-alarm variables whose do() silences
  every alarm. Brute force is seconds.
- Alarms are never allowed as levers.
- Report **validity**, **sparsity** (number of fixes, which equals L0
  proximity on boolean data), and **actionability**.
- Remove the claim that CF metrics "degenerate" on boolean data. Proximity
  becomes Hamming distance, which is still meaningful.

**Acceptance** `results/recourse.csv`, one row per fault.

## C3 - Hydraulic pressure as a continuous lever  **[optional - cut first]**

`Hyd_Pressure` is the only real-valued signal that is a fault cause (4 of 9
hydraulics scenarios). Keep it numeric as an input to its children, give it
its own equation, use tigramite's `RegressionCI` for mixed data, and expose a
real value slider in hydraulics only. About a day. Otherwise future work.

---

# Part D - Dashboard: Fault advisor  **[days 5-6]**

**Why.** The C-MAPSS dashboard on `main` is easier to read because it has one
outcome everyone understands - a health curve with a failure line - and every
control moves that curve. The current dashboard is organised by pipeline stage,
which suits an examiner, not an engineer.

**Do** add one engineer-facing page, "Fault advisor". It reads frozen results
only, so it does not reopen the freeze. Keep the existing 11 pages under
"How it works".

| Section | Question it answers | Who, when |
|---|---|---|
| Diagnose this event | Which alarm is the root, what to inspect, which alarms not to chase | Technician, after the machine stops |
| Why it happened | Risk curve: actual vs with the fix, same noise (from C1) | Technician or engineer, same visit |
| Prevent it next time | Over all runs of this fault type: which fix prevents it, and how often (from C2) | Maintenance planner, weekly or monthly |

**What carries over from C-MAPSS**

| C-MAPSS | causRCA |
|---|---|
| Health score + failure line | Alarm risk over seconds + 50% line |
| Sensor value slider | On/off fix toggles (value slider only for `Hyd_Pressure`, if C3) |
| Before/after health chart | Actual vs with-the-fix risk curve |
| Dose-response sweep | Optional "how much earlier would detection have needed to be" view |
| Which lever helps most | Each candidate fix -> alarms prevented |
| Validity / proximity / sparsity cards | Validity, fixes needed, evidence badge |
| Recommendations page | Inspect / don't chase / act / would it have worked |

**Rules**
- No pipeline vocabulary on this page. Use the expert graph's plain variable
  descriptions (`comment` column).
- Alarms are never offered as levers.
- Every recommendation shows its evidence: link confirmed by the expert graph,
  scenario type validated against real interventions. Unvalidated items say so.
- When no fix works, say "inspect or replace the component" rather than nothing.
- The time slider is not a main control.
- A **replay mode** plays a recording so the advisor fires live at the first
  alarm, for the demo and the defence.

---

# Part E - Writing items

## E1 - Introduction and motivation

- The four-level table above.
- Alarm floods as the industrial problem: IEC 62682 / ISA-18.2 separate
  root-cause alarms from consequential alarms. This is the strongest industrial
  motivation and is currently missing.
- Cost of downtime: Siemens, *True Cost of Downtime 2024* - unplanned downtime
  costs the 500 largest companies about $1.4 trillion a year (11% of revenue);
  a lost hour costs about $2.3M in a large automotive plant and about $36,000
  in consumer goods. Based on 181 interviews - cite as an industry survey.

## E2 - Deployment architecture figure

Map the system onto ISO 13374 (data acquisition -> data manipulation -> state
detection -> health assessment -> prognostics -> advisory generation). The
Fault advisor is the advisory block.

- **Offline, periodic:** historian / OPC-UA logs -> learn graph -> **expert
  reviews the graph** -> fit SCM -> versioned approved model.
- **Event-triggered:** alarm -> fetch window -> RCA + counterfactual -> work
  order in the CMMS (e.g. SAP PM, Maximo) -> technician confirms or corrects ->
  feedback.
- **Periodic report:** recurring fault families and the fixes that prevent them.

The thesis builds and validates the analytical core. Integration is described,
not built.

## E3 - Lead time on gradual degradation

One section using the existing C-MAPSS prototype results from `main`. Shows the
same causal prescriptive layer gives actionable lead time when faults develop
slowly. Not causally validated - say so.

## E4 - Potential impact

A transparent cost model, not a result:

    annual saving ≈ alarm events per year
                  × diagnosis hours saved per event
                  × downtime cost per hour

- Low / medium / high scenarios, every assumption labelled.
- Feed in measured results, e.g. top-1 accuracy as the share of visits that go
  straight to the right component.
- Phrase as "could be worth X under these assumptions", never "saves X".

## E5 - Related work

- Mehling et al. 2026 (causRCA) - the benchmark.
- **MATERO-RCA** (Tao, Huang, Xiao, arXiv 2607.29092, July 2026, preprint).
  Uses causRCA with the **expert** graph. Set F1: BARO 85.5, CIRCA 89.5,
  RCG 88.7, StableRCA 90.4, MATERO-RCA 100. Different metric from MAP@3.
  Position ours as **learned graph, end to end, with validated interventions**,
  not as competing on RCA given the expert graph.
- Budhathoki et al. 2022 - our score is mean surprise of each variable given
  its parents; theirs is Shapley-based attribution. Write "inspired by", and fix
  the `rca.py` docstring that says "follows".
- Karimi et al. 2021 (recourse), Tian and Pearl 2000 (probabilities of
  causation), Mooij et al. 2020 (JCI), Günther et al. 2023 (J-PCMCI+).
- Counterfactual RCA in industry: Trilla et al. 2024, Cummins et al. 2024,
  Weilbach et al. 2024.

## E6 - Limitations

- causRCA faults are abrupt (median warning about 1 s); lead time is shown on
  C-MAPSS, not causRCA.
- Learned graphs are imperfect and need expert review in practice.
- Mostly boolean PLC signals limit what the model can say about values.
- Interventional and observational runs are pooled without a context variable;
  PCMCI+ assumes one stationary SCM. JCI / J-PCMCI+ is the principled fix.
- Few scenarios per subsystem (coolant 4).
- Counterfactuals are consistency-checked, not validated.
- Real-plant ground truth would come from maintenance logs, not injected faults.

## E7 - Future work

- Industrial pilot measuring MTTR before and after. The only way to show money
  saved.
- Usability study (3-5 people, one task, SUS questionnaire), if not done.
- J-PCMCI+ with regime context; `Hyd_Pressure` as continuous lever if C3 is cut.
- Paper after submission (CIRP conferences, IEEE ETFA / CASE, PHM Society
  Europe); send it to the causRCA authors at Fraunhofer IWU.

---

# Explicitly out of scope - state these in the thesis

- **Supervised RCA (their Table 5).** Ranks diagnosis labels, not causal
  variables. Different task.
- **Additional discovery algorithms** (Bagged-PCMCI+, LPCMCI, DYNOTEARS).
- **RUL prediction or early warning on causRCA.** The data cannot support it.
- **Real-time control.** Fast protection is the PLC's job.
- **Building CMMS / historian integration.** Described in E2, not built.

---

# Order of work

| Day | Tasks |
|---|---|
| 1 | A1, A2, A3, A4, then **T1 (gate)** |
| 2 | **T8** (majority vote + full graph), B2, T3 |
| 3 | **B1** (held-out tuning, 2x2), C1, C2, T2 |
| 4 | T4, T6, T7, T5 if time. Regenerate everything. **Freeze and tag.** |
| 5-6 | Part D - Fault advisor page |
| 7-14 | Write: E1-E7, then results and discussion |

**If late, cut in this order:** C3, T5, usability study, T4 PageRank.
**Never cut:** A1-A4, T1, T8, B1, C1.

**The freeze on day 4 matters more than any single improvement.** Writing
against a moving codebase is what makes theses late.

---

# Working rules for this repository

- Branch `causal-rca-pipeline`. Never merge `main` (superseded prototype); read
  C-MAPSS results from it for E3 only.
- Advise before coding when given a plan or diagram; wait for an explicit go.
- Commit messages: short, lowercase, human. No long bodies.
- **No AI attribution anywhere** in code, comments or commit history.
- Code must read plainly: clear names, flat structure, comments explaining
  *why* not *what*.
- Settings live in `settings.py`. No command-line flags.
- Run `pytest tests/` before every commit. 39 tests currently pass.
