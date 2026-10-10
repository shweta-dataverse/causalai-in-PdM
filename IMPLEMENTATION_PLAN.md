# Implementation plan - finalising for SOTA comparison

Written 2026-10-10. Thesis submission in two weeks.
Read `HANDOFF.md` first for project state, then `references/causrca_benchmark.md`
for the published protocol we must match.

**Budget: 4 days of implementation, then freeze the code and write.**

---

## The governing constraint

Every number we publish must be produced under the same protocol as
Mehling et al., *Procedia CIRP* 139 (2026) 114-120, or the difference must be
stated explicitly. Three protocol gaps exist today:

| Gap | Theirs | Ours | Action |
|---|---|---|---|
| Discovery protocol | Per-dataset runs, edges by majority vote | All recordings pooled as realisations in one run | Implement theirs as a variant (T4), or state the difference |
| RCA setting | Both **Full** and **Sub** | **Sub** only | Add Full (T5) |
| MAP@3 definition | Cited to [26], exact formula not printed | `running_precision / min(len(true), k)` | **Must be validated (T1)** |

T1 is a gate. Until it passes, no comparison is trustworthy.

---

# T1 - Validate our MAP@3 by reproducing a published baseline  **[GATE]**

**Why first.** If our MAP@3 is computed differently from theirs, every
comparison in the thesis is invalid. The only way to find out is to implement
one of their methods and check we get their number.

**Do**
- Create `src/causalpdm/pipeline/baselines.py`.
- Implement **TimeRank**, verbatim from their definition: *"ranks variables by
  their proximity to alarm activations, prioritizing those with the most recent
  value changes before the alarm."* No causal graph involved.
- Run on coolant, **Sub** setting, all 25 fault recordings, score with our
  existing `evaluate.score_rankings(..., k=3)`.

**Acceptance criterion**
- Published TimeRank on coolant Sub = **0.85**.
- We should land within **±0.07**. 
- If we do: our MAP@3 and our evaluation protocol are validated. Proceed.
- If we do not: **stop and diagnose before anything else.** Likely causes, in
  order: (a) MAP@k denominator differs, (b) they rank only variables, not
  alarms, (c) their "proximity to alarm" uses the labelled `alarms_detected_at`
  timestamp rather than the first alarm seen in data, (d) Sub means subgraph
  variables only, which we should confirm we match.

**Deliverable** `results/baseline_validation.json` recording our TimeRank score
against the published one. This single file is what makes the thesis
comparison defensible.

---

# T2 - Graph-based candidate restriction in RCA

**Why.** Our MAP@3 is 0.45 against their best of 1.00, and the cause is now
identified. Their winning method restricts candidates to the graph ancestors of
the alarms that fired; ours scores all variables. Our signature matches that
diagnosis exactly: top-1 = 0.92 but recall@3 = 0.47, i.e. we find one cause of
two or three.

**Do**
- In `src/causalpdm/pipeline/rca.py`, add a function that, given the learned
  graph and the alarms that fired, returns the union of each alarm's ancestors.
- Add a parameter `restrict_to_alarm_ancestors: bool = True` to `analyse()`.
  When true, score only the candidate set; when false, behave exactly as now.
- **Keep the unrestricted path working** - the comparison between the two is an
  ablation row and a result in its own right.
- The existing surprise score (`score_unexplained_behaviour`) is unchanged.

**Acceptance criteria**
- Coolant MAP@3 rises materially from 0.45. Target >= 0.75.
- recall@3 rises from 0.47. Target >= 0.70.
- top-1 does **not** fall below 0.85.
- All 39 existing tests still pass.
- Add a test asserting that for a fault whose alarms have known ancestors, those
  ancestors appear in the candidate set.

**Note.** Where the learned graph is missing an edge, the true cause will not be
reachable from its alarm and will not be recovered. This is expected and is
exactly the error-propagation effect we report - do not patch around it.

---

# T3 - Run all three subsystems

**Do**
- Run coolant, hydraulics and **probe** with the final settings.
- Probe has not been run at all yet. 11 signals, 15 expert links, 34 faults.
- Write a small script `scripts/run_all_subsystems.py` that loops and writes
  one combined results table.

**Acceptance criterion**
`results/all_subsystems.csv` with one row per subsystem and columns: graph F1,
graph SHD, RCA MAP@3, RCA top-1, intervention F1, counterfactual prevention
rate. This is the main results table of the thesis.

**Expected cost.** Coolant ~35 s, probe ~1 min (smallest graph), hydraulics
~36 min. Run hydraulics last or overnight.

---

# T4 - Re-implement the published baselines in our framework

**Why.** Right now we compare our number against their printed number. A
reviewer will ask whether preprocessing, recording subsets and splits match.
Running their methods on our data removes that objection entirely and converts
a citation into a controlled experiment.

**Do** in `baselines.py`, all following `references/causrca_benchmark.md`:
- **TimeRank** - already built in T1.
- **Baro** - robust scaling to separate the period before the alarm from after,
  rank variables by deviation magnitude. No graph.
- **CausTR** - ancestors of active alarms from the graph, then TimeRank over
  them. Run it twice: once with the **expert** graph (reproduces their number)
  and once with **our learned** graph (this is their Table 7 experiment).
- **PageRank** - node importance on the causal graph. One line with networkx.

Skip RandomWalk: it scores 0.14-0.20 and adds nothing.

**Acceptance criteria**
- Expert-graph CausTR on coolant Sub should land near the published **1.00**.
- Produces `results/baseline_comparison.csv`: rows = methods (TimeRank, Baro,
  CausTR-expert, CausTR-learned, PageRank, **ours**), columns = three
  subsystems, values = MAP@3.
- Any method we cannot reproduce within ±0.10 of its published value must be
  reported with that discrepancy stated, not quietly dropped.

---

# T5 - Add the "Full" evaluation setting

**Why.** Their Tables 5 and 6 report both Full and Sub. We only do Sub, so half
their table has no counterpart in ours.

**Do**
- Allow RCA to be presented with **all 92 machine variables** while being scored
  against subgraph faults, matching their Full definition.
- Requires preprocessing to build tables over the full variable set while
  discovery and scoring stay on the subgraph.

**Acceptance criterion**
`results/full_vs_sub.csv` with MAP@3 for both settings on all three subsystems.

**If time is short, cut this one** and state in the thesis that only the Sub
setting was evaluated, with the reason.

---

# T6 - Ablation study

**Why.** Every methods paper has one. We already have the evidence scattered
across experiments; it has never been assembled.

**Do.** One run per row, all on coolant, all other settings held fixed:

| Row | Change from the full method |
|---|---|
| Full method | - |
| No FDR correction | `MULTIPLE_TEST_CORRECTION = "none"` |
| Continuous test | `INDEPENDENCE_TEST = "continuous"` (ParCorr on boolean data) |
| Untuned settings | 1.0 s grid, lag 1, alpha 0.05 |
| No candidate restriction | T2 flag off |
| Single model in SCM | honest model used for simulation too |

**Acceptance criterion** `results/ablation.csv` with columns: graph F1, RCA
MAP@3, intervention F1. Each row should show which design decision earned which
part of the headline result.

---

# T7 - Variance across repeats

**Why.** Every number is currently from a single run with no error bars.

**Do** Re-run discovery 5 times on different random subsets (e.g. 80% of
recordings), report mean +/- standard deviation for graph F1 and RCA MAP@3.

**Acceptance criterion** `results/variance.csv`. Headline numbers in the thesis
reported as mean +/- sd.

---

# T8 - Majority-voting discovery, and the full 92-signal graph  **[P0]**

**This is the enabling task, not an optional extra.**

Our current protocol pools every recording into one PCMCI run. That works well
on a 15-signal subsystem and does not scale to 92 signals - a pooled run was
killed after seven hours with no output.

The paper's protocol runs each recording **separately** and decides each edge by
majority vote across runs. They state the reason: *"Concatenation approaches
showed significantly worse performance and runtime."*

**Measured here:** one recording on the full 92-variable graph takes **3.1
seconds** (tau_max 1, 5 s grid, max_conditions 2). Across 270 available
recordings that is roughly **14 minutes sequentially**, and the runs are
independent so they parallelise trivially.

So majority voting both (a) makes the discovery comparison exactly
apples-to-apples with Table 3, and (b) is the only way the full graph is
computable at all.

**Do**
- In `discovery.py`, add `discover_by_majority_vote(tables, variables, ...)`:
  run `discover()` on each table individually, collect the directed edges, keep
  an edge when it appears in at least `vote_threshold` of the runs
  (default 0.5).
- Keep the existing pooled `discover()` unchanged - comparing the two protocols
  on a subsystem is an ablation row and a result in its own right.
- Record the per-edge vote share, so edge confidence can be reported.

**Acceptance criteria**
- Runs on the full 92-signal graph and finishes in under an hour.
- Produces F1 and SHD against the 104-link expert graph, comparable with
  Table 3's Full column (**PC 0.15 / SHD 99, FGES 0.20 / 157, PCMCI 0.20 / 264**).
- Target: **beat 0.20**. Do not expect anything near the 0.71 we reach on
  coolant - a single recording has only ~106 rows against 92 variables and
  individually finds almost nothing (measured: 3 edges, F1 0.0). Majority voting
  across hundreds of weak runs is what recovers signal, which is also why the
  published full-graph scores are low.
- Also run it on the three subsystems, so the pooled-vs-voted comparison is
  available at every scale.

**Why this matters beyond the number.** Evaluating only on hand-picked
subsystems assumes someone already knows which subsystem failed, which is most
of the diagnosis. The paper is explicit that Sub *"mimics situations where
expert knowledge is present to limit the scope."* A thesis on actionable root
cause analysis has to show the method works when nobody has narrowed it down
first.

---

# Explicitly out of scope - state these in the thesis

- **Supervised RCA (their Table 5).** Different task: ranks diagnosis labels,
  not causal variables, with cross-validation. Not what this pipeline does.
- **Additional discovery algorithms** (Bagged-PCMCI+, LPCMCI, DYNOTEARS).
  Future work.
- **Standard CF-XAI metrics** (validity, proximity, sparsity, plausibility).
  They degenerate on boolean signals: proximity and plausibility are meaningless
  when a signal can only flip, and validity is what we report as prevention
  rate. Explain this rather than computing them.

---

# Order of work

| Day | Tasks |
|---|---|
| 1 | **T1 (gate)**, then T2 |
| 2 | **T8** (majority voting + full graph), then T3 |
| 3 | T4, T5 - re-run everything, freeze results |
| 4 | T6, T7. **Freeze the code.** |
| 5-14 | Write |

T8 moves to day 2 because the full graph is a required result, not an optional
one, and because it unblocks the Full setting in T5.

**The freeze on day 4 matters more than any single improvement.** Writing
against a moving codebase is what makes theses late.

---

# Working rules for this repository

- Branch `causal-rca-pipeline`. Never merge `main` (superseded prototype).
- Advise before coding when given a plan or diagram; wait for an explicit go.
- Commit messages: short, lowercase, human. No long bodies.
- **No AI attribution anywhere** in code, comments or commit history.
- Code must read plainly: clear names, flat structure, comments explaining
  *why* not *what*.
- Settings live in `settings.py`. No command-line flags.
- Run `pytest tests/` before every commit. 39 tests currently pass.
