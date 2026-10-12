# Thesis writing plan

Written 2026-10-12, after the code freeze (tag `code-freeze`). Submission
around 2026-10-24. Every number below comes from `results/final/`, which
`python scripts/reproduce_all.py` regenerates exactly.

**Title:** Toward Actionable Root Cause Analysis: Integrating Causal Discovery
and Counterfactual XAI for Predictive Maintenance

---

## Before writing: confirm with the university / supervisor

- [ ] Page or word limit, and whether appendices count
- [ ] Required template (LaTeX / Word), citation style
- [ ] Whether the C-MAPSS section (lead time) is wanted in the main text or an appendix
- [ ] Whether a short usability check of the dashboard is expected
- [ ] Submission format for code (link to the tagged repository is usually enough)

---

## The story in one paragraph

Predictive maintenance tells you *that* a machine will fail, not *why* or
*what to do*. Alarm floods hide the root cause, and wrong diagnoses cost
repair time. This thesis builds the missing diagnostic and prescriptive
layer: it learns how a machine's signals cause one another from its own PLC
recordings, names the root cause of each alarm event, predicts what an
intervention would do, and finds the smallest fix that would have prevented
the fault. Unlike almost all causal XAI work, every step is checked against
ground truth, because the causRCA dataset records the real injected faults.

---

## Research questions

| RQ | Question | Answered by |
|---|---|---|
| RQ1 | Can causal discovery on PLC signals learn a graph good enough for root cause analysis, without the engineers' graph? | Graph F1 vs published; root cause MAP@3 with learned vs expert graph |
| RQ2 | Can a model learned from observational recordings predict the effect of real interventions? | Intervention F1 against injected faults |
| RQ3 | Can counterfactual explanations name an actionable fix, and is that fix the true cause? | Prevention probability; minimal fix = injected cause |
| RQ4 | Does it hold for unseen fault types and at the scale of a whole machine? | Held-out scenarios; full-machine run and scaling |

---

## Contributions (claim exactly these, in this order)

1. **Root cause analysis with a learned graph** that matches or beats the best
   published methods that need the expert graph: MAP@3 1.00 / 1.00 / 0.69
   (coolant / hydraulics / probe), held out 0.66 / 0.96 / 0.47.
2. **Interventional validation.** Predicted alarms under the real injected
   interventions: F1 1.00 / 1.00 in-sample, 0.73 / 0.96 on unseen fault types.
3. **Counterfactual explanations checked against ground truth.** The smallest
   fix that would have prevented the fault equals the injected cause in 92% /
   90% of faults (coolant / hydraulics). Counterfactuals are probabilities
   over sampled hidden randomness, not one arbitrary choice.
4. **Causal discovery for PLC data.** Alarms-as-effects rule, discrete test,
   window from the documented alarm delay. Even untuned it beats the published
   best on every subsystem (0.56 / 0.53 / 0.42 vs 0.29 / 0.40 / 0.37). The
   ablation shows what each decision earns.
5. **Plant-scale feasibility.** Naive discovery on the full 92-signal machine
   does not finish (measured); Sparse-Candidate screening makes it computable
   and leaves the subsystem graphs unchanged where that can be checked.
   *Final full-machine number: pending (see IMPLEMENTATION_PLAN.md status).*

Plus findings worth a paragraph each: the evaluation-ruler checks (scoring
reproduces all published numbers; the full expert graph lists two links twice,
so the published full-graph best is 0.190, not 0.20); one inverted dataset
label (exp_10, exp_15); prevention rate is not bounded by fixing the true
causes; a model that predicts well is not the right causal model (label-free
tuning by prediction fails).

### Never claim

- That it saves money (only a cost model with stated assumptions).
- That counterfactuals are *validated* - they are consistency-checked; the
  interventions are validated.
- Early warning on causRCA (median warning about 1 s).
- Prevention rate on its own - always next to root cause accuracy.

---

## Chapters

Page estimates assume about 70 pages of main text; scale to the real limit.

### 1 Introduction (~6 pages)
- Downtime cost (Siemens *True Cost of Downtime 2024*: ~$1.4 trillion a year
  for the 500 largest companies, 11% of revenue; ~$2.3M per hour in large
  automotive plants) - cite as an industry survey.
- Alarm floods: IEC 62682 / ISA-18.2, root-cause vs consequential alarms.
- The four levels of maintenance analytics (descriptive, predictive,
  diagnostic, prescriptive); this thesis builds the last two.
- Why causal, not correlational: acting on a symptom fixes nothing.
- RQs, contributions, thesis structure.

### 2 Background and related work (~12 pages)
- Causal discovery for time series: PC, PCMCI, PCMCI+ (Runge 2020); joint
  causal inference across contexts (Mooij et al. 2020; J-PCMCI+, Günther et
  al. 2023); Sparse Candidate (Friedman et al. 1999), sure independence
  screening (Fan and Lv 2008), MMHC (Tsamardinos et al. 2006).
- Structural causal models, interventions, counterfactuals (Pearl);
  probabilities of causation and non-identifiability for binary variables
  (Tian and Pearl 2000).
- Root cause analysis: outlier attribution (Budhathoki et al. 2022); RCA in
  microservices (RCAEval, BARO, CIRCA); manufacturing (causRCA benchmark,
  Mehling et al. 2026; MATERO-RCA preprint, 2026 - uses the expert graph).
- Counterfactual XAI and recourse (Wachter et al. 2017; Karimi, Schölkopf and
  Valera 2021); counterfactuals in PdM (Cummins et al. 2024; Trilla et al.
  2024; Weilbach et al. 2024).
- Gap: almost no causal-XAI work can check its explanations; causRCA can.

### 3 The causRCA dataset (~6 pages)
- Machine, subsystems, healthy vs fault recordings, injected faults, expert
  graph (102 distinct links; two listed twice).
- Data facts that drive design: change logs not tables; mostly boolean;
  alarms never fire when healthy; bimodal alarm delays (0 s / 5 s); measured
  warning time median ~1 s; pressure never logged when healthy.
- The inverted label in exp_10 / exp_15 and how it was handled.
- Figures: alarm-delay histogram; one recording's alarm timeline.

### 4 Method (~14 pages)
- Pipeline figure (load -> tables -> discovery -> SCM -> RCA -> interventions
  -> counterfactuals -> advisor).
- Preprocessing: grid, carry-forward, healthy fill, shared cut points.
- Discovery: PCMCI+ with the G-test; alarms only as effects; window from the
  alarm rule; same-moment screening; conditioning cap; full-machine screening
  of earlier causes.
- SCM: logistic mechanisms, two models per signal (honest vs balanced) and why.
- RCA: unexplained-behaviour score; suspects = learned causes of fired alarms.
- Interventions (do-operator) and counterfactuals (abduction by sampling the
  hidden randomness; replay property).
- Minimal fix search (recourse).
- The Fault advisor (screenshots in chapter 6 or 7).

### 5 Evaluation design (~6 pages)
- Three tiers: direct ground truth (graph, RCA), interventional validation,
  counterfactual consistency.
- Metrics and the ruler check (`results/final/baseline_validation.json`).
- Protocols: in-sample; leave one scenario out; tuned / untuned / label-free;
  pooled vs voted; ablation; baselines run on the same faults.

### 6 Results (~14 pages)

| Section | Table / figure | Source in `results/final/` |
|---|---|---|
| 6.1 Causal discovery | graph F1, precision, recall, SHD per subsystem; learned vs expert graph figure | `all_subsystems.csv`, `*/edge_comparison.csv` |
| 6.2 Where the gain comes from | tuned / untuned / label-free / their protocol | `protocol_comparison.csv` |
| 6.3 Root cause analysis | ours vs TimeRank, Baro, PageRank, CausTR (expert and learned graph) | `root_cause_comparison.csv` |
| 6.4 Interventions | per-scenario predicted vs real alarms | `*/intervention_by_scenario.csv` |
| 6.5 Counterfactuals and minimal fixes | prevention probabilities; minimal fix = injected cause | `*/scores.json`, `minimal_fixes*.csv` |
| 6.6 Unseen fault types | held-out table, variance across scenarios | `held_out_scenarios.csv`, `held_out_by_scenario.csv` |
| 6.7 Ablation | one decision off at a time | `ablation.csv` |
| 6.8 The full machine | scaling curve (tests/s over time) and the screened result | `full_graph_scaling.log` + pending |
| 6.9 The Fault advisor | screenshots: diagnosis, live what-if, deadline effect | dashboard |

### 7 Discussion (~8 pages)
- What the results mean for each RQ.
- Deployment architecture (ISO 13374 mapping: offline learning with expert
  review of the graph; event-triggered diagnosis; work order to the CMMS;
  technician feedback). Figure.
- Lead time: causRCA faults are abrupt; the C-MAPSS prototype on `main`
  shows the same prescriptive layer gives lead time on gradual degradation
  (not causally validated there).
- Potential impact: transparent cost model, low / medium / high assumptions.
- Limitations: probe; interlocks not learnable; on/off simplification of the
  analog pressure; interventional and observational data pooled without a
  context variable; few scenarios per subsystem; held-out drop where an alarm
  only fired in one scenario; full machine needs screening; prevention can
  reward fixing a downstream signal.

### 8 Conclusion and future work (~3 pages)
- Answers to RQ1-RQ4 in one paragraph each.
- Future work: industrial pilot measuring repair time (the only way to show
  savings); J-PCMCI+ with context variables; interlocks from the alarm
  configuration; analog signals as multi-state mechanisms; usability study;
  paper and contact with the causRCA authors.

### Appendices
- A: all settings (`settings.py`) with reasons.
- B: per-scenario tables.
- C: full-machine scaling log.
- D: reproducing every number (`scripts/reproduce_all.py`, tag `code-freeze`).

---

## Figures to make

1. Four levels of maintenance analytics (table-figure).
2. Pipeline architecture.
3. Alarm-delay histogram (0 s / 5 s peaks).
4. Learned vs expert graph, coolant (found / missed / false links).
5. Root cause comparison bar chart.
6. Ablation chart.
7. Full-machine scaling: tests per second against time.
8. Fault advisor screenshots (diagnosis; what-if with the deadline effect).
9. Deployment architecture mapped to ISO 13374.

---

## Schedule

| Date | Work |
|---|---|
| Oct 12 | This plan; full-machine result (running); collect figures 3, 4, 7 |
| Oct 13 | Chapter 3 (dataset) and chapter 4 (method) - closest to the code, fastest to write |
| Oct 14 | Chapter 4 finished; chapter 5 (evaluation design) |
| Oct 15 | Chapter 6 (results) - tables first, then text |
| Oct 16 | Chapter 6 finished; figures 5, 6, 8 |
| Oct 17 | Chapter 2 (related work) |
| Oct 18 | Chapter 7 (discussion, deployment, cost model, limitations) |
| Oct 19 | Chapter 1 and 8 (written last, when the story is fixed); abstract |
| Oct 20 | Full read-through; send to supervisor |
| Oct 21-22 | Supervisor feedback; fix |
| Oct 23 | Proofread, references, formatting, appendices |
| Oct 24 | Submit |

Introduction and conclusion are written last on purpose: they summarise a
story that is only certain once the results chapter exists.

---

## Writing rules

- Every number is copied from `results/final/`, never from memory or chat.
- Language for claims: *validated* for interventions; *consistency-checked*
  for counterfactuals; *could be worth, under these assumptions* for money.
- State held-out numbers next to in-sample numbers, always.
- Report what did not work (probe, label-free tuning, voting, the full
  machine's first attempts) - it is part of the contribution.
- Credit reused code: the causRCA authors' baselines (Apache-2.0).
