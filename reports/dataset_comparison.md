# Dataset Literature Review — Comparison of causRCA, C-MAPSS and SCANIA Component X

**Context.** Candidate datasets for a thesis on causal AI in predictive maintenance, evaluated
against the intended workflow:

> Causal Discovery → Root Cause Analysis → Interventions → Counterfactual XAI → Actionable Recommendations

Every cell in the tables below is backed by executed analysis in the accompanying notebooks
(`notebooks/01_causrca_eda.ipynb`, `02_cmapss_eda.ipynb`, `03_scania_eda.ipynb`) or by the
datasets' primary documentation. Nothing here is opinion-based.

## 1. The datasets

| | causRCA | C-MAPSS | SCANIA Component X |
|---|---|---|---|
| Asset | CNC vertical lathe (real machine + hardware-in-the-loop digital twin) | Turbofan engine fleet (thermodynamic simulation) | Anonymised engine component, real truck fleet |
| Publisher, year | Fraunhofer IWU, 2025 | NASA Ames PCoE, 2008 | SCANIA / Stockholm University, 2024 |
| Source | [Zenodo 15876410](https://doi.org/10.5281/zenodo.15876410) | [NASA PCoE #6](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) | [Research Data Sweden 2024-34](https://researchdata.se/en/catalogue/dataset/2024-34) |
| License | Apache-2.0 | Public (NASA data repository) | CC BY 4.0 |
| Size on disk | ~90 MB | ~45 MB | ~1.7 GB |
| Companion paper | Mehling et al. 2025, Procedia CIRP | Saxena et al. 2008, PHM08 | Kharazian et al. 2025, Scientific Data 12:493 |

## 2. The mentor's question table, answered

| Question | causRCA | C-MAPSS | SCANIA |
|---|---|---|---|
| Number of variables/sensors | 92 variables (11 numeric, 81 boolean/categorical incl. 19+ alarms) | 21 sensors + 3 settings, all continuous (15 informative in FD001) | 105 anonymised features (8 counters + 6 histogram families of 10–36 bins) |
| Time-series length | 270 recordings; normal ≈ 600 s, fault ≈ 220 s each; ~4–5 events/s → T ≈ 200–600 at 1 Hz resampling | 700+ engine trajectories; 128–362 cycles each (median 199, FD001); 160k rows total | 33,641 vehicles; **median 43 readouts/vehicle**, only 4.4% have ≥ 100; irregular gaps (CV = 1.05) |
| Ground-truth causal graph | ✔ expert-derived DAG, 92 nodes / 104 edges, mechanism comment per edge | ✘ (physics documented, graph not released) | ✘ |
| Intervention data | ✔ 100 fault runs = do-interventions on 14 known variables, 19 scenarios, timing labels (verified: all 100 runs consistent, injection lag ≤ 62 s) | ✘ purely observational | ✘ observational; repairs are condition-confounded |
| Publicly downloadable | ✔ (Zenodo, one 3.6 MB zip) | ✔ (NASA S3, 12 MB zip) | ✔ (SND direct download, no registration) |
| Easy preprocessing | Medium — event log must be resampled to a regular grid (~30 lines, implemented in `src/causalpdm/causrca.py`); mixed types need encoding | Excellent — fixed-width text, zero missing values, zero gaps | Medium-hard — 1.1 M rows, irregular sampling, per-split files; missingness low (0.3%) but scale slows iteration |
| Good documentation | Good — thorough machine-readable README, per-run JSON metadata, GitHub benchmark code; young dataset, few external resources | Excellent — 18 years of literature, hundreds of baselines | Good — Scientific Data paper + 2 PDF annexes; anonymisation limits interpretability, not documentation |
| Suitable for PCMCI | Conditional ✔ — regular grid after resampling (median inter-event gap 214 ms → 1 Hz grid is adequate), but mostly discrete variables ⇒ mixed-data CI tests; OPC UA publish-time caveat | ✔ — regular cycles, continuous variables; first-differencing makes signals stationary (ADF-verified); degradation trend = latent confounder to handle | Weak ✘ — irregular sparse snapshots per vehicle; median 27 points/vehicle is below what PCMCI needs |
| Suitable for counterfactuals | ✔ — expert graph gives SCM skeleton; 170 normal baselines + repeated fault runs allow twin-world-style validation | Partial — counterfactuals can be generated (common in XAI literature) but never validated (no simulator access) | ✘ — counterfactuals over anonymised features are neither explainable nor validatable |
| Suitable for dashboard | ✔ — human-readable variables, subsystems, mechanism-annotated edges | ✔ — physically named sensors, RUL KPI, small/static | Partial — risk/survival views yes; "what to fix" views impossible (anonymised) |

## 3. Support per workflow stage

| Workflow stage | causRCA | C-MAPSS | SCANIA |
|---|---|---|---|
| Causal discovery | **Strong** (benchmarkable against ground truth) | Runnable, not benchmarkable | Poor |
| Root cause analysis | **Strong** (cause + diagnosis + timing labels per run) | Weak (fault mode at subset level only) | Poor (component-level repair flag only) |
| Interventions | **Strong — unique among public PdM datasets** | None | None |
| Counterfactual XAI | **Good** (SCM skeleton + repeated runs) | Generate yes, validate no | Poor |
| Actionable recommendations | **Strong** (interpretable end-to-end) | Good | Limited to scheduling |

## 4. Key risks per dataset (from the focused EDA)

**causRCA**
- Event-log format requires a resampling design decision (grid frequency, forward-fill) — implemented and demonstrated in notebook 01.
- 81 of 92 variables are boolean/categorical → PCMCI needs mixed-data conditional-independence tests (e.g. `CMIsymb`, regression-based CI), not default `ParCorr`.
- Authors document that OPC UA timestamps are publish times → sub-second cross-variable ordering not guaranteed.
- Injection labels mark the start of the fault *script*; the signal-level change follows within 2–62 s (median 32 s) → treat `cause_start_at` as a window start.
- Published July 2025: no established external baselines yet (risk for comparability, opportunity for novelty).

**C-MAPSS**
- Simulated: fleet variation comes from seeded initial wear; kind to methods, less kind to claims about real-world validity.
- Degradation trend acts as a slowly-moving latent confounder for all sensors (non-stationarity confirmed by ADF tests; first differences stationary).
- 6/21 sensors constant in FD001; strong correlation blocks → dense/unstable discovered graphs without dimensionality reduction.
- No causal ground truth of any kind: discovery, RCA, interventions and counterfactuals can be *demonstrated* but never *scored*.

**SCANIA Component X**
- Anonymisation (names, value scales, times, repair frequencies all perturbed) removes interpretability — fatal for the XAI and recommendation stages.
- Median 43 readouts per vehicle with highly irregular gaps → temporal causal discovery underpowered.
- Extreme class imbalance (97.3% healthy in validation) and 1.1 M-row scale slow thesis iteration.
- Data quality itself is good (0.3% missing overall, monotone counters) — weak *for this thesis*, not weak in general.

## 5. Evidence-based recommendation

1. **causRCA as the centerpiece.** It is the only public dataset found that supports all three
   causal pillars — discovery (scored against ground truth), interventions (real do-experiments
   with verified labels), and counterfactual reasoning (SCM skeleton + repeated runs + normal
   baselines) — while remaining fully interpretable for the recommendation/dashboard stage.
2. **C-MAPSS as the methodological anchor.** Zero-friction preprocessing and the richest baseline
   ecosystem make it the right dataset to develop and sanity-check the pipeline (especially the
   PCMCI stage) and to keep the thesis comparable to the literature. Its causal claims stay
   unvalidatable, so it cannot carry the thesis alone.
3. **SCANIA as out-of-scope for the causal core.** It answers a different (and valuable)
   question — fleet-scale survival prediction — but fails the causal workflow at every stage
   where interpretability or ground truth is required. Recommend citing it in the related-work
   chapter as the justification for why real fleet data was *not* chosen, or reserving it for a
   robustness study of the purely predictive component.

## 6. Citations

- Mehling, C. W., Pieper, S., Lüke, T. (2025). causRCA: Real-World Dataset for Causal Discovery
  and Root Cause Analysis in Machinery (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.15876410
- Mehling, C. W. et al. (2025). Enabling Joint Benchmarking of Automated Root Cause Analysis and
  Causal Discovery in Manufacturing Using the causRCA Dataset. Procedia CIRP.
- Saxena, A., Goebel, K. (2008). Turbofan Engine Degradation Simulation Data Set. NASA Ames
  Prognostics Data Repository, Moffett Field, CA.
- Saxena, A., Goebel, K., Simon, D., Eklund, N. (2008). Damage Propagation Modeling for Aircraft
  Engine Run-to-Failure Simulation. PHM08.
- Kharazian, Z., Lindgren, T., Magnússon, S., Steinert, O., Andersson Reyna, O. (2025). SCANIA
  Component X dataset: a real-world multivariate time series dataset for predictive maintenance.
  Scientific Data 12:493. https://doi.org/10.1038/s41597-025-04802-6
- Runge, J. et al. (2019). Detecting and quantifying causal associations in large nonlinear time
  series datasets. Science Advances 5(11). (PCMCI)
