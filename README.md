# Toward Actionable Root Cause Analysis

**Integrating Causal Discovery and Counterfactual XAI for Predictive Maintenance**

Master's thesis code. Two parts:

1. **Dataset literature review** — an evidence-based comparison of three candidate datasets
   (causRCA, NASA C-MAPSS, SCANIA Component X) against the intended causal workflow.
2. **The framework** — a nine-module pipeline that goes from raw sensor data to actionable
   maintenance recommendations, prototyped on NASA C-MAPSS FD001, plus an interactive dashboard.

---

## Quick start

```bash
pip install -r requirements.txt
```

```bash
bash scripts/download_data.sh
```

```bash
python scripts/run_pipeline.py
```

```bash
streamlit run dashboard/app.py
```

Those four commands, in that order, take you from an empty checkout to the interactive dashboard.
Each is explained below.

---

## How to run things

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Python 3.10 or newer.

### 2. Download the datasets

```bash
bash scripts/download_data.sh
```

Fetches all three datasets (~1.8 GB total; SCANIA is the big one) into `data/`. It is safe to
re-run — files that already exist are skipped. To get just one dataset:

```bash
bash scripts/download_data.sh cmapss
```

Valid options are `causrca`, `cmapss`, `scania`, or `all`.

### 3. Run the pipeline

```bash
python scripts/run_pipeline.py
```

This runs all nine modules end to end and prints a progress line per stage:

```
[   0.0s] 1 preprocessing    20,631 rows from 100 engines
[   0.0s]                    using 11 sensors as causal variables
[   1.8s] 2 features         health score captures 69% of sensor variation
[  12.9s] 3 discovery        found 16 causal links across 10 engines
[ 267.5s] 4 validation       8 of 16 links were stable enough to keep
[ 267.5s] 5 scm fitting      fitted 11 equations, average fit R2 = 0.49
[ 267.5s] 6 root cause       engine 39 degraded at cycle 119 of 128
[ 268.4s] 7 intervention     most useful action: control s_7
[ 269.1s] 8 counterfactual   found a change that would have avoided failure
[ 269.3s] 9 dashboard        saved to reports/actionable_rca_dashboard.html
```

**There are no command-line flags.** Every setting lives in
[`pipeline_config.py`](pipeline_config.py), with a plain-English comment explaining what it does.
Edit that file and re-run.

**About the timing.** The first run takes about 4–5 minutes, almost all of it in Modules 3 and 4
(causal discovery). Results are cached to `artifacts/framework/`, so **later runs finish in about
2 seconds** unless you change a discovery setting. To force a full recompute, set
`USE_CACHE = False` in the config.

### 4. Open the interactive dashboard

```bash
streamlit run dashboard/app.py
```

Opens at <http://localhost:8501>. It loads the cached pipeline results, so it starts instantly.

The dashboard has ten pages. Two of them are **interactive**:

- **7 · Interventions** — pick a sensor, drag a slider to choose the value to hold it at, and the
  causal model is re-simulated live to show what would happen to engine health.
- **8 · Counterfactual XAI** — ask "what if this sensor had been different?" for one specific
  engine, and watch the twin-world trajectory update as you move the slider.

The other pages explain each module, show the causal graph, the fitted equations, the root-cause
ranking, every hyperparameter with its meaning and sensible range, and the final recommendations.

### 5. Or work through the notebooks

```bash
jupyter lab notebooks/
```

- `notebooks/` — the dataset literature review (one notebook per dataset)
- `notebooks/framework/` — the pipeline, stage by stage, with full explanation

---

## The nine modules

| # | Module | What it does | Validated on C-MAPSS? |
|---|---|---|---|
| 1 | Preprocessing | Clean data; produce trended and detrended versions | Yes |
| 2 | Features & health score | Rolling features; degradation score; failure event | Yes |
| 3 | Causal discovery | Learn the causal graph with PCMCI+ | Runs, but no ground truth to score against |
| 4 | Graph validation | Stability, sensitivity, cross-method, physics checks | Robustness only, not correctness |
| 5 | SCM fitting | Turn the graph into equations plus noise | Yes |
| 6 | Root cause analysis | Separate true causes from downstream symptoms | Qualitatively |
| 7 | Intervention analysis | `do()` operations, effect ranking | **No — estimates only** |
| 8 | Counterfactual XAI | Abduction → action → prediction | **No — estimates only** |
| 9 | Dashboard | Decision support | n/a |

Modules 7 and 8 cannot be validated **because of the dataset, not the code**: C-MAPSS contains no
interventional data and no ground-truth causal graph. Every module interface is dataset-agnostic,
so pointing the same pipeline at **causRCA** — which provides an expert ground-truth graph and 100
labelled real interventions — converts Modules 3, 4, 7 and 8 from demonstrations into measured
results. That is the argument of the dataset review.

---

## Repository layout

```
├── pipeline_config.py          # ALL settings live here - edit this, not the code
├── data/                       # raw datasets (gitignored, fetched by the script)
│   ├── causrca/ cmapss/ scania/
├── src/causalpdm/
│   ├── config.py               # path configuration
│   ├── causrca.py cmapss.py scania.py    # dataset loaders
│   └── framework/              # the nine pipeline modules
│       ├── preprocessing.py    # 1
│       ├── features.py         # 2
│       ├── discovery.py        # 3  (PCMCI+)
│       ├── validation.py       # 4
│       ├── scm.py              # 5
│       ├── rca.py              # 6
│       ├── intervention.py     # 7
│       ├── counterfactual.py   # 8
│       ├── dashboard.py        # 9  (static HTML report)
│       └── artifacts.py        # stage result cache
├── dashboard/                  # interactive Streamlit app
│   ├── app.py                  # the app itself
│   ├── charts.py               # Plotly charts
│   └── explain.py              # all explanatory text
├── notebooks/
│   ├── 01_causrca_eda.ipynb    # dataset review
│   ├── 02_cmapss_eda.ipynb
│   ├── 03_scania_eda.ipynb
│   └── framework/              # the pipeline, stage by stage
├── reports/
│   ├── dataset_comparison.md   # the dataset decision, with evidence
│   └── actionable_rca_dashboard.html   # static report (generated)
├── scripts/
│   ├── download_data.sh
│   └── run_pipeline.py
└── artifacts/framework/        # cached stage results (gitignored)
```

---

## Datasets

| Dataset | Asset | Source | Licence |
|---|---|---|---|
| causRCA | CNC vertical lathe (real + digital twin) | [Zenodo 15876410](https://doi.org/10.5281/zenodo.15876410) | Apache-2.0 |
| C-MAPSS | Turbofan engines (NASA simulation) | [NASA PCoE #6](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) | Public |
| SCANIA Component X | Truck fleet component (real, anonymised) | [Research Data Sweden 2024-34](https://researchdata.se/en/catalogue/dataset/2024-34) | CC BY 4.0 |

The comparison and the reasoning behind choosing causRCA as the primary dataset are in
[reports/dataset_comparison.md](reports/dataset_comparison.md).

---

## Notes on reproducibility

- All logic lives in `src/causalpdm/`. Notebooks import it rather than redefining it.
- Notebooks are committed with their outputs, so the evidence is reviewable without running
  anything or downloading 1.8 GB.
- Raw data is never modified; loaders only read from `data/<dataset>/raw/`.
- `artifacts/` and `data/` are gitignored — both are fully regenerable.
