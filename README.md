# Causal AI in Predictive Maintenance — Dataset Literature Review

Evidence-based evaluation of three candidate datasets for a thesis on causal AI in predictive
maintenance, structured around the intended analysis workflow:

> Causal Discovery → Root Cause Analysis → Interventions → Counterfactual XAI → Actionable Recommendations

## Datasets under review

| Dataset | Asset | Source |
|---|---|---|
| causRCA | CNC vertical lathe (real + digital twin) | [Zenodo 15876410](https://doi.org/10.5281/zenodo.15876410) |
| C-MAPSS | Turbofan engines (NASA simulation) | [NASA PCoE data set #6](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) |
| SCANIA Component X | Truck fleet component (real, anonymised) | [Research Data Sweden 2024-34](https://researchdata.se/en/catalogue/dataset/2024-34) |

## Repository layout

```
├── data/                      # raw data (gitignored; fetched via scripts/download_data.sh)
│   ├── causrca/raw/
│   ├── cmapss/raw/
│   └── scania/raw/
├── notebooks/
│   ├── 01_causrca_eda.ipynb   # general + workflow-focused EDA, causRCA
│   ├── 02_cmapss_eda.ipynb    # general + workflow-focused EDA, C-MAPSS
│   └── 03_scania_eda.ipynb    # general + workflow-focused EDA, SCANIA
├── reports/
│   └── dataset_comparison.md  # evidence-based comparison table and recommendation
├── scripts/
│   └── download_data.sh       # reproducible dataset download
└── src/causalpdm/             # shared loaders and path configuration
```

## Getting started

```bash
# 1. Install dependencies (Python >= 3.10)
pip install -r requirements.txt

# 2. Download the datasets (~1.8 GB total; SCANIA is the large one)
bash scripts/download_data.sh            # or: causrca | cmapss | scania

# 3. Run the notebooks
jupyter lab notebooks/
```

Each notebook is self-contained: dataset inventory, general EDA (structure, cleanliness,
temporal characteristics), focused EDA per workflow stage, and a verdict scorecard. The
cross-dataset synthesis lives in [reports/dataset_comparison.md](reports/dataset_comparison.md).

## Reproducibility notes

- All analysis code shared across notebooks lives in `src/causalpdm/` and is imported, not
  duplicated.
- Notebooks are committed with executed outputs so the evidence is reviewable without
  downloading the data.
- Raw data is never modified; loaders read from `data/<dataset>/raw/` only.
