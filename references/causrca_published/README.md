# Published causRCA results, as released by the authors

Copied unchanged from github.com/causalgraph/causRCA (Apache-2.0), the code
release accompanying Mehling et al., *Procedia CIRP* 139 (2026) 114-120.

| File | Source in their repository |
|---|---|
| `*_majority.gml` | `eval/cd/results/` - the graphs each discovery method learned |
| `evaluation_results.csv` | `eval/cd/results/` - their Table 3 numbers |
| `unsupervised_results.csv` | `eval/rca/results/` - their Table 6 numbers, unrounded |

Kept here so `scripts/validate_baseline.py` can check, without network access,
that our scoring code gives back their published numbers from their own
outputs.

Two facts found while doing that, worth knowing when comparing:

- Their F1 is plain directed-edge F1 (gcastle `MetricsDAG`; an edge learned in
  both directions counts as two predicted edges). Same as ours.
- Their **coolant** evaluation included two lubrication signals (`Lubr_On`,
  `Lubr_P_Ok`) and the expert link between them - 17 signals and 11 links
  instead of coolant's 15 and 10. Their other subsystems use exactly the
  subsystem's own signals.
