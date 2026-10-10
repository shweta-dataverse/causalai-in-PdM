# causRCA published benchmark - protocol and results

Source: Mehling, C.W., Pieper, S., Lüke, T., Döbelt, J., Ihlenfeldt, S. (2025).
"Enabling Joint Benchmarking of Automated Root Cause Analysis and Causal
Discovery in Manufacturing Using the causRCA Dataset."
*Procedia CIRP* 139 (2026) 114-120. DOI 10.1016/j.procir.2025.09.010
Open access, CC BY 4.0.

Everything below is quoted or transcribed from that paper. Our results must be
produced under the same protocol if they are to be compared against it.

---

## Table 3 - Causal discovery (F1 and SHD against the expert graph)

Protocol stated by the authors: *"We implemented dataset ensemble learning with
majority voting - executing algorithms on multiple sub-datasets and determining
edge presence by majority votes. Concatenation approaches showed significantly
worse performance and runtime."*

| Algorithm | Coolant F1/SHD | Hydraulics F1/SHD | Probe F1/SHD | Full F1/SHD |
|---|---|---|---|---|
| PC | 0.29 / 14 | 0.40 / 13 | 0.37 / 10 | 0.15 / 99 |
| FCI | 0.29 / 14 | 0.40 / 13 | 0.37 / 10 | 0.15 / 99 |
| FGES | 0.13 / 30 | 0.27 / 33 | 0.25 / 17 | 0.20 / 157 |
| PCMCI | 0.26 / 14 | 0.16 / 37 | 0.29 / 26 | 0.20 / 264 |

Authors' own caveats:
- *"Our evaluation used default hyperparameters without extensive tuning, which
  likely contributed to the modest performance. Better parameter selection can
  improve causal discovery accuracy."*
- *"PCMCI on hydraulics reached an FDR above 0.8, indicating that the majority
  of discovered edges were incorrect."*
- *"FGES and PCMCI tend to overfit by adding spurious edges when applied to
  subsystem graphs, whereas PC and FCI maintain a more conservative structure."*

---

## Table 6 - Unsupervised RCA (MAP@3, ranking VARIABLES)

k = 3, because *"no benchmark fault scenario involves more than three
simultaneous anomalous variables."* Evaluated on all 100 fault datasets.

Methods marked [*] take the **expert** causal graph as input.

| Method | Coolant Full/Sub | Hydraulics Full/Sub | Probe Full/Sub | Mean Full/Sub |
|---|---|---|---|---|
| TimeRank | 0.35 / 0.85 | 0.18 / 0.47 | 0.19 / 0.47 | 0.24 / 0.60 |
| Baro | 0.00 / 0.88 | 0.26 / 0.75 | 0.00 / 0.63 | 0.09 / 0.75 |
| CausTR [*] | 0.99 / 1.00 | 0.74 / 0.74 | 0.57 / 0.58 | 0.77 / 0.78 |
| RandWalk [*] | 0.00 / 0.14 | 0.00 / 0.14 | 0.00 / 0.32 | 0.00 / 0.20 |
| PageRank [*] | 0.00 / 1.00 | 0.11 / 0.98 | 0.00 / 0.26 | 0.04 / 0.75 |

### Method definitions, verbatim

- **TimeRank** - *"ranks variables by their proximity to alarm activations,
  prioritizing those with the most recent value changes before the alarm."*
- **Baro** - *"uses robust scaling to differentiate normal (before alarm) and
  abnormal periods (after alarm) and ranks variables by deviation magnitude."*
- **CausTR** - *"uses the provided causal graph to get all variables that can
  potentially cause the active alarms and then ranks these using TimeRank."*
- **RandomWalk** - *"performs random walks on the causal graph and ranks
  variables by visitation frequency."*
- **PageRank** - *"evaluates node importance in the causal graph based on the
  number and quality of incoming edges."*

### Full vs Sub

*"Full presents all available variables and alarms to the model, leaving it to
the model to learn which inputs are relevant. Sub presents only variables and
alarms from the designated subgraph."*

**Our pipeline currently runs Sub only.**

---

## Table 5 - Supervised RCA (MAP@2, ranking DIAGNOSES)

A different task: predicts diagnosis labels, not causal variables, using
three-fold stratified cross-validation. **Out of scope for our pipeline** -
state this explicitly rather than appearing to ignore it.

| Method | Coolant Full/Sub | Hydraulics Full/Sub | Probe Full/Sub | Mean Full/Sub |
|---|---|---|---|---|
| Base | 1.0 / 1.0 | 0.93 / 0.93 | 0.54 / 0.54 | 0.82 / 0.82 |
| LogReg | 1.0 / 1.0 | 0.99 / 0.98 | 0.80 / 0.97 | 0.93 / 0.98 |
| CausReg | 1.0 / 1.0 | 0.98 / 0.98 | 0.97 / 0.97 | 0.98 / 0.98 |

---

## Section 5 / Table 7 - Graph quality vs RCA performance

The authors already study this: CausTR run on graphs learned by each of the
four CD algorithms, compared against CausTR on the expert graph. Metric AVG@3.
Figure 3 plots MAP@3 against graph F1 from 0.3 to 1.0.

*"our results also establish a clear, positive relationship between the quality
of a discovered causal graph and RCA performance"*

**Implication for our contribution:** graph quality -> RCA is already published.
Our extension is graph quality -> *counterfactual* outcome, which they do not do.

---

## Do not cite this number

A leaderboard site (sota2.com) lists "44.7 F1" for causRCA causal discovery.
**It does not appear anywhere in the paper**, whose best full-graph F1 is 0.20.
Cite Table 3.
