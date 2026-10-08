"""Module 9 — Actionable RCA dashboard & decision support.

Renders the pipeline output as a single self-contained HTML file: no CDN, no
server, no external assets, so it opens offline and can be projected or
archived alongside the thesis. Figures are embedded as base64 PNG; the causal
graph is emitted as inline SVG with hover highlighting.

The dashboard deliberately ends with a scope panel. A decision-support tool that
does not state which of its numbers are validated and which are model-based
estimates is not decision support - it is a demo.
"""

from __future__ import annotations

import base64
import datetime as _dt
import html
import io
from pathlib import Path

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = {
    "ink": "#12181f",
    "muted": "#5b6b7c",
    "accent": "#1f6feb",
    "warn": "#d97706",
    "danger": "#dc2626",
    "ok": "#059669",
    "grid": "#dde5ee",
}


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _fig_to_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def health_figure(rca_result: dict, threshold: float, featured: pd.DataFrame) -> str:
    """Health trajectory of the analysed engine against the fleet envelope."""
    engine = rca_result["engine"]
    fig, ax = plt.subplots(figsize=(9, 3.1))

    fleet = featured.groupby("cycle")["health_indicator"].quantile([0.1, 0.9]).unstack()
    ax.fill_between(
        fleet.index, fleet[0.1], fleet[0.9], color=PALETTE["accent"], alpha=0.10,
        label="fleet 10-90th percentile",
    )
    ax.plot(
        engine["cycle"], engine["health_indicator"], color=PALETTE["accent"], lw=2,
        label=f"engine {rca_result['unit']}",
    )
    ax.axhline(threshold, color=PALETTE["danger"], ls="--", lw=1.2, label="failure threshold")
    ax.axvline(
        rca_result["onset_cycle"], color=PALETTE["warn"], ls=":", lw=1.6,
        label=f"onset (cycle {rca_result['onset_cycle']})",
    )
    ax.set_xlabel("operating cycle")
    ax.set_ylabel("health loss")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False, ncol=2)
    ax.grid(alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return _fig_to_base64(fig)


def deviation_figure(rca_result: dict, top_n: int = 12) -> str:
    """Signed deviation of each variable in the pre-failure window."""
    dev = rca_result["deviations"].head(top_n).iloc[::-1]
    roots = set(rca_result["root_causes"]["variable"])
    colours = [PALETTE["danger"] if v in roots else PALETTE["muted"] for v in dev["variable"]]

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.barh(dev["variable"], dev["deviation_z"], color=colours)
    ax.axvline(0, color=PALETTE["ink"], lw=0.8)
    ax.set_xlabel("deviation from healthy baseline (sd)")
    ax.tick_params(labelsize=8)
    ax.grid(axis="x", alpha=0.25)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    return _fig_to_base64(fig)


def intervention_figure(intervention_result: dict) -> str:
    """Estimated health improvement per candidate intervention."""
    ranking = intervention_result["ranking"]
    if ranking.empty:
        return ""
    data = ranking.iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    colours = [PALETTE["ok"] if v > 0 else PALETTE["muted"] for v in data["health_improvement"]]
    ax.barh(data["variable"], data["health_improvement"], color=colours)
    ax.axvline(0, color=PALETTE["ink"], lw=0.8)
    ax.set_xlabel("estimated reduction in health loss under do()")
    ax.tick_params(labelsize=8)
    ax.grid(axis="x", alpha=0.25)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    return _fig_to_base64(fig)


def dose_response_figure(intervention_result: dict) -> str:
    curve = intervention_result.get("dose_response")
    if curve is None or curve.empty:
        return ""
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    ax.plot(curve["do_value"], curve["health_intervened"], "o-", color=PALETTE["accent"], lw=1.8)
    ax.set_xlabel(f"do({intervention_result['best_variable']} = value)")
    ax.set_ylabel("resulting health loss")
    ax.grid(alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return _fig_to_base64(fig)


def counterfactual_figure(cf_result: dict, threshold: float) -> str:
    """Factual vs counterfactual (twin-world) health trajectory."""
    if not cf_result.get("found") or not cf_result.get("best"):
        return ""
    best = cf_result["best"]["result"]
    h_fact, h_cf = best["health_factual"], best["health_counterfactual"]

    fig, ax = plt.subplots(figsize=(9, 3.0))
    ax.plot(h_fact.to_numpy(), color=PALETTE["danger"], lw=2, label="factual (engine failed)")
    ax.plot(
        h_cf.to_numpy(), color=PALETTE["ok"], lw=2,
        label=f"counterfactual: do({best['variable']} = {best['cf_value']:.2f})",
    )
    ax.axhline(threshold, color=PALETTE["ink"], ls="--", lw=1, label="failure threshold")
    ax.axvline(best["start_idx"], color=PALETTE["warn"], ls=":", lw=1.5, label="intervention point")
    ax.set_xlabel("cycle index")
    ax.set_ylabel("health loss")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=2)
    ax.grid(alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return _fig_to_base64(fig)


# ---------------------------------------------------------------------------
# Causal graph as inline SVG
# ---------------------------------------------------------------------------

def causal_graph_svg(
    edges: pd.DataFrame, rca_result: dict, width: int = 900, height: int = 470
) -> str:
    """Render the validated causal graph, highlighting root causes and symptoms."""
    graph = nx.DiGraph()
    for _, e in edges.iterrows():
        if e["source"] != e["target"]:
            graph.add_edge(e["source"], e["target"], weight=abs(float(e["abs_strength"])))
    if graph.number_of_nodes() == 0:
        return "<p class='empty'>No directed edges survived validation.</p>"

    roots = set(rca_result["root_causes"]["variable"])
    symptoms = set(rca_result["symptoms"]["variable"])
    deviations = rca_result["deviations"].set_index("variable")["abs_deviation"].to_dict()

    pos = nx.spring_layout(graph, seed=7, k=1.5 / np.sqrt(graph.number_of_nodes()))
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    pad = 46

    def sx(x: float) -> float:
        span = (max(xs) - min(xs)) or 1
        return pad + (x - min(xs)) / span * (width - 2 * pad)

    def sy(y: float) -> float:
        span = (max(ys) - min(ys)) or 1
        return pad + (y - min(ys)) / span * (height - 2 * pad)

    parts = [
        f"<svg viewBox='0 0 {width} {height}' class='cgraph' "
        f"xmlns='http://www.w3.org/2000/svg' role='img' "
        f"aria-label='Discovered causal graph'>",
        "<defs><marker id='arrow' viewBox='0 0 10 10' refX='9' refY='5' "
        "markerWidth='6' markerHeight='6' orient='auto-start-reverse'>"
        "<path d='M 0 0 L 10 5 L 0 10 z' fill='currentColor'/></marker></defs>",
    ]

    max_w = max((d["weight"] for _, _, d in graph.edges(data=True)), default=1.0)
    for u, v, d in graph.edges(data=True):
        x1, y1, x2, y2 = sx(pos[u][0]), sy(pos[u][1]), sx(pos[v][0]), sy(pos[v][1])
        dx, dy = x2 - x1, y2 - y1
        dist = (dx**2 + dy**2) ** 0.5 or 1
        # Stop the arrow at the node boundary so the head stays visible.
        x2t, y2t = x2 - dx / dist * 22, y2 - dy / dist * 22
        stroke = 0.8 + 2.6 * (d["weight"] / max_w)
        parts.append(
            f"<line class='edge' data-u='{html.escape(u)}' data-v='{html.escape(v)}' "
            f"x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2t:.1f}' y2='{y2t:.1f}' "
            f"stroke-width='{stroke:.2f}' marker-end='url(#arrow)'/>"
        )

    for node in graph.nodes:
        x, y = sx(pos[node][0]), sy(pos[node][1])
        if node in roots:
            cls, radius = "root", 20
        elif node in symptoms:
            cls, radius = "symptom", 17
        else:
            cls, radius = "plain", 14
        title = f"{node} — deviation {deviations.get(node, 0):.2f} sd"
        parts.append(
            f"<g class='node {cls}' data-name='{html.escape(node)}'><title>{html.escape(title)}</title>"
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{radius}'/>"
            f"<text x='{x:.1f}' y='{y + 3.5:.1f}' text-anchor='middle'>{html.escape(node)}</text></g>"
        )

    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

_CSS = """
:root{--bg:#f7f9fc;--card:#ffffff;--ink:#12181f;--muted:#5b6b7c;--line:#e2e8f0;
--accent:#1f6feb;--ok:#059669;--warn:#d97706;--danger:#dc2626;--chip:#eef3fa;}
@media (prefers-color-scheme:dark){:root{--bg:#0d1117;--card:#161b22;--ink:#e6edf3;
--muted:#8b98a5;--line:#2a323d;--accent:#58a6ff;--ok:#3fb950;--warn:#d29922;
--danger:#f85149;--chip:#1f2630;}}
:root[data-theme="dark"]{--bg:#0d1117;--card:#161b22;--ink:#e6edf3;--muted:#8b98a5;
--line:#2a323d;--accent:#58a6ff;--ok:#3fb950;--warn:#d29922;--danger:#f85149;--chip:#1f2630;}
:root[data-theme="light"]{--bg:#f7f9fc;--card:#ffffff;--ink:#12181f;--muted:#5b6b7c;
--line:#e2e8f0;--accent:#1f6feb;--ok:#059669;--warn:#d97706;--danger:#dc2626;--chip:#eef3fa;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
.wrap{max-width:1180px;margin:0 auto;padding:28px 20px 60px}
header.top{border-bottom:2px solid var(--line);padding-bottom:18px;margin-bottom:26px}
h1{font-size:25px;margin:0 0 6px;letter-spacing:-.02em}
.sub{color:var(--muted);font-size:13.5px;margin:0}
.meta{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px}
.chip{background:var(--chip);border:1px solid var(--line);border-radius:999px;
padding:3px 11px;font-size:11.5px;color:var(--muted);white-space:nowrap}
.pipeline{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 26px}
.stage{flex:1 1 118px;background:var(--card);border:1px solid var(--line);
border-radius:8px;padding:9px 10px;font-size:11px;position:relative}
.stage b{display:block;font-size:10px;color:var(--muted);font-weight:600;
text-transform:uppercase;letter-spacing:.05em;margin-bottom:3px}
.stage .val{font-size:12.5px;font-weight:600}
.stage.done{border-left:3px solid var(--ok)}
.stage.est{border-left:3px solid var(--warn)}
section{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:22px 24px;margin-bottom:20px}
section>h2{font-size:12px;text-transform:uppercase;letter-spacing:.07em;
color:var(--muted);margin:0 0 4px}
section>.lead{margin:0 0 18px;font-size:13.5px;color:var(--muted)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:22px}
img.fig{width:100%;height:auto;display:block}
.figwrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)}
th{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
td.num{text-align:right;font-variant-numeric:tabular-nums}
tr.rootrow td:first-child{font-weight:700;color:var(--danger)}
.tag{display:inline-block;font-size:10.5px;padding:1px 8px;border-radius:999px;
border:1px solid currentColor;font-weight:600}
.tag.cause{color:var(--danger)}.tag.sym{color:var(--muted)}
.tablewrap{overflow-x:auto}
.cgraph{width:100%;height:auto;color:var(--muted)}
.cgraph .edge{stroke:var(--muted);opacity:.42}
.cgraph .node circle{fill:var(--chip);stroke:var(--muted);stroke-width:1.4}
.cgraph .node.root circle{fill:var(--danger);stroke:var(--danger);fill-opacity:.22;stroke-width:2.4}
.cgraph .node.symptom circle{fill:var(--warn);stroke:var(--warn);fill-opacity:.16}
.cgraph .node text{font-size:9.5px;fill:var(--ink);pointer-events:none;font-weight:500}
.cgraph .node:hover circle{stroke:var(--accent);stroke-width:3}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:11.5px;color:var(--muted);margin-top:10px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}
.cf{background:var(--chip);border-left:3px solid var(--accent);border-radius:8px;
padding:15px 18px;font-size:14px;line-height:1.6}
.metrics{display:flex;flex-wrap:wrap;gap:10px;margin-top:14px}
.metric{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:9px 14px;min-width:104px}
.metric b{display:block;font-size:10px;color:var(--muted);text-transform:uppercase;
letter-spacing:.05em;margin-bottom:2px}
.metric span{font-size:16px;font-weight:650;font-variant-numeric:tabular-nums}
ol.recs{margin:0;padding-left:20px}
ol.recs li{margin-bottom:11px}
ol.recs code{background:var(--chip);padding:1px 6px;border-radius:4px;font-size:12.5px}
.scope{border-left:3px solid var(--warn)}
.scope ul{margin:8px 0 0;padding-left:20px;font-size:13px;color:var(--muted)}
.scope li{margin-bottom:6px}
footer{color:var(--muted);font-size:11.5px;text-align:center;margin-top:34px;
padding-top:16px;border-top:1px solid var(--line)}
"""


def _table(df: pd.DataFrame, numeric_cols: set[str], root_col: str | None = None) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = []
    for _, row in df.iterrows():
        cls = " class='rootrow'" if root_col and bool(row.get(root_col)) else ""
        cells = []
        for col in df.columns:
            value = row[col]
            if col in numeric_cols and isinstance(value, (int, float, np.floating)):
                cells.append(f"<td class='num'>{value:.3f}</td>")
            elif isinstance(value, (bool, np.bool_)):
                tag = "cause" if value else "sym"
                label = "root cause" if value else "symptom"
                cells.append(f"<td><span class='tag {tag}'>{label}</span></td>")
            else:
                cells.append(f"<td>{html.escape(str(value))}</td>")
        body.append(f"<tr{cls}>{''.join(cells)}</tr>")
    return (
        "<div class='tablewrap'><table><thead><tr>"
        + head
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def _recommendations(rca_result: dict, intervention_result: dict, cf_result: dict) -> list[str]:
    """Turn the causal findings into maintenance instructions."""
    recs: list[str] = []
    roots = rca_result["root_causes"]

    if not roots.empty:
        top = roots.iloc[0]
        recs.append(
            f"<b>Inspect the subsystem measured by <code>{top['variable']}</code> first.</b> "
            f"It deviates {top['abs_deviation']:.1f} sd from this engine's healthy baseline and "
            f"causally precedes {int(top['n_symptoms_explained'])} of the other deviating signals, "
            f"so it is an upstream cause rather than a symptom."
        )
    if len(roots) > 1:
        others = ", ".join(f"<code>{v}</code>" for v in roots["variable"].iloc[1:4])
        recs.append(
            f"Secondary root-cause candidates to check in the same visit: {others}."
        )

    symptoms = rca_result["symptoms"]
    if not symptoms.empty:
        names = ", ".join(f"<code>{v}</code>" for v in symptoms["variable"].head(3))
        recs.append(
            f"<b>Do not action {names} directly.</b> These deviate strongly but sit downstream "
            f"of the identified causes in the causal graph — treating them would address the "
            f"symptom while the cause persists."
        )

    best = intervention_result.get("best_variable")
    ranking = intervention_result.get("ranking")
    if best and ranking is not None and not ranking.empty:
        gain = float(ranking.iloc[0]["health_improvement"])
        if gain > 0:
            recs.append(
                f"Holding <code>{best}</code> within its healthy range from "
                f"{intervention_result['lead_time']} cycles before onset lowers the projected "
                f"health-loss score by {gain:.3f} under the fitted model — the highest-yield "
                f"single lever among the candidates tested."
            )
        else:
            recs.append(
                f"No single-variable intervention among the tested candidates improves projected "
                f"health, which points to component replacement rather than operational "
                f"adjustment."
            )

    if cf_result.get("found") and cf_result.get("best"):
        metrics = cf_result["best"]["metrics"]
        recs.append(
            f"The counterfactual is achievable: a {metrics['proximity_sd']:.1f} sd shift is "
            f"enough to keep this engine below the failure threshold, and the required value "
            f"lies inside its observed operating range"
            + (" (plausible)." if metrics["plausible"] else " (borderline plausibility).")
        )
    else:
        recs.append(
            "No achievable single-variable counterfactual avoids failure for this engine — "
            "schedule component replacement rather than an operating-point change."
        )
    return recs


def render(
    prep: dict,
    feat: dict,
    disc: dict,
    valid: dict | None,
    scm_result: dict,
    rca_result: dict,
    intervention_result: dict,
    cf_result: dict,
    out_path: str | Path,
    dataset_label: str = "NASA C-MAPSS FD001",
) -> Path:
    """Render the full dashboard to a self-contained HTML file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    edges_for_graph = (
        valid["validated_edges"] if valid and not valid["validated_edges"].empty
        else disc["edges"]
    )

    fig_health = health_figure(rca_result, feat["failure_threshold"], feat["featured"])
    fig_dev = deviation_figure(rca_result)
    fig_int = intervention_figure(intervention_result)
    fig_dose = dose_response_figure(intervention_result)
    fig_cf = counterfactual_figure(cf_result, feat["failure_threshold"])
    graph_svg = causal_graph_svg(edges_for_graph, rca_result)

    stages = [
        ("1 Preprocessing", f"{prep['n_rows']:,} rows · {prep['n_units']} engines", "done"),
        ("2 Features", f"HI var {feat['health_indicator_meta']['explained_variance_ratio']:.0%}", "done"),
        ("3 Discovery PCMCI+", f"{disc['n_directed_edges']} directed edges", "done"),
        ("4 Validation", f"{valid['n_validated']} stable edges" if valid else "not run", "done" if valid else "est"),
        ("5 SCM fitting", f"mean R² {scm_result['mean_r2']:.2f}", "done"),
        ("6 Root cause", f"{len(rca_result['root_causes'])} causes ranked", "done"),
        ("7 Intervention", f"do({intervention_result.get('best_variable', '-')})", "est"),
        ("8 Counterfactual", "valid" if cf_result.get("found") else "none achievable", "est"),
    ]
    stage_html = "".join(
        f"<div class='stage {cls}'><b>{html.escape(name)}</b>"
        f"<span class='val'>{html.escape(str(value))}</span></div>"
        for name, value, cls in stages
    )

    ranking = rca_result["ranking"][
        ["variable", "abs_deviation", "causal_reach", "n_symptoms_explained",
         "is_root_cause", "root_cause_score"]
    ].head(10).rename(
        columns={
            "variable": "signal", "abs_deviation": "deviation (sd)",
            "causal_reach": "causal reach", "n_symptoms_explained": "symptoms explained",
            "is_root_cause": "verdict", "root_cause_score": "score",
        }
    )
    rca_table = _table(
        ranking,
        numeric_cols={"deviation (sd)", "causal reach", "score"},
        root_col="verdict",
    )

    int_ranking = intervention_result.get("ranking")
    int_table = (
        _table(
            int_ranking[["variable", "do_value", "health_factual", "health_intervened",
                         "health_improvement"]].rename(
                columns={"variable": "signal", "do_value": "do(value)",
                         "health_factual": "health loss (factual)",
                         "health_intervened": "health loss (under do)",
                         "health_improvement": "improvement"}
            ),
            numeric_cols={"do(value)", "health loss (factual)",
                          "health loss (under do)", "improvement"},
        )
        if int_ranking is not None and not int_ranking.empty
        else "<p class='empty'>No interventions evaluated.</p>"
    )

    cf_metrics = ""
    if cf_result.get("found") and cf_result.get("best"):
        m = cf_result["best"]["metrics"]
        r = cf_result["best"]["result"]
        cf_metrics = "".join(
            f"<div class='metric'><b>{label}</b><span>{value}</span></div>"
            for label, value in [
                ("Valid", "yes" if m["valid"] else "no"),
                ("Proximity", f"{m['proximity_sd']:.2f} sd"),
                ("Sparsity", f"{m['sparsity']} var"),
                ("Plausible", "yes" if m["plausible"] else "no"),
                ("Health cut", f"{m['health_reduction']:.3f}"),
                ("do value", f"{r['cf_value']:.2f}"),
            ]
        )

    recs_html = "".join(f"<li>{r}</li>" for r in _recommendations(rca_result, intervention_result, cf_result))

    domain = (valid or {}).get("domain", {})
    domain_line = (
        f"{domain['share_downstream_consistent']:.0%} of validated edges follow the turbofan "
        f"gas-path ordering ({domain['n_downstream_consistent']}/{domain['n_checked']})."
        if domain.get("n_checked") else "Domain consistency not evaluated."
    )

    body = f"""
<div class="wrap">
<header class="top">
  <h1>Actionable Root Cause Analysis — Causal AI Pipeline</h1>
  <p class="sub">Integrating causal discovery and counterfactual explainability for predictive
  maintenance. Prototype run on {html.escape(dataset_label)}.</p>
  <div class="meta">
    <span class="chip">Dataset: {html.escape(dataset_label)}</span>
    <span class="chip">Discovery: PCMCI+ (&tau;<sub>max</sub>={disc['tau_max']}, &alpha;={disc['pc_alpha']})</span>
    <span class="chip">{disc['n_realisations']} engines pooled as realisations</span>
    <span class="chip">SCM: linear-Gaussian, mean R&sup2; {scm_result['mean_r2']:.2f}</span>
    <span class="chip">Engine under analysis: {rca_result['unit']}</span>
    <span class="chip">Generated {stamp}</span>
  </div>
</header>

<div class="pipeline">{stage_html}</div>

<section>
  <h2>Machine health overview</h2>
  <p class="lead">Health loss is an unsupervised composite of the sensor channels, computed as a
  readout function rather than a node in the causal graph. Engine
  {rca_result['unit']} crosses the failure threshold at cycle {rca_result['onset_cycle']} of
  {rca_result['total_cycles']}.</p>
  <div class="figwrap"><img class="fig" alt="Health trajectory" src="data:image/png;base64,{fig_health}"></div>
</section>

<section>
  <h2>Discovered causal structure</h2>
  <p class="lead">Graph learned by PCMCI+ from {disc['n_realisations']} engines and filtered to
  edges that survive bootstrap resampling. Red nodes are root causes, amber nodes are downstream
  symptoms. {html.escape(domain_line)}</p>
  {graph_svg}
  <div class="legend">
    <span><i style="background:var(--danger)"></i>root cause</span>
    <span><i style="background:var(--warn)"></i>downstream symptom</span>
    <span><i style="background:var(--chip);border:1px solid var(--muted)"></i>other variable</span>
    <span>edge thickness = causal strength · hover a node to highlight</span>
  </div>
</section>

<section>
  <h2>Root cause analysis</h2>
  <p class="lead">At failure most signals deviate at once, so magnitude alone cannot separate
  cause from consequence. Ranking combines deviation from this engine's own healthy baseline with
  causal reach over the other deviating signals; variables that have a deviating ancestor are
  classified as symptoms and demoted.</p>
  <div class="grid2">
    <div class="figwrap"><img class="fig" alt="Deviation profile" src="data:image/png;base64,{fig_dev}"></div>
    <div>{rca_table}</div>
  </div>
</section>

<section>
  <h2>Intervention analysis — do-operations</h2>
  <p class="lead">Each candidate root cause is clamped to its healthy range
  {intervention_result['lead_time']} cycles before onset and the fitted SCM is re-simulated
  forward. The health readout is recomputed from the simulated sensors, so the effect on machine
  health is derived rather than assumed.</p>
  <div class="grid2">
    <div class="figwrap"><img class="fig" alt="Intervention effects" src="data:image/png;base64,{fig_int}"></div>
    {"<div class='figwrap'><img class='fig' alt='Dose response' src='data:image/png;base64," + fig_dose + "'></div>" if fig_dose else "<div></div>"}
  </div>
  <div style="margin-top:18px">{int_table}</div>
</section>

<section>
  <h2>Counterfactual explanation</h2>
  <p class="lead">Abduction recovers this engine's exogenous noise from its observed trajectory;
  the intervention is then applied and the model re-simulated with that noise held fixed, so the
  twin world differs from the factual one only through the intervention.</p>
  <div class="cf">{cf_result.get('explanation', 'No counterfactual generated.')}</div>
  <div class="metrics">{cf_metrics}</div>
  {"<div class='figwrap' style='margin-top:18px'><img class='fig' alt='Counterfactual trajectory' src='data:image/png;base64," + fig_cf + "'></div>" if fig_cf else ""}
</section>

<section>
  <h2>Maintenance recommendations</h2>
  <p class="lead">Decision support output: what to inspect, what to ignore, and what to change.</p>
  <ol class="recs">{recs_html}</ol>
</section>

<section class="scope">
  <h2>Scope &amp; validity of these results</h2>
  <p class="lead">Stated explicitly, because a decision-support tool that hides which numbers are
  validated is not decision support.</p>
  <ul>
    <li><b>Green stages (1-6)</b> are computed from data. <b>Amber stages (7-8)</b> are
    model-based estimates: C-MAPSS contains no interventional data, so no intervention or
    counterfactual result here can be checked against ground truth.</li>
    <li>The causal graph is <b>robustness-validated, not correctness-validated</b>. C-MAPSS ships
    no ground-truth graph; stability, sensitivity, cross-method agreement and gas-path
    plausibility are what can be established.</li>
    <li>Degradation is a latent common driver of all sensors. It is handled by per-engine
    detrending, which restores the PCMCI stationarity and sufficiency assumptions but removes the
    trend as a discoverable pathway.</li>
    <li>C-MAPSS is <b>simulated</b>. Effect magnitudes should be read as relative rankings, not
    absolute engineering quantities.</li>
    <li>The composite health indicator is deliberately excluded from the causal variable set: a
    deterministic function of other nodes violates faithfulness and would corrupt discovery.</li>
    <li>This pipeline is a <b>prototype</b>. The same interfaces run against causRCA, which
    provides an expert ground-truth graph and 100 labelled interventions — enabling the
    validation that C-MAPSS structurally cannot support.</li>
  </ul>
</section>

<footer>
  Actionable RCA framework · Modules 1-9 · generated {stamp} ·
  self-contained report, no external dependencies
</footer>
</div>
"""

    doc = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Actionable RCA Dashboard — Causal AI for Predictive Maintenance</title>"
        f"<style>{_CSS}</style></head><body>{body}</body></html>"
    )
    out_path.write_text(doc, encoding="utf-8")
    return out_path


def run(*args, **kwargs) -> Path:
    """Execute Module 9 (alias of `render`)."""
    return render(*args, **kwargs)
