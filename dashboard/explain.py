"""
All the explanatory text for the dashboard lives here.

Keeping the words separate from the code means app.py stays readable, and the
explanations can be edited (for a presentation, or for the thesis text) without
touching any logic.
"""

# =============================================================================
# Human-readable sensor names, so charts never show a bare code like "s_11"
# =============================================================================

SENSOR_LABELS = {
    "s_1": "Fan inlet temperature",
    "s_2": "LPC outlet temperature",
    "s_3": "HPC outlet temperature",
    "s_4": "LPT outlet temperature",
    "s_5": "Fan inlet pressure",
    "s_6": "Bypass duct pressure",
    "s_7": "HPC outlet pressure",
    "s_8": "Fan speed",
    "s_9": "Core speed",
    "s_10": "Engine pressure ratio",
    "s_11": "HPC outlet static pressure",
    "s_12": "Fuel flow / pressure ratio",
    "s_13": "Corrected fan speed",
    "s_14": "Corrected core speed",
    "s_15": "Bypass ratio",
    "s_16": "Burner fuel-air ratio",
    "s_17": "Bleed enthalpy",
    "s_18": "Demanded fan speed",
    "s_19": "Demanded corrected fan speed",
    "s_20": "HPT coolant bleed",
    "s_21": "LPT coolant bleed",
    "setting_1": "Altitude (commanded)",
    "setting_2": "Mach number (commanded)",
    "setting_3": "Throttle angle (commanded)",
}

# Which part of the engine each sensor sits in - used to colour graphs.
SENSOR_SECTION = {
    "s_1": "Fan / inlet", "s_5": "Fan / inlet", "s_6": "Fan / inlet",
    "s_8": "Fan / inlet", "s_13": "Fan / inlet",
    "s_2": "Low-pressure compressor",
    "s_3": "High-pressure compressor", "s_7": "High-pressure compressor",
    "s_11": "High-pressure compressor", "s_9": "High-pressure compressor",
    "s_14": "High-pressure compressor",
    "s_12": "Combustor", "s_16": "Combustor", "s_17": "Combustor",
    "s_20": "High-pressure turbine",
    "s_4": "Low-pressure turbine", "s_21": "Low-pressure turbine",
    "s_10": "Whole engine", "s_15": "Whole engine",
}


def label(name: str) -> str:
    """Turn a sensor code into something a human can read."""
    friendly = SENSOR_LABELS.get(name)
    return f"{name} — {friendly}" if friendly else name


def short_label(name: str) -> str:
    """Shorter version, for chart axes where space is tight."""
    return SENSOR_LABELS.get(name, name)


# =============================================================================
# What each module does - short version for cards, long version for the page
# =============================================================================

MODULES = {
    "preprocessing": {
        "number": 1,
        "title": "Data Preprocessing",
        "short": "Clean the sensor data and normalise each engine against itself.",
        "long": """
**What this step does**

The raw C-MAPSS files are already very clean — no missing values, no gaps. So this step is mostly
about making the data *suitable for causal analysis* rather than fixing errors:

- **Drop sensors that never change.** Six of the 21 channels are constant in FD001. A constant
  carries no information and would break the statistical tests used later.
- **Normalise each engine against itself.** Every engine starts with a different, unknown amount
  of initial wear. If we standardised across the whole fleet instead, that engine-to-engine
  difference would look like a shared cause of everything.
- **Compute Remaining Useful Life (RUL)** from each engine's run-to-failure history.

**The one design decision that really matters**

Engine degradation is a slow trend that pushes *every* sensor in the same direction at the same
time. In causal language that is a **hidden common cause**, and it creates a genuine conflict:

- Causal discovery needs the trend *removed*, because a hidden common driver makes every sensor
  look connected to every other one, and because the algorithm assumes the data is stationary.
- The health score and all simulation need the trend *kept*, because the trend **is** the
  degradation we are trying to explain.

So this step produces **two versions** of the data, and every later step uses whichever one is
correct for it. That is why the pipeline never has to compromise on either side.
        """,
    },
    "features": {
        "number": 2,
        "title": "Feature Engineering & Health Score",
        "short": "Build rolling features and a single health score that defines failure.",
        "long": """
**What this step does**

Adds rolling averages, rolling standard deviations, trend slopes and derivatives — features that
describe *where a signal is heading*, not just where it currently is. Then it builds a single
**health score** from all the sensors, and uses that score to define when an engine has entered a
degraded state.

**Why the health score is deliberately kept OUT of the causal graph**

This is a real methodological trap, and avoiding it is a point worth making in a thesis defence.

The health score combines many sensors into one number, so it is a *deterministic function* of
other variables in the graph. Feeding it into causal discovery would break the **faithfulness**
assumption that PCMCI depends on — the algorithm would find impossible independence patterns and
produce nonsense. And if the score had been fitted against RUL, the label would leak into the
graph, making every "explanation" just a restatement of the target.

So the health score is built as a **readout function**: fitted once, then applied to whatever
sensor values it is handed — real, intervened, or counterfactual. It reports on the system without
ever being part of it. This is exactly what lets the intervention and counterfactual pages below
show an effect on *health* while keeping the causal model clean.

**Defining the failure event**

Root cause analysis has to explain *something specific*. C-MAPSS gives no per-engine fault label,
so the event is defined as the first cycle where the health score crosses a fleet-level threshold.
Every root-cause result downstream is conditional on this definition — which is why it is stated
openly rather than buried.
        """,
    },
    "discovery": {
        "number": 3,
        "title": "Causal Discovery (PCMCI+)",
        "short": "Learn which sensors actually influence which, and after how long.",
        "long": """
**What PCMCI+ is**

PCMCI+ (Runge, 2020) is a causal discovery algorithm built specifically for time series. The name
comes from its two phases:

- **PC** (after Peter Spirtes and Clark Glymour) — the *condition-selection* phase. For each
  variable it narrows a long list of possible causes down to a short list of likely parents, by
  repeatedly asking: *"are these two variables still dependent once I account for a third?"*
- **MCI** — *Momentary Conditional Independence*. Each surviving link is then tested again, this
  time conditioning on the parents of **both** ends of the link. This second phase is what makes
  PCMCI+ suitable for sensor data: without it, strong autocorrelation produces large numbers of
  false links.

The "+" version additionally handles **contemporaneous** links — causes that act within the same
time step — not just delayed ones.

**Why it fits this problem**

Turbofan sensors are heavily autocorrelated, physically coupled, and act on each other with
delays. PCMCI+ handles exactly that combination, and it reports *at which time lag* each influence
acts, which is directly meaningful for maintenance: a cause acting two cycles ahead gives you two
cycles of warning.

**What the algorithm assumes** — and a thesis must defend these:

1. **Causal sufficiency** — no hidden common causes. Degradation is one, which is precisely why
   the trend is removed before this step runs.
2. **Faithfulness** — no variable is a deterministic function of others. This is why the health
   score is excluded.
3. **Stationarity** — the mechanism does not change over an engine's life. Detrending helps here
   too.

**How the engines are pooled**

We have many short trajectories, not one long one. Concatenating them end-to-end would invent
causal links across engine boundaries — the last cycle of one engine "causing" the first cycle of
the next. Instead each engine is passed as a separate **realisation** of the same mechanism. This
is both statistically correct and, measured on this data, faster than concatenating.

**A practical finding worth reporting**

Discovery cost is driven by how **dense** the graph is, not by how many samples there are. That
leads to a counterintuitive result measured on this data: *more data made it slower*. Doubling the
data doubles statistical power, so weak links that were previously rejected now survive, the
candidate graph stays dense, and the number of conditioning sets to check grows combinatorially.

The lesson — **constraint-based causal discovery does not scale naively to industrial sensor data,
because sensor redundancy keeps the graph dense** — is a genuine methodological contribution, not
an implementation complaint. It is why the variable list is curated and the search is bounded.
        """,
    },
    "validation": {
        "number": 4,
        "title": "Graph Validation",
        "short": "Check whether the discovered graph can be trusted.",
        "long": """
**The honest limitation, stated first**

C-MAPSS ships **no ground-truth causal graph**. NASA released the simulator's outputs, not its
internal wiring. So it is impossible to check whether the discovered graph is *correct*.

Conflating "robust" with "correct" is the most common overclaim in applied causal-discovery work,
and an examiner will look for it. What *can* be established is robustness and plausibility, via
four independent checks:

1. **Stability** — re-run discovery on random subsets of engines. An edge that keeps reappearing
   is a real finding; one that appears occasionally is probably a false positive.
2. **Sensitivity** — does the graph survive changes to the significance level and the maximum lag?
   If the graph changed completely whenever a setting moved, it would not be a result.
3. **Cross-method agreement** — does a completely different kind of estimator (lagged regression,
   which is score-based rather than constraint-based) agree on the strongest links?
4. **Physical plausibility** — air flows through a turbofan in a fixed order: fan → low-pressure
   compressor → high-pressure compressor → combustor → high-pressure turbine → low-pressure
   turbine. Edges that follow that flow are physically expected; ones running against it need a
   feedback-control explanation.

**Where this goes next**

When this same framework is pointed at **causRCA** — a dataset that *does* ship an expert-built
ground-truth causal graph — these four robustness checks are replaced by real correctness metrics:
precision, recall, F1 and structural Hamming distance. The code for that already exists
(`validation.score_against_ground_truth`); C-MAPSS simply cannot support it.
        """,
    },
    "scm": {
        "number": 5,
        "title": "SCM Fitting",
        "short": "Turn the graph into equations, so we can simulate changes.",
        "long": """
**Why this step is mandatory**

The causal graph says *which* sensors influence which. It says nothing about *how much*, or in what
direction, or with what noise. A graph on its own cannot answer "what if we changed this?"

For that we need a **Structural Causal Model** — one equation per sensor:

$$X_i(t) = f_i\\big(\\text{its causal parents}\\big) + e_i(t)$$

**The noise term is the important part**

That $e_i(t)$ is not a nuisance to be minimised. It is what makes counterfactual reasoning
possible at all. It captures everything about *this particular engine* that the equations do not
explain. Pearl's counterfactual procedure works by recovering it and holding it fixed — which is
what makes an explanation specific to one engine rather than to the fleet average.

Without this module, the intervention and counterfactual pages simply could not exist. This was
the box missing from the original pipeline sketch.

**Three deliberate choices**

- **Linear equations.** For a prototype this is right: identifiable given the graph, fast, and
  fully auditable — you can read the coefficients and see what the model believes. The code accepts
  any scikit-learn regressor, so a nonlinear mechanism can be substituted later without changing
  any downstream module.
- **Fitted on the trend-intact data**, because the equations must live in the same space as the
  health score and every simulation.
- **Each sensor's own history is included** as a parent. Sensors are strongly autocorrelated;
  omitting self-history would push that variance into the other coefficients and inflate the
  apparent causal effects.
        """,
    },
    "rca": {
        "number": 6,
        "title": "Root Cause Analysis",
        "short": "Separate the true causes from the downstream symptoms.",
        "long": """
**The problem this solves**

Near failure, almost every sensor looks abnormal at once. Rank them by "how abnormal" and you get
the loudest **symptom**, not the **cause** — and an engineer sent to fix a symptom fixes nothing.

Correlation genuinely cannot solve this, because correlation has no direction. Direction is
exactly what a causal graph provides. This is the core argument of the thesis, and this module is
where it pays off.

**How the ranking is built**

1. **Measure deviation.** For each sensor, compare the pre-failure window against *this engine's
   own* healthy baseline. A within-engine comparison, so normal fleet variation cannot masquerade
   as a cause.
2. **Trace causal paths.** For each deviating sensor, which *other* deviating sensors can it reach
   through the causal graph?
3. **Split causes from symptoms.** If a deviating sensor has another deviating sensor as a causal
   ancestor, it sits downstream — it is a symptom. If nothing deviating explains it, it is a root
   cause candidate.
4. **Rank** by deviation size × causal reach, with symptoms explicitly demoted.

**Why the output is explainable**

Every root cause comes with the actual causal chains through which it produces the observed
symptoms. That is a genuine explanation an engineer can check against their own understanding of
the machine — not a score from a black box.
        """,
    },
    "intervention": {
        "number": 7,
        "title": "Intervention Analysis",
        "short": "Ask: if we controlled this sensor, what would happen to engine health?",
        "long": """
**The question**

*If we had held this sensor inside its healthy range, what would have happened to engine health?*

Written formally, for a candidate cause $V$ and the health readout $H$:

$$\\text{effect}(V) = E[H \\mid do(V = v)] - E[H]$$

The $do(\\cdot)$ notation is Pearl's: it means *force* the variable to a value, as an external
action, rather than merely *observe* it at that value. That distinction is the whole point.
Observing low pressure tells you the engine is healthy; *forcing* low pressure is a maintenance
action, and only a causal model can tell you what it would achieve.

**How it is computed here**

The chosen sensor is clamped in the fitted SCM and the model is re-simulated forward through the
mechanisms in causal order. The health score is then **recomputed from the simulated sensors** —
so the effect on health is derived, never assumed. This is exactly why the health score was built
as a readout function rather than a graph node.

**Scope — and this matters for the thesis**

On C-MAPSS these are **model-based estimates that cannot be validated.** The dataset contains no
interventional data, so there is no ground truth to compare against. The identical code run on
**causRCA** *can* be validated, against its 100 labelled real interventions. That contrast is the
central argument for causRCA being the thesis centrepiece rather than C-MAPSS.
        """,
    },
    "counterfactual": {
        "number": 8,
        "title": "Counterfactual XAI",
        "short": "Ask: would this specific engine have failed, if one thing had been different?",
        "long": """
**How this differs from the previous page**

Interventions ask a *general* question: what happens if we clamp this variable. Counterfactuals
ask the retrospective, engine-specific question a maintenance engineer actually asks:

> *"This engine failed. Would it have failed if this sensor had stayed in range?"*

Answering that requires **Pearl's three-step procedure**, and the code implements it literally:

| Step | What happens | Why it matters |
|---|---|---|
| **Abduction** | Recover this engine's own noise $e(t)$ from its observed history | Makes the answer about *this* engine, not an average one |
| **Action** | Modify the model with $do(X = x')$ | The hypothetical change |
| **Prediction** | Re-simulate with that noise **held fixed** | The twin world differs *only* through the change |

Holding the noise fixed is the essential trick. Without it you would be comparing this engine
against a generic engine, and any difference you saw could just be the difference between engines.
With it, the two trajectories are the same engine living two different histories.

**What makes a counterfactual explanation good**

A useful explanation is not just any change that avoids failure. Counterfactual XAI has standard
quality criteria, and reporting all of them is what separates a rigorous thesis from a demo:

- **Validity** — does the change actually avoid the failure?
- **Proximity** — how big a change was needed? Smaller is better.
- **Sparsity** — how many variables had to move? Fewer is better.
- **Plausibility** — is the required value inside the engine's real operating range? An
  explanation demanding a physically impossible value is not a maintenance instruction.
        """,
    },
}


# =============================================================================
# Hyperparameters: what they mean, what range makes sense, what we chose
# =============================================================================

HYPERPARAMETERS = [
    {
        "group": "Data",
        "name": "DATASET",
        "meaning": "Which C-MAPSS subset to use. FD001 has one operating condition and one fault "
                   "mode, which is the cleanest setting for causal analysis.",
        "range": "FD001, FD002, FD003, FD004",
        "effect": "FD002/FD004 add six operating conditions, which act as an extra context "
                  "variable and need conditioning on the regime.",
    },
    {
        "group": "Data",
        "name": "ENGINES_FOR_DISCOVERY",
        "meaning": "How many engines are pooled together when learning the causal graph.",
        "range": "5 – 100 (dataset has 100)",
        "effect": "More engines means more statistical power, but discovery gets dramatically "
                  "slower because more links survive testing and the graph gets denser. "
                  "Measured: 10 engines ≈ 11 s; 20 engines without search bounds did not finish "
                  "in 9 minutes.",
    },
    {
        "group": "Data",
        "name": "CAUSAL_VARIABLES",
        "meaning": "Which sensors are treated as causal variables.",
        "range": "any subset of the 14 non-constant channels",
        "effect": "Near-duplicate sensors (fan speed vs corrected fan speed) make the graph dense "
                  "and slow without adding information. One sensor per physical location is the "
                  "sensible default.",
    },
    {
        "group": "Discovery",
        "name": "MAX_LAG",
        "meaning": "How many cycles back in time a cause is allowed to act (τ_max in PCMCI+).",
        "range": "1 – 10",
        "effect": "Larger values can find slower physical processes, but cost grows fast: the "
                  "number of candidate links is roughly variables² × (lag + 1).",
    },
    {
        "group": "Discovery",
        "name": "SIGNIFICANCE",
        "meaning": "The p-value threshold for keeping a causal link (pc_alpha).",
        "range": "0.001 – 0.05",
        "effect": "Stricter (smaller) means fewer edges, fewer false positives, and much faster "
                  "runtime. Looser values can make discovery intractable because the graph stays "
                  "dense — 0.05 was measured to be slower than the whole rest of the pipeline.",
    },
    {
        "group": "Discovery",
        "name": "MAX_CONDITION_SET_SIZE",
        "meaning": "Largest set of other variables the algorithm conditions on when testing a link.",
        "range": "1 – 5, or unlimited",
        "effect": "This is a tractability control, not a nicety. Unbounded, the search is "
                  "combinatorial and may never finish. Lower values are faster but may leave in "
                  "some links that a deeper search would have removed.",
    },
    {
        "group": "Discovery",
        "name": "MAX_COMBINATIONS",
        "meaning": "How many different conditioning sets of each size to try.",
        "range": "1 – 10",
        "effect": "Same trade-off as above: lower is faster, higher is more thorough.",
    },
    {
        "group": "Discovery",
        "name": "REMOVE_TREND_FOR_DISCOVERY",
        "meaning": "Whether to remove the degradation trend before learning the graph.",
        "range": "True / False",
        "effect": "Should stay True. The trend is a hidden common cause of every sensor; leaving "
                  "it in produces edges almost everywhere and breaks PCMCI's stationarity "
                  "assumption.",
    },
    {
        "group": "Validation",
        "name": "BOOTSTRAP_RUNS",
        "meaning": "How many times discovery is repeated on a random subset of engines.",
        "range": "3 – 50",
        "effect": "More runs give a finer-grained stability estimate but cost one discovery run "
                  "each. With only 8 runs, the recovery rate can only take 9 distinct values.",
    },
    {
        "group": "Validation",
        "name": "STABILITY_THRESHOLD",
        "meaning": "Fraction of bootstrap runs an edge must appear in to be kept.",
        "range": "0.3 – 0.9",
        "effect": "0.5 means 'found in the majority of runs', the usual convention. Measured on "
                  "this data: 0.5 keeps 8 of 16 edges, 0.6 keeps 5, 0.75 keeps 3. Set it too high "
                  "and the graph becomes too sparse to tell causes from symptoms at all.",
    },
    {
        "group": "Analysis",
        "name": "MAINTENANCE_LEAD_TIME",
        "meaning": "How many cycles before failure onset maintenance is assumed to act.",
        "range": "5 – 50",
        "effect": "Intervening at the exact moment of failure would be operationally meaningless. "
                  "Longer lead times give the intervention more time to work, but assume the "
                  "problem was detected earlier.",
    },
    {
        "group": "Analysis",
        "name": "INTERVENTION_CANDIDATES",
        "meaning": "How many of the top-ranked root causes get tested as intervention targets.",
        "range": "1 – 10",
        "effect": "Purely a cost/coverage trade-off; each candidate needs its own simulation.",
    },
]


# =============================================================================
# Scope statement - shown on the overview page and repeated where relevant
# =============================================================================

SCOPE_NOTES = [
    ("Computed from data (Modules 1, 2, 5, 6)",
     "These results follow directly from the dataset and the fitted model.", "ok"),
    ("Robustness-validated only (Modules 3, 4)",
     "The causal graph is checked for stability, sensitivity, cross-method agreement and physical "
     "plausibility — but **not** for correctness, because C-MAPSS ships no ground-truth graph.",
     "warn"),
    ("Model-based estimates (Modules 7, 8)",
     "C-MAPSS contains no interventional data, so intervention effects and counterfactuals cannot "
     "be checked against any ground truth. They are what the fitted model implies.", "warn"),
    ("C-MAPSS is simulated data",
     "Effect sizes should be read as relative rankings, not absolute engineering quantities.",
     "warn"),
    ("Where this goes next",
     "Every module interface is dataset-agnostic. Pointing the same pipeline at **causRCA** — "
     "which provides an expert ground-truth causal graph and 100 labelled real interventions — "
     "converts Modules 3, 4, 7 and 8 from demonstrations into measured results.", "ok"),
]
