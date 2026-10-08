"""
All settings for the causal root cause analysis pipeline.

Edit values here and re-run. Nothing else needs changing.

Every setting has a comment explaining what it does, what it affects, and what
a sensible range is. Values marked TUNED were selected by the tuning stage by
scoring against the causRCA expert graph; values marked FIXED were chosen from
the dataset documentation or measured from the data.
"""

# =============================================================================
# 1. WHICH DATA
# =============================================================================

# Which subsystem of the lathe to analyse.
# causRCA provides three subsystems plus the full machine:
#   "coolant"     15 variables, 10 expert edges, 25 fault runs   <- start here
#   "hydraulics"  17 variables, 18 expert edges, 41 fault runs
#   "probe"       11 variables, 15 expert edges, 34 fault runs
#   "full"        92 variables, 104 expert edges, 100 fault runs
SUBSYSTEM = "coolant"

# How many normal-operation recordings to include alongside the fault runs.
# Normal recordings are needed because alarms never fire in them, which gives
# the model examples of the healthy state. Set to None to use all 170.
NORMAL_RECORDINGS = 60


# =============================================================================
# 2. TURNING EVENT LOGS INTO A TABLE
# =============================================================================

# causRCA records a row only when a variable CHANGES value, so the raw files
# are event logs, not regular time series. We resample onto a fixed grid.

# Seconds between rows after resampling.
# 1.0 s is a good fit here: the documented alarm rules act at 0 s and 5 s, so a
# 1 s grid separates them cleanly without creating a huge table.
RESAMPLE_SECONDS = 1.0

# Shortest recording to keep, in rows. Very short recordings carry too little
# information for the statistical tests.
MIN_RECORDING_ROWS = 30


# =============================================================================
# 3. CAUSAL DISCOVERY (PCMCI+)
# =============================================================================

# How many time steps back a cause may act.
# MEASURED FROM THE DATA: alarm rules in causRCA fire either immediately (~0 s)
# or after a 5 s hold. With a 1 s grid, we need at least lag 5 to see the
# delayed rules, plus headroom. Using 2 would miss them entirely.
MAX_LAG = 8

# Significance level for keeping a causal link.
# Set to None to let the algorithm choose from [0.001, 0.005, 0.01, 0.025, 0.05]
# using its own model-selection score. A fixed number overrides that.
SIGNIFICANCE = 0.01

# Correct for testing many links at once. With 15 variables and 9 lags there
# are over 2000 candidate links, so without correction a good number of edges
# would appear by chance alone.
# Options: "fdr_bh" (Benjamini-Hochberg) or "none".
MULTIPLE_TEST_CORRECTION = "fdr_bh"

# Limits on how hard the algorithm searches for explanations of a link.
# These keep the run time manageable. Higher = more thorough but slower.
MAX_CONDITIONS = 3
MAX_COMBINATIONS = 2

# Every causRCA coolant variable is True/False, so the independence test must
# handle discrete data. A test built for continuous values would be wrong here.
# Options: "discrete" (CMIsymb) or "continuous" (ParCorr).
INDEPENDENCE_TEST = "discrete"


# =============================================================================
# 4. TUNING
# =============================================================================

# causRCA ships an expert-built causal graph, so we can tune by scoring
# candidate settings against it directly. These are the values searched.
TUNING_LAGS = [2, 5, 8, 10]
TUNING_SIGNIFICANCE = [0.001, 0.01, 0.05]

# Which score decides the winner. "f1" balances finding real edges against
# inventing false ones. Alternatives: "precision", "recall", "shd".
TUNING_OBJECTIVE = "f1"


# =============================================================================
# 5. STRUCTURAL CAUSAL MODEL
# =============================================================================

# How each variable's equation is estimated from its causes.
#   "logistic" suits True/False variables (the whole coolant subsystem)
#   "linear"   suits continuous variables
SCM_METHOD = "logistic"


# =============================================================================
# 6. ROOT CAUSE ANALYSIS
# =============================================================================

# How many candidate root causes to report per fault.
TOP_K_ROOT_CAUSES = 3


# =============================================================================
# 7. OUTPUT
# =============================================================================

# Reuse saved results instead of recomputing. Causal discovery is the slow
# step, so this makes repeat runs fast. Set False to recompute everything.
USE_CACHE = True
