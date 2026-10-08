"""
All the settings for the Actionable RCA pipeline live here.

Change a value in this file and re-run `python scripts/run_pipeline.py`.
There are no command-line flags to remember.

The values below are PROTOTYPE settings, chosen so the whole pipeline finishes
quickly on a laptop. Making them bigger/better is hyperparameter tuning, which
is a later step - right now the goal is just to get every stage working.
"""

# =============================================================================
# 1. WHICH DATA TO USE
# =============================================================================

# Which C-MAPSS subset. FD001 is the simplest one: a single operating
# condition and a single fault mode, which is what we want for a prototype.
DATASET = "FD001"

# How many engines to pool together for causal discovery.
# Start small. Causal discovery gets dramatically slower as this grows,
# because more data means more links pass the statistical test, which makes
# the graph denser and the search much bigger.
# 10 engines is verified to run in roughly 10 seconds.
ENGINES_FOR_DISCOVERY = 10

# Which engine to analyse in the root-cause / counterfactual stages.
# Leave as None to automatically pick the engine that degrades earliest.
ENGINE_TO_ANALYSE = None


# =============================================================================
# 2. WHICH SENSORS TO TREAT AS CAUSAL VARIABLES
# =============================================================================

# C-MAPSS has 21 sensors, but several measure nearly the same thing
# (for example s_8 is fan speed and s_13 is *corrected* fan speed).
# Feeding near-duplicates into causal discovery makes the graph dense and slow
# without adding real information, so we keep one sensor per physical location.
#
# Set this to None to use every non-constant channel instead.
CAUSAL_VARIABLES = [
    "s_2",   # temperature at LPC outlet
    "s_3",   # temperature at HPC outlet
    "s_4",   # temperature at LPT outlet
    "s_7",   # pressure at HPC outlet
    "s_8",   # fan speed
    "s_11",  # static pressure at HPC outlet
    "s_12",  # ratio of fuel flow to pressure
    "s_15",  # bypass ratio
    "s_17",  # bleed enthalpy
    "s_20",  # HPT coolant bleed
    "s_21",  # LPT coolant bleed
]

# Variables that can cause other things but can never be caused by them.
# The operational settings are the commanded flight condition (altitude, Mach,
# throttle) - the engine reacts to them, never the other way around.
# Telling the algorithm this is both physically correct and saves time.
EXOGENOUS_VARIABLES = ["setting_1", "setting_2"]


# =============================================================================
# 3. CAUSAL DISCOVERY (PCMCI+)
# =============================================================================

# How many cycles back in time a cause is allowed to act.
# 2 means a sensor can influence another sensor now, or 1 or 2 cycles later.
MAX_LAG = 2

# Statistical significance level for keeping a causal link.
# Smaller = stricter = fewer edges = faster and fewer false positives.
SIGNIFICANCE = 0.01

# These two put a ceiling on how hard the algorithm searches.
# Without them the search is combinatorial and may never finish.
# They trade a little accuracy for the ability to actually get an answer.
MAX_CONDITION_SET_SIZE = 3
MAX_COMBINATIONS = 2

# Remove the slow degradation trend before learning the graph.
# The trend is a hidden common cause that affects every sensor at once, so
# leaving it in creates edges everywhere. Keep this True.
REMOVE_TREND_FOR_DISCOVERY = True


# =============================================================================
# 4. GRAPH VALIDATION
# =============================================================================

# Validation re-runs causal discovery several times, so it is the slowest
# stage. Set to False for a quick demo run.
RUN_VALIDATION = True

# How many times to re-run discovery on a random subset of engines.
# An edge that keeps reappearing is trustworthy; one that rarely appears
# is probably a false positive. Each run costs a few seconds.
BOOTSTRAP_RUNS = 8

# An edge must appear in at least this fraction of the bootstrap runs
# to be kept in the final graph.
# 0.5 means "found in the majority of runs", which is the usual convention.
# Measured on this data: 0.5 keeps 8 of 16 edges, 0.6 keeps 5, 0.75 keeps 3.
# Setting it too high leaves a graph too sparse to tell causes from symptoms.
STABILITY_THRESHOLD = 0.5


# =============================================================================
# 5. INTERVENTIONS AND COUNTERFACTUALS
# =============================================================================

# How many top root causes to test interventions on.
INTERVENTION_CANDIDATES = 4

# How many cycles before failure we assume maintenance happens.
# Intervening at the exact moment of failure would be pointless.
MAINTENANCE_LEAD_TIME = 20


# =============================================================================
# 6. OUTPUT AND CACHING
# =============================================================================

# Where the final dashboard HTML is written.
DASHBOARD_PATH = "reports/actionable_rca_dashboard.html"

# Reuse saved results from earlier runs instead of recomputing.
# This is what makes re-running cheap: causal discovery only happens once.
# Set to False to force everything to recompute from scratch.
USE_CACHE = True
