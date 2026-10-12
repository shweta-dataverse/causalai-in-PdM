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
#
# This is the most important setting in the file, and it was measured, not
# guessed. causRCA's alarm rules fire either immediately or after the fault has
# been present for 5 seconds. What the method needs is a lag window that covers
# those 5 seconds - and widening the window is far cheaper by making each step
# bigger than by adding more steps.
#
# Measured on the coolant subsystem at significance 0.001 (F1 against the
# expert graph; full table in results/final/tuning_results.csv):
#   grid 1.0s, lag 2  -> covers  2.0s, F1 0.50,  10s to run
#   grid 1.0s, lag 3  -> covers  3.0s, F1 0.50,  13s to run
#   grid 2.5s, lag 2  -> covers  5.0s, F1 0.67,   5s to run
#   grid 2.5s, lag 3  -> covers  7.5s, F1 0.71,   6s to run   <- chosen
#   grid 5.0s, lag 2  -> covers 10.0s, F1 0.56,   3s to run
#
# These were tuned on coolant only. Hydraulics, probe and the full machine are
# run with the same values unchanged, so they are not tuned on their own
# answers.
RESAMPLE_SECONDS = 2.5

# Shortest recording to keep, in rows. Very short recordings carry too little
# information for the statistical tests.
MIN_RECORDING_ROWS = 20


# =============================================================================
# 3. CAUSAL DISCOVERY (PCMCI+)
# =============================================================================

# How many time steps back a cause may act.
#
# With a 2.5 second grid, 3 steps reaches 7.5 seconds back, comfortably past
# the 5 second delay the machine's alarm rules use.
#
# Chosen by scripts/tune_settings.py, which tried every combination and scored
# each against the engineers' own causal map. Lag 3 won at F1 0.71; lag 2 came
# second at 0.67; lag 1 could not reach the delayed rules and scored 0.50.
MAX_LAG = 3

# Significance level for keeping a causal link. Smaller means stricter.
#
# Also chosen by tuning. Being strict matters more here than it usually does:
# at 0.001 the pipeline finds 7 links of which 6 are real, while at 0.05 it
# finds 14 of which only 5 are real. The extra links are noise, and they
# make everything downstream slower and less trustworthy.
#
# There is no automatic choice for the discrete test (tigramite does not
# implement model selection for it). Choosing the level by how well the model
# predicts unseen recordings - the option a plant without an expert graph has -
# picks looser levels (0.01-0.05) and lower graph accuracy: good prediction is
# not the same as the right causes. See scripts/protocol_comparison.py.
SIGNIFICANCE = 0.001

# Correct for testing many links at once. With 15 variables and 9 lags there
# are over 2000 candidate links, so without correction a good number of edges
# would appear by chance alone.
# Options: "fdr_bh" (Benjamini-Hochberg) or "none".
MULTIPLE_TEST_CORRECTION = "fdr_bh"

# Limits on how hard the algorithm searches for explanations of a link.
# These keep the run time manageable. Higher = more thorough but slower.
MAX_CONDITIONS = 3
MAX_COMBINATIONS = 2

# Domain knowledge: an alarm is the controller reporting a condition, so it can
# be an effect but never a cause. Links out of alarms are not even tested.
#
# Measured, pooled discovery (F1 against the expert graph):
#   coolant    0.71 -> 0.74     hydraulics 0.43 -> 0.64     probe 0.18 -> 0.30
#
# The cost: interlocks - an alarm that switches a pump off or sets the error
# state - cannot be learned (5 of 18 hydraulics links, 19 of 104 on the full
# machine). In a plant those actions are documented in the alarm configuration
# and would be added as known links.
ALARMS_ONLY_AS_EFFECTS = True

# How many already-known causes of EACH end of a link the final test conditions
# on. Every extra conditioned signal doubles the counting table the test has to
# fill, so this is the main speed lever (Runge et al. 2019 discuss limiting it).
# At 6 the full machine's final phase took over 0.2 s per test and would have
# run for hours; at 3 it is about 17 ms.
# Measured: coolant unchanged (F1 0.74); hydraulics 0.64 -> 0.62 (precision
# 0.90 -> 1.00); probe 0.30 -> 0.32.
MAX_PARENTS_CONDITIONED = 3

# Full machine only: each signal may be tested against just its 10 strongest
# earlier-in-time candidate causes (its own past always kept). Without this,
# the full machine's final phase did not finish in five hours. Measured on the
# subsystems, where the exact search is feasible: coolant and hydraulics give
# identical graphs; probe 0.32 -> 0.25. Subsystems are run without it.
FULL_MACHINE_EARLIER_CAUSES = 10

# Screening before the search: each signal may be linked within the same time
# step only to its 5 strongest same-moment partners (lagged links unscreened).
#
# Why: on the full machine a cycle starting or stopping switches dozens of
# signals in the same step, so each signal moves with about 71 others and
# PCMCI+ would need ~2.3 million tests (13-45 hours). With the screen it runs
# in minutes. Measured: no change at all on coolant, hydraulics or probe.
# Set to None to switch screening off.
SAME_MOMENT_PARTNERS = 5

# Every causRCA coolant variable is True/False, so the independence test must
# handle discrete data. A test built for continuous values would be wrong here.
# Options: "discrete" (CMIsymb) or "continuous" (ParCorr).
INDEPENDENCE_TEST = "discrete"


# =============================================================================
# 4. TUNING
# =============================================================================

# causRCA ships an expert-built causal graph, so we can tune by scoring
# candidate settings against it directly. These are the values searched.
TUNING_LAGS = [1, 2, 3]
TUNING_SIGNIFICANCE = [0.001, 0.01, 0.05]

# Resampling grids to try, in seconds. This matters more than the lag.
TUNING_GRIDS = [1.0, 2.5, 5.0]

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

# Who is suspected first: the signals the learned graph links to the alarms
# that fired. "parents" = the alarms' direct causes (what the best published
# method does, there with the expert graph), "ancestors" = everything upstream,
# "all" = no restriction. Everything else is still ranked, just after them.
ROOT_CAUSE_SUSPECTS = "parents"


# =============================================================================
# 7. OUTPUT
# =============================================================================

# Reuse saved results instead of recomputing. Causal discovery is the slow
# step, so this makes repeat runs fast. Set False to recompute everything.
USE_CACHE = True
