"""
End-to-end checks.

These run the real pipeline on the real data, so they are slower than the rest
of the suite. They exist to catch the kind of problem unit tests miss: a stage
that works alone but produces something the next stage cannot use.
"""

import pytest

from causalpdm.pipeline import runner


@pytest.fixture(scope="module")
def finished_run():
    return runner.run(subsystem="coolant", normal_recordings=20,
                      seconds_per_step=2.5, max_lag=3, significance=0.001,
                      use_cache=False, verbose=False)


@pytest.mark.slow
def test_pipeline_runs_all_the_way_through(finished_run):
    assert finished_run.causal_graph is not None
    assert len(finished_run.tables) > 0
    assert len(finished_run.causal_model.mechanisms) > 0
    assert len(finished_run.root_cause_results) > 0


@pytest.mark.slow
def test_the_discovered_graph_is_better_than_guessing(finished_run):
    """
    A graph no better than chance would mean something is badly wrong. The bar
    here is deliberately low - this is a smoke test, not a quality target.
    """
    assert finished_run.graph_score["correct"] >= 3
    assert finished_run.graph_score["precision"] > 0.3


@pytest.mark.slow
def test_root_cause_analysis_usually_names_a_real_cause(finished_run):
    """The right cause should come first far more often than by chance."""
    assert finished_run.root_cause_score["found_in_top_1"] > 0.5


@pytest.mark.slow
def test_predicted_alarms_match_the_faults_that_really_happened(finished_run):
    """
    The check that makes this dataset worth using: our what-if predictions are
    compared against faults that were genuinely injected into the machine.
    """
    assert finished_run.intervention_score["scenarios_checked"] > 0
    assert finished_run.intervention_score["mean_f1"] > 0.5


@pytest.mark.slow
def test_fixing_the_real_causes_prevents_most_faults(finished_run):
    """
    If we fix exactly the variables that were really forced, the fault should
    mostly disappear. When it does not, it is because causal discovery missed a
    link - which is worth knowing, so the bar allows for some of that.
    """
    assert finished_run.counterfactual_score["known_causes_success_rate"] > 0.5


@pytest.mark.slow
def test_headline_numbers_are_all_present(finished_run):
    headline = finished_run.headline()
    for key in ("graph_f1", "root_cause_top1", "intervention_f1", "counterfactual_success"):
        assert headline[key] is not None, f"{key} is missing from the summary"
