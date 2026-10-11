"""The data loader should match what the causRCA documentation says is there."""

import pytest

from causalpdm.pipeline import data


# Counts taken from the causRCA dataset README. If these fail, either the data
# on disk is incomplete or the loader is reading the wrong thing.
EXPECTED = {
    "coolant":    {"variables": 15, "edges": 10, "runs": 25, "scenarios": 4},
    "hydraulics": {"variables": 17, "edges": 18, "runs": 41, "scenarios": 9},
    "probe":      {"variables": 11, "edges": 15, "runs": 34, "scenarios": 6},
}


@pytest.mark.parametrize("subsystem", sorted(EXPECTED))
def test_subsystem_matches_documentation(subsystem):
    expected = EXPECTED[subsystem]
    graph = data.load_expert_graph(subsystem)
    runs = data.load_fault_runs(subsystem)

    assert len(graph.variable_names) == expected["variables"]
    assert len(graph.edges) == expected["edges"]
    assert len(runs) == expected["runs"]
    assert len({run.experiment for run in runs}) == expected["scenarios"]


def test_full_machine_graph():
    graph = data.load_expert_graph("full")
    assert len(graph.variable_names) == 92
    assert len(graph.edges) == 104


def test_there_are_170_healthy_recordings():
    assert len(data.load_normal_recordings()) == 170


def test_every_fault_run_knows_its_cause(coolant):
    for run in coolant.fault_runs:
        assert run.caused_by, f"{run.name} has no recorded cause"
        assert run.diagnosis
        assert run.fault_started_at >= 0


def test_causes_are_real_variables(coolant):
    """A fault cannot be caused by a variable that is not in the graph."""
    known = set(coolant.expert_graph.variable_names)
    for run in coolant.fault_runs:
        for cause in run.caused_by:
            assert cause in known, f"{run.name} blames unknown variable {cause}"


def test_expert_edges_connect_real_variables(coolant):
    known = set(coolant.expert_graph.variable_names)
    for cause, effect in coolant.expert_graph.edge_pairs:
        assert cause in known and effect in known


def test_unknown_subsystem_is_rejected():
    with pytest.raises(ValueError):
        data.load("not_a_real_subsystem")


def test_change_logs_have_the_expected_columns(coolant):
    log = data.read_change_log(coolant.fault_runs[0].path)
    assert list(log.columns) == ["time_s", "node", "value", "type"]
    assert len(log) > 0
    assert log["time_s"].is_monotonic_increasing


@pytest.mark.parametrize("subsystem", sorted(EXPECTED))
def test_every_forced_variable_has_a_labelled_state(subsystem):
    """The intervention check needs to know what each cause was forced to."""
    for run in data.load_fault_runs(subsystem):
        assert set(run.forced_states) == set(run.caused_by), run.name


def test_faults_force_switches_both_ways():
    """
    Not every fault switches something off. Replaying all of them as "off"
    was a real bug: it turned the coolant level fault into no fault at all.
    """
    states = {run.experiment: run.forced_states for run in data.load_fault_runs("coolant")}
    assert states["exp_27"]["CLT_Level_lt_Min"] is True
    assert states["exp_27"]["HP_Pump_Ok"] is False
