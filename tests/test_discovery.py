"""Causal discovery: the domain rule about alarms, and how recordings vote."""

from causalpdm.pipeline import discovery

VARIABLES = ["pump_ok", "level_ok", "pump_alarm", "level_alarm"]
ALARMS = ["pump_alarm", "level_alarm"]


def test_no_link_may_start_at_an_alarm():
    """An alarm reports a condition; it is never allowed to cause another signal."""
    allowed = discovery.alarms_only_as_effects(VARIABLES, ALARMS, max_lag=2)
    for j, effect in enumerate(VARIABLES):
        for (i, lag), mark in allowed[j].items():
            cause = VARIABLES[i]
            if cause in ALARMS and i != j:
                # Only the mirror of "signal -> alarm" may appear: "<?-" means
                # the arrow points back into the alarm, not out of it.
                assert lag == 0 and mark == "<?-", f"{cause} may cause {effect}"


def test_an_alarm_may_still_depend_on_its_own_past():
    allowed = discovery.alarms_only_as_effects(VARIABLES, ALARMS, max_lag=2)
    alarm = VARIABLES.index("pump_alarm")
    assert (alarm, -1) in allowed[alarm]


def test_same_moment_links_are_declared_from_both_ends():
    """Tigramite rejects a same-moment link that is only declared one way round."""
    allowed = discovery.alarms_only_as_effects(VARIABLES, ALARMS, max_lag=1)
    mirror = {"-?>": "<?-", "<?-": "-?>", "o?o": "o?o"}
    for j in range(len(VARIABLES)):
        for (i, lag), mark in allowed[j].items():
            if lag == 0:
                assert allowed[i][(j, 0)] == mirror[mark]


def test_signals_still_point_into_alarms():
    allowed = discovery.alarms_only_as_effects(VARIABLES, ALARMS, max_lag=1)
    pump, alarm = VARIABLES.index("pump_ok"), VARIABLES.index("pump_alarm")
    assert allowed[alarm][(pump, 0)] == "-?>"
    assert allowed[alarm][(pump, -1)] == "-?>"


LABELS = ([f"fault/exp_{s}/run_{r}" for s in (1, 2, 3) for r in range(5)]
          + [f"normal/rec_{n}" for n in range(10)])


def test_every_replicate_sees_every_fault_scenario():
    """A replicate is a repeat of the whole experiment: one run of each fault."""
    groups = discovery._vote_groups(LABELS, "replicates", seed=0)
    assert len(groups) == 5
    for group in groups:
        scenarios = {LABELS[i].split("/")[1] for i in group if LABELS[i].startswith("fault")}
        assert scenarios == {"exp_1", "exp_2", "exp_3"}
        assert any(LABELS[i].startswith("normal") for i in group)


def test_every_recording_is_used_exactly_once():
    for grouping in ("replicates", "recording"):
        groups = discovery._vote_groups(LABELS, grouping, seed=0)
        used = sorted(i for group in groups for i in group)
        assert used == list(range(len(LABELS)))
