"""Causal discovery: the domain rule about alarms, and how recordings vote."""

from causalpdm.pipeline import discovery

VARIABLES = ["pump_ok", "level_ok", "pump_alarm", "level_alarm"]
ALARMS = ["pump_alarm", "level_alarm"]


def test_no_link_may_start_at_an_alarm():
    """An alarm reports a condition; it is never allowed to cause another signal."""
    allowed = discovery.allowed_links(VARIABLES, 2, ALARMS)
    for j, effect in enumerate(VARIABLES):
        for (i, lag), mark in allowed[j].items():
            cause = VARIABLES[i]
            if cause in ALARMS and i != j:
                # Only the mirror of "signal -> alarm" may appear: "<?-" means
                # the arrow points back into the alarm, not out of it.
                assert lag == 0 and mark == "<?-", f"{cause} may cause {effect}"


def test_an_alarm_may_still_depend_on_its_own_past():
    allowed = discovery.allowed_links(VARIABLES, 2, ALARMS)
    alarm = VARIABLES.index("pump_alarm")
    assert (alarm, -1) in allowed[alarm]


def test_same_moment_links_are_declared_from_both_ends():
    """Tigramite rejects a same-moment link that is only declared one way round."""
    allowed = discovery.allowed_links(VARIABLES, 1, ALARMS)
    mirror = {"-?>": "<?-", "<?-": "-?>", "o?o": "o?o"}
    for j in range(len(VARIABLES)):
        for (i, lag), mark in allowed[j].items():
            if lag == 0:
                assert allowed[i][(j, 0)] == mirror[mark]


def test_signals_still_point_into_alarms():
    allowed = discovery.allowed_links(VARIABLES, 1, ALARMS)
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


def test_screening_removes_same_moment_links_but_not_lagged_ones():
    pairs = {frozenset(("pump_ok", "pump_alarm"))}
    allowed = discovery.allowed_links(VARIABLES, 1, ALARMS, same_moment_pairs=pairs)
    pump, level = VARIABLES.index("pump_ok"), VARIABLES.index("level_ok")
    pump_alarm = VARIABLES.index("pump_alarm")
    assert (pump, 0) in allowed[pump_alarm]          # screened in
    assert (level, 0) not in allowed[pump]           # screened out
    assert (level, -1) in allowed[pump]              # lagged links untouched


def test_screening_keeps_each_signals_strongest_partners():
    import numpy as np
    import pandas as pd
    from tigramite import data_processing as pp
    from tigramite.independence_tests.gsquared import Gsquared

    random = np.random.default_rng(0)
    driver = random.integers(0, 2, 2000)
    noisy = lambda flip: np.where(random.random(2000) < flip, 1 - driver, driver)
    table = pd.DataFrame({"a": driver, "b": noisy(0.05), "c": noisy(0.30),
                          "d": random.integers(0, 2, 2000)}).astype(float)
    test = Gsquared()
    test.set_dataframe(pp.DataFrame(table.to_numpy(), var_names=list(table.columns)))
    kept = discovery.screen_same_moment_links(test, list(table.columns), partners=1,
                                              significance=0.001)
    assert frozenset(("a", "b")) in kept               # a's strongest partner
    assert not any("d" in pair for pair in kept)       # unrelated noise never kept
