"""Root cause analysis: who counts as a prime suspect."""

from causalpdm.pipeline import rca


class TinyModel:
    """filter_clogged -> pump_ok -> pump_alarm, and pump_ok -> pump_on (a consequence)."""

    variables = ["filter_clogged", "pump_ok", "pump_on", "pump_alarm"]
    links = {"pump_ok": [("filter_clogged", 1)],
             "pump_alarm": [("pump_ok", 0)],
             "pump_on": [("pump_ok", 1)]}

    def causes_of(self, variable):
        return self.links.get(variable, [])


def test_direct_causes_of_the_alarm_are_suspected():
    assert rca.suspects_behind_alarms(TinyModel(), ["pump_alarm"], "parents") == {"pump_ok"}


def test_ancestors_reach_further_upstream():
    found = rca.suspects_behind_alarms(TinyModel(), ["pump_alarm"], "ancestors")
    assert found == {"pump_ok", "filter_clogged"}


def test_consequences_of_the_fault_are_not_suspected():
    """The pump switching off is caused by the fault, so it must not be blamed for it."""
    for reach in ("parents", "ancestors"):
        assert "pump_on" not in rca.suspects_behind_alarms(TinyModel(), ["pump_alarm"], reach)


def test_no_restriction_suspects_everything():
    assert rca.suspects_behind_alarms(TinyModel(), ["pump_alarm"], "all") == set(TinyModel.variables)


def test_without_a_fired_alarm_nobody_is_excluded():
    assert rca.suspects_behind_alarms(TinyModel(), [], "parents") == set(TinyModel.variables)
