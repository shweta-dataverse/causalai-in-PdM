"""
All the explanatory writing for the dashboard.

Keeping the words here rather than inside app.py means the explanations can be
edited, reused in a report, or read on their own without wading through
Streamlit code.

The writing aims at someone who knows machines but not causal inference. Every
technical idea is introduced through the machine it describes, not through its
mathematics.
"""

# =============================================================================
# What each signal actually is, in plain words
# =============================================================================

SIGNAL_MEANING = {
    "CLF_Filter_Ok":    "Fine-particle filter is in good condition",
    "CLF_A_700307":     "ALARM: coolant filter is clogged",
    "F_Filter_Ok":      "Fleece filter still has fleece left",
    "F_A_700313":       "ALARM: fleece supply is empty, needs replacing",
    "HP_Pump_Ok":       "High-pressure coolant pump is healthy",
    "HP_Pump_isOff":    "High-pressure pump is currently switched off",
    "HP_A_700304":      "ALARM: high-pressure pump motor protection tripped",
    "LP_Pump_Ok":       "Low-pressure coolant pump is healthy",
    "LP_Pump_On":       "Low-pressure pump has been commanded on",
    "LP_A_700301":      "ALARM: low-pressure pump motor protection tripped",
    "LT_Level_Ok":      "Lifting tank level is within range, no overflow",
    "LT_Pump_Ok":       "Lifting tank pump is reporting back correctly",
    "LT_A_700317":      "ALARM: lifting tank is overflowing",
    "CLT_Level_lt_Min": "Coolant tank has dropped below minimum level",
    "CLT_A_700310":     "ALARM: coolant tank below minimum fill level",

    # Hydraulics - rewritten from the dataset's own descriptions.
    "Hyd_IsEnabled":    "Hydraulics is switched on",
    "Hyd_Pressure":     "Hydraulic pressure",
    "Hyd_Valve_P_Up":   "Accumulator charge valve is open (building pressure)",
    "Hyd_Filter_Ok":    "Hydraulic return-oil filter is clean",
    "Hyd_Level_Ok":     "Hydraulic oil level is sufficient",
    "Hyd_Pump_isOff":   "Hydraulic pump is switched off",
    "Hyd_Pump_Ok":      "Hydraulic pump motor is healthy",
    "Hyd_Pump_On":      "Hydraulic pump has been commanded on",
    "Hyd_Temp_lt_70":   "Hydraulic oil is below 70 °C",
    "Hyd_Temp_lt_80":   "Hydraulic oil is below 80 °C",
    "Hyd_A_700202":     "ALARM: hydraulic pressure out of range",
    "Hyd_A_700203":     "WARNING: hydraulic oil above 70 °C",
    "Hyd_A_700204":     "ALARM: hydraulic oil above 80 °C",
    "Hyd_A_700205":     "WARNING: hydraulic oil level close to minimum",
    "Hyd_A_700206":     "ALARM: hydraulic oil level below minimum",
    "Hyd_A_700207":     "ALARM: hydraulic return-oil filter clogged",
    "Hyd_A_700208":     "ALARM: hydraulic pump motor protection tripped",

    # Measuring probe - rewritten from the dataset's own descriptions.
    "MP_Inactive":      "Measuring probe is idle (not measuring)",
    "MPA_InitPos":      "Probe arm is in its parked position",
    "MPA_toInitPos":    "Probe arm has been commanded to park",
    "MPA_toWorkPos":    "Probe arm has been commanded to the measuring position",
    "MPA_WorkPos":      "Probe arm is in the measuring position",
    "MPC_close":        "Probe cover has been commanded to close",
    "MPC_Closed":       "Probe cover is closed",
    "MPC_isOpen":       "Probe cover is open",
    "MPC_open":         "Probe cover has been commanded to open",
    "MPA_A_701124":     "ALARM: probe arm did not reach its end position in time",
    "MPA_A_701125":     "ALARM: probe arm reports both end positions at once",
}


def meaning(signal: str) -> str:
    """Plain-English description of a signal, or its raw name if unknown."""
    return SIGNAL_MEANING.get(signal, signal)


def is_alarm(signal: str) -> bool:
    return "_A_" in signal


# =============================================================================
# One entry per pipeline stage
# =============================================================================

STAGES = {

"data": {
"title": "The data",
"one_line": "Recordings from a real CNC lathe, including faults deliberately created so the true cause is known.",
"body": """
### What we are working with

The data comes from an **industrial CNC vertical lathe** - a large machine that
shapes metal parts while a coolant system sprays fluid to carry away heat and
swarf. If that coolant system fails, the cutting tool overheats and the part is
ruined.

There are two kinds of recording:

**170 recordings of the machine running normally.** Nothing goes wrong. These
tell us what healthy looks like.

**100 recordings of the machine breaking down.** These are the valuable ones,
and they are valuable because of *how* they were made. The researchers connected
the machine's real controller to a simulated copy of the machine, then
deliberately broke things in the simulation - switched off a pump, emptied a
filter - and recorded what the controller did about it.

### Why that matters more than it sounds

Normally when a machine breaks you do not know what caused it. That is the whole
problem you are trying to solve, so you have no answer key to check your working
against.

Here we do. For every one of those 100 breakdowns, the dataset records exactly
which signal was tampered with, at what second, and which alarms followed. The
engineers also drew a map by hand of how the machine's signals affect each other.

So every claim this system makes can be marked right or wrong. That is rare, and
it is the reason this dataset was chosen over the alternatives.

### What a recording looks like

Not a neat table. Each file is a **change log**: a line is written only at the
moment a signal changes value.

```
time_s,node,value,type
0.0,CBC_Closed,True,Binary
0.0,CBC_close,True,Binary
12.4,HP_Pump_Ok,False,Binary
17.4,HP_A_700304,True,Alarm
```

A signal that stays the same for the whole recording produces no lines at all,
so it simply does not appear in the file. Turning this into something a
statistical method can read is the job of the next stage.

### Everything here is a yes/no signal

This surprises people expecting temperature and pressure readings. The coolant
system reports **15 signals and all 15 are true/false**: is the pump healthy,
is the filter clogged, is the alarm ringing.

That is what a machine controller actually exposes. It has already made the
judgement - "pressure has been below threshold for five seconds, therefore the
pump is not OK" - and publishes the conclusion, not the raw reading. The
hydraulics subsystem has one genuine analog value, hydraulic pressure, and the
full machine has fourteen, mostly spindle positions and speeds.
""",
},

"preprocess": {
"title": "Preparing the data",
"one_line": "Turning change logs into regular tables, and working out what healthy looks like.",
"body": """
### The problem

Causal methods expect a table: one row per moment in time, one column per
signal. What we have is a list of changes at irregular moments. Two things need
solving.

**Filling the gaps between changes.** If the pump was reported healthy at second
3 and faulty at second 40, what was it at second 20? Still healthy - a signal
holds its last value until something changes it. So we carry each value forward.
This is not a guess or an approximation; it is exactly how the machine's
controller publishes data, so forward-filling recovers the original signal
rather than inventing one.

**Filling in signals that never appear.** If the fleece filter never changed
during a recording, it is absent from that file entirely. We cannot tell from
the file whether it was fine or whether nobody was watching.

We solve this by learning what healthy looks like from the 170 normal
recordings, and using that. The system works out on its own that every `_Ok`
signal sits at true, and every alarm sits at false, when nothing is wrong. No
one had to tell it.

### Choosing how long a row should cover

This turned out to be the single most important setting in the whole pipeline,
and it is worth understanding why.

The machine's alarm rules work on two timescales. Some fire the instant
something goes wrong. Others require the fault to persist - the manual says
*"when the filter is not OK for at least five seconds, raise alarm 700307"*.
We measured this in the raw timestamps and found exactly that pattern: 31 alarm
events firing almost immediately, and 18 firing right around the five second
mark.

For the method to see a cause five seconds before its effect, its search window
must reach back at least five seconds. There are two ways to do that: take more
steps back, or make each step longer. **Making each step longer is dramatically
cheaper.**

| Row covers | Steps back | Reaches back | Accuracy | Time to run |
|---|---|---|---|---|
| 1 second | 2 | 2 seconds | 0.50 | 7 s |
| **2.5 seconds** | **2** | **5 seconds** | **0.64** | **3 s** |
| 2.5 seconds | 3 | 7.5 seconds | **0.71** | 6 s |
| 1 second | 5 | 5 seconds | - | did not finish in 9 minutes |
| 2.5 seconds | 4 | 10 seconds | 0.57 | 352 s |

Reaching five seconds back is what matters. Reaching it with a coarse grid takes
six seconds; reaching it with a fine grid does not finish at all.
""",
},

"discovery": {
"title": "Finding the causal map",
"one_line": "Working out which signals affect which, and after how long, using PCMCI+.",
"body": """
### What we are asking the computer to do

Given only recordings, work out which signals cause which. Not which move
together - which *cause*. Those are different, and the difference is the whole
point: a thermometer and a thermostat move together, but only one of them makes
the room warm.

### What PCMCI+ is

PCMCI+ is a method built specifically for this, published by Jakob Runge in
2020. The name comes from its two stages.

**Stage one narrows the field.** For each signal it starts with a long list of
everything that might cause it, and repeatedly asks a question of the form:
*"these two still look related - but do they, once I account for a third signal
that might explain both?"* Anything that fails that test is dropped. This is
quick and gets rid of most of the obvious non-causes.

**Stage two is the careful pass.** Every link that survived is tested again,
this time accounting for the likely causes of *both* ends of the link. This
second pass is what makes the method suitable for machine data, where signals
are strongly related to their own recent past. Without it, two signals that are
each merely slow-moving will look causally linked.

The "+" means it also handles causes that act within the same instant, not only
delayed ones. In this machine both happen, so we need it.

### What it assumes, and where those assumptions bite

A method like this is only as good as its assumptions, and a supervisor will
ask about these.

**Nothing important is unmeasured.** If some hidden thing drives two signals at
once, the method will report a link between them that is not real. Here we are
reasonably safe, because the controller publishes essentially everything it
knows about the coolant system.

**No signal is an exact copy of others.** If one signal is calculable exactly
from two others, the mathematics behind the tests breaks down. This is why
composite "health scores" must be kept out of the map.

**The rules do not change over time.** The machine behaves the same way on
Tuesday as on Monday. True here, since these are fixed control rules.

### Why we use a different test than most examples do

Almost every tutorial uses a test built for continuous measurements - it looks
for straight-line relationships. Every coolant signal is true/false. Asking "is
there a straight-line relationship between the pump being healthy and the alarm
ringing" is the wrong question.

So we use the **G-test**, which is built for yes/no data: it compares how often
combinations actually occur against how often they would occur if the signals
were unrelated.

### Correcting for having asked thousands of questions

With 15 signals and four time lags there are over 900 possible links, and we
test each one. If you run 900 tests at a one-in-a-hundred threshold, roughly
nine will look significant purely by chance.

The pipeline applies a standard correction for this. Without it, a good number
of the links on the map would be noise - and every later stage would be
reasoning about a machine that does not exist.
""",
},

"validation": {
"title": "Checking the map",
"one_line": "Scoring the discovered map against the one the engineers drew by hand.",
"body": """
### Why this page is unusual

In most causal analysis you cannot do this at all. You produce a map, and
nobody can tell you whether it is right, because the right answer is exactly
what you were trying to find.

Here the engineers who built the lathe drew their own map of how its signals
affect each other. So we can lay our map over theirs and count.

### The three numbers

**Of the links we found, how many are real?** If we claim seven links and six
appear on the engineers' map, that is 86%. High means we are not making things
up.

**Of the real links, how many did we find?** The engineers drew ten and we found
six, so 60%. High means we are not missing much.

**F1** combines the two into one number, so that a method cannot win by being
timid (claim one link, be right, score 100% on the first measure) or reckless
(claim everything, find all the real ones, score 100% on the second).

### Missing links matter more than wrong ones

A wrong link is noise. A **missing** link is a hole, and holes have consequences
further down the pipeline.

In this run the system never found the link between the lifting tank level and
its overflow alarm. Later, when the system is asked *"would fixing the tank
level have stopped the overflow alarm?"*, it answers no - not because the answer
is no, but because as far as its map is concerned those two signals have nothing
to do with each other.

Six of the twenty-five faults fail for exactly this reason. That is worth
seeing, because it shows how an error in one stage travels through to the final
recommendation, and it tells you which stage to fix.

### A detail worth noticing

The system did not just find the right links - it found the right **delays**.
The filter and fleece alarms came back with a two-step delay, which at 2.5
seconds per step is five seconds: precisely the five second hold written in the
machine's own documentation. The pump alarms came back as instant, which is also
what the manual says.

Nobody told the system about those rules. It read them out of the data.
""",
},

"scm": {
"title": "Building a working model",
"one_line": "Turning the map into equations, so the machine can be replayed and experimented on.",
"body": """
### Why a map is not enough

The map says the pump affects the alarm. It does not say *how*. Does the alarm
follow immediately or eventually, always or usually, strongly or weakly? Without
that, you cannot answer "what would happen if...", because you have no way to
work out what follows from what.

So each signal gets its own small equation, estimated from the recordings:
given what its causes were doing a moment ago, how likely is this signal to be
on right now?

### The part that makes explanation possible

Each equation also carries a term for everything it cannot explain - the
particular quirks of this particular machine on this particular day.

That term is not a nuisance to be minimised. It is the key to the whole thing.
When we later ask *"would this breakdown have been avoided?"*, we first work out
what that unexplained part was during the real breakdown, and then hold it
fixed while changing one thing. Without it we would be comparing this machine
against an average machine, and any difference we saw might just be the
difference between machines.

There is a guarantee that comes with this, and the test suite checks it on every
run: **replay a recording with its own unexplained part and change nothing, and
you get back exactly what really happened.** If that ever failed, every
explanation built on the model would be suspect.

### A real bug, and what it taught us

The first version of this model was quietly broken, in a way worth describing
because it is easy to miss.

Alarms ring about two percent of the time. A model fitted to that data learns,
quite correctly, that alarms are almost always silent. It learns it so well that
when we simulated the pump failing, the alarm **still would not ring**. The
probability came out at 0.4849 - just under the halfway mark needed to switch
it on. Every simulated fault produced a machine where nothing ever went wrong.

The fix is to keep **two** fitted versions of each equation:

- an **honest** one, giving the real probabilities, used whenever we need to
  ask how surprising something was;
- a **re-weighted** one, where rare events count as heavily as common ones,
  used whenever we need to decide whether something switches on.

After the fix, the same situation gives 0.9652 and the alarm rings. The accuracy
of predicting what a fault would do went from zero to 0.83.
""",
},

"rca": {
"title": "Finding the cause",
"one_line": "Telling apart the signal that caused the fault from the signals merely reacting to it.",
"body": """
### The problem, in one sentence

When a machine breaks, everything looks wrong at once.

The pump alarm rings, the pump switches itself off, the tank level drops, a
second alarm rings. Five signals are abnormal. Four of them are consequences.
Send an engineer to the loudest one and they fix a symptom while the real
problem carries on.

Ranking by "how abnormal does this look" cannot solve this, because the symptom
often looks *more* abnormal than the cause. An alarm going from silent to
ringing is a dramatic change. A pump health flag quietly flipping is not.

### The idea that does solve it

> **A symptom is explained by its own causes. A root cause is not.**

Think about what the model expects. When the pump has failed, the alarm ringing
is completely unsurprising - the model knew it would ring, because the pump had
failed. The alarm is *accounted for*.

But the pump failing is accounted for by nothing. Nothing upstream of it
changed. As far as the model is concerned it came out of nowhere.

That is the signature of something external acting on the machine - which is
exactly what a fault is. So we score every signal by how badly its own causes
explain its behaviour during the fault, and the ones at the top are the ones
something happened to.

This follows the counterfactual attribution approach of Budhathoki and
colleagues (2022), which scores root causes by how far each signal's unexplained
part departed from normal.

### Why it suits this data so well

causRCA's faults were created by *forcing* a signal to a value. A forced signal
is, by definition, one whose behaviour its own causes cannot account for. The
method is looking for precisely the fingerprint that the fault injection leaves
behind.

### How well it works

The correct cause is ranked **first in 92% of the 25 faults**. Averaged over
all of them, the first correct answer appears at position 1.04.

The method also knows what to ignore: alarms are never offered as causes, since
an alarm reports a fault and cannot create one.
""",
},

"intervention": {
"title": "Predicting what an action would do",
"one_line": "Forcing a signal to a value in the model, and checking the prediction against faults that really happened.",
"body": """
### The question

*If the pump fails, what else will go wrong?*

This is a forward-looking question, and answering it needs more than
correlation. Observing that the pump is healthy tells you the machine is fine.
*Forcing* the pump to be healthy is a maintenance action, and only a causal
model can tell you what that action would achieve.

The distinction has a name - the **do-operation**. Seeing a value and setting a
value are different things, and treating them the same is the mistake causal
inference exists to prevent.

### How it is computed

We take a healthy recording, override the chosen signal in the model, and let
everything downstream react. Each signal is recalculated in dependency order so
that nothing is computed before the things it depends on.

### Why this page is the strongest evidence in the whole system

Every fault in this dataset was created by forcing a known signal to a known
value, and the machine's real response was recorded.

So we can do something almost nobody can do: **force the same signal in our
model, and compare the alarms we predict against the alarms the real machine
actually raised.**

Across the fault scenarios, every alarm the model predicted really did fire,
and it caught three quarters of the alarms that occurred - an F1 of 0.83.

That turns "here is a prediction" into "here is a prediction that was checked",
which is the difference between a demonstration and a result.
""",
},

"counterfactual": {
"title": "What would have prevented it",
"one_line": "Rewinding one specific breakdown, changing one thing, and replaying it.",
"body": """
### A different question from the last page

The previous page asked a general question: if a pump fails, what tends to
happen? This page asks the question an engineer standing in front of a broken
machine actually asks:

> *"This machine broke. Would it have broken if the pump had stayed healthy?"*

That is about **this** machine and **this** breakdown, with all its particular
circumstances intact - not about machines in general.

### The three steps

This is the standard procedure for counterfactual reasoning, and the code
follows it literally.

**Step one - work out what was unusual about this run.** For every signal at
every moment we compare what the model expected against what actually happened.
The difference is everything about this specific breakdown the model cannot
explain.

**Step two - change one thing.** Hold the chosen signal at a different value
from a chosen moment onward.

**Step three - replay, keeping everything else exactly as it was.** Including
all that unexplained particularity from step one.

Step three is what makes it a counterfactual rather than a comparison. Without
holding the unexplained part fixed, you would be comparing this broken machine
against a typical machine, and you could not tell whether a difference came from
your change or simply from the machines being different.

### Reading the answer honestly

Most faults in this dataset were created by forcing **two or three** signals at
once - a coolant leak trips both the pump and the tank level. So fixing one
cause usually silences one alarm and leaves the others ringing.

That is the correct answer, not a failure, and the dashboard reports it that way
rather than as a yes or no. Fixing the top suspect silenced at least one alarm
in **23 of 25** faults.

### The honest test of the machinery itself

There is one more check worth knowing about. If we fix **exactly the signals
that were really tampered with** - using the dataset's own answer key - the
breakdown should disappear completely.

It does, in **19 of 25** faults. The six failures are all the same scenario, and
they all trace back to that one missing link on the map between the tank level
and its overflow alarm. The reasoning is sound; the map has a hole.

That is a useful thing to be able to say. It means the fix is to improve causal
discovery, not the explanation machinery.
""",
},
}


# =============================================================================
# Settings, what they mean, and what they were measured to do
# =============================================================================

SETTINGS_GUIDE = [
{
"name": "SUBSYSTEM",
"plain": "Which part of the machine to analyse",
"detail": "The lathe is split into a coolant system, a hydraulics system and a "
          "measuring probe, each with its own hand-drawn map. The coolant system "
          "is the smallest and cleanest, which makes it the right place to start.",
"range": "coolant, hydraulics, probe, full",
"effect": "The full machine has 92 signals instead of 15, which makes causal "
          "discovery far slower and the map much harder to read.",
},
{
"name": "NORMAL_RECORDINGS",
"plain": "How many healthy recordings to include",
"detail": "Healthy recordings are needed even though nothing goes wrong in them. "
          "Without them the system has no example of what normal looks like, and "
          "alarms would appear to be permanently ringing.",
"range": "10 to 170",
"effect": "More healthy data improved accuracy: going from 20 to 60 recordings "
          "lifted F1 from 0.50 to 0.63 at the same settings.",
},
{
"name": "RESAMPLE_SECONDS",
"plain": "How many seconds one row of the table covers",
"detail": "The most important setting here. It decides how far back in time the "
          "method can look for a cause, because the search window is this value "
          "multiplied by the number of steps.",
"range": "0.5 to 10 seconds",
"effect": "Measured: 1 second scores 0.50, 2.5 seconds scores 0.71, 5 seconds "
          "scores 0.56. Too fine and the window cannot reach the five second "
          "alarm rules; too coarse and genuine detail is lost.",
},
{
"name": "MAX_LAG",
"plain": "How many steps back a cause is allowed to act",
"detail": "Combined with the row length, this sets the search window. At 2.5 "
          "seconds per row, 3 steps reaches 7.5 seconds back - comfortably past "
          "the machine's five second alarm rules.",
"range": "1 to 10",
"effect": "Measured: 3 steps scores 0.71, 2 steps scores 0.67, 1 step scores "
          "0.50 because it cannot reach the delayed rules at all. Going to 4 "
          "made it worse and took 352 seconds.",
},
{
"name": "SIGNIFICANCE",
"plain": "How much evidence is needed before a link is kept",
"detail": "Smaller numbers mean stricter. This matters more here than it usually "
          "does, because a map with extra links makes every later stage slower "
          "and less trustworthy.",
"range": "0.001 to 0.05",
"effect": "Measured: at 0.001 the system finds 7 links of which 6 are real. At "
          "0.05 it finds 15, of which still only 6 are real - the eight extra "
          "links are noise.",
},
{
"name": "INDEPENDENCE_TEST",
"plain": "Which statistical test decides whether two signals are related",
"detail": "Must match the kind of data. Every coolant signal is true/false, so "
          "we use the G-test, which compares how often combinations occur "
          "against how often they would if the signals were unrelated.",
"range": "discrete (G-test), continuous (partial correlation)",
"effect": "Using the continuous test on true/false data asks the wrong question "
          "and quietly produces a worse map.",
},
{
"name": "MULTIPLE_TEST_CORRECTION",
"plain": "Correcting for having asked hundreds of questions",
"detail": "With 15 signals and four lags there are over 900 possible links. "
          "Testing all of them at a one-in-a-hundred threshold would produce "
          "about nine links by pure chance.",
"range": "fdr_bh, none",
"effect": "Without it, roughly a fifth of the map could be noise.",
},
{
"name": "MAX_CONDITIONS / MAX_COMBINATIONS",
"plain": "How hard the method searches for an alternative explanation",
"detail": "When testing whether A causes B, the method checks whether some other "
          "signal explains the link instead. These two cap how many other "
          "signals it considers at once, and how many combinations it tries.",
"range": "1 to 5 conditions, 1 to 10 combinations",
"effect": "Necessary rather than optional: without a cap the search is "
          "combinatorial and may never finish.",
},
{
"name": "TOP_K_ROOT_CAUSES",
"plain": "How many suspects to report per fault",
"detail": "Most faults here were caused by two or three signals at once, so "
          "reporting only one would always be incomplete.",
"range": "1 to 10",
"effect": "Affects how the results are reported, not how they are computed.",
},
]


# =============================================================================
# How far the results can be pushed
# =============================================================================

HONESTY_NOTES = [
("Everything on this dashboard was checked against something recorded",
 "The causal map is scored against the engineers' own map. The root causes are "
 "scored against the signals the test rig really tampered with. The predictions "
 "are compared against faults that really happened. None of it rests on the "
 "system's own opinion of itself.", "good"),

("The faults were simulated, the machine was real",
 "Faults were injected into a digital copy of the lathe, with the real "
 "controller responding to them. So the machine's reactions are genuine, but "
 "the faults themselves were created rather than encountered in service.",
 "warn"),

("Four of the ten real links were not found",
 "Mostly links between one signal and another, rather than signal to alarm. "
 "Those are harder because the signals involved change in only a handful of "
 "recordings. This is the clearest thing to improve next.", "warn"),

("Results are for the coolant system only",
 "The hydraulics and probe systems have their own maps and faults and have not "
 "been analysed yet. Nothing here should be assumed to carry over to them "
 "without checking.", "warn"),

("Where this goes next",
 "Every part of this pipeline is written to work on any subsystem, and on other "
 "datasets. The obvious next steps are the remaining two subsystems, then the "
 "full 92-signal machine, then comparing alternative methods at each stage.",
 "good"),
]
