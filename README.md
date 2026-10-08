# Causal Root Cause Analysis for Industrial Machinery

Finds out **why** a machine broke, not just **that** it will.

Standard predictive maintenance gives you a warning and stops there. An
engineer still has to work out which of a dozen flashing alarms is the actual
problem and which are just consequences. This project answers that, by learning
how the machine's signals cause one another and reasoning over that map.

Built on **causRCA**: recordings from a real CNC vertical lathe, with faults
injected deliberately so the true cause of every breakdown is known.

## Why this dataset

Most causal analysis cannot be checked. There is no way to rewind a machine and
run the week again, so papers produce a causal map, produce an explanation, and
ask you to trust both.

causRCA makes checking possible. It provides:

- a causal map drawn by the engineers who built the machine, and
- 100 faults created by forcing a known signal to a known value.

So every claim this pipeline makes can be scored against something recorded.

## What it does

| Stage | What it does |
|---|---|
| 1. Load | Read the recordings and the engineers' causal map |
| 2. Preprocess | Turn change logs into regular tables |
| 3. Discover | Learn which signals affect which, using PCMCI+ |
| 4. Score | Compare the learned map against the engineers' |
| 5. Fit | Turn the map into equations that can be simulated |
| 6. Diagnose | Find which signal caused a given fault |
| 7. Predict | Force a signal and see what happens |
| 8. Explain | Say what would have prevented the fault |

## How well it works

Measured on the coolant subsystem, against the dataset's own ground truth:

| What was measured | Result |
|---|---|
| Causal links found that are real | 86% |
| Real causal links found | 60% |
| Causal map accuracy (F1) | 0.71 |
| Correct cause ranked first | 92% of faults |
| MAP@3 for root cause | 0.45 |
| Alarms correctly predicted when a fault is simulated | F1 0.83 |
| Faults prevented by fixing their real causes | 76% |

The last two rows are the unusual ones. They compare what the model predicts a
fault would do against what the fault really did, which is normally impossible.

## Getting started

Install:

```bash
pip install -r requirements.txt
```

Download the dataset (3.6 MB) into `data/causrca/raw/`:

```bash
bash scripts/download_data.sh
```

Look at the data first:

```bash
python scripts/explore_data.py
```

Run the pipeline:

```bash
python scripts/run_pipeline.py
```

Open the dashboard:

```bash
streamlit run dashboard/app.py
```

## Settings

Everything tunable lives in `settings.py`, with a comment on each value
explaining what it does and what range makes sense. There are no command-line
options.

To search for better settings automatically, scoring each against the
engineers' causal map:

```bash
python scripts/tune_settings.py
```

## Two findings worth knowing

**The resampling grid matters more than the lag window.** The machine's alarm
rules fire either immediately or after a five second delay. What the method
needs is a window covering those five seconds, and widening it by making each
step bigger is far cheaper than adding more steps. A 2.5 second grid with a lag
of 3 scores F1 0.71 in 6 seconds; a 1 second grid reaching the same distance
back takes over 240 seconds and scores worse.

**Rare events need care.** Alarms fire about two percent of the time. A model
fitted on that learns the alarm is almost always off - so firmly that it will
not switch on even when the pump it watches has failed, which makes simulating
a fault useless. The fix is to keep two fitted models per signal: honest
probabilities for measuring surprise, and re-weighted ones for deciding whether
something switches on.

## Layout

```
settings.py              every tunable value, with explanations
src/causalpdm/pipeline/
    data.py              reading causRCA
    preprocess.py        change logs into tables
    discovery.py         PCMCI+
    evaluate.py          all the scoring
    tuning.py            searching for better settings
    scm.py               equations and simulation
    rca.py               finding the cause
    explain.py           what-if and what-would-have-prevented-it
    runner.py            runs it all end to end
scripts/
    explore_data.py      a first look at the data
    run_pipeline.py      run everything
    tune_settings.py     search for better settings
dashboard/app.py         the dashboard
tests/                   the test suite
results/                 output, regenerable
```

## Tests

```bash
pytest tests/
```

Fast tests only:

```bash
pytest tests/ -m "not slow"
```

The most important test checks that replaying a recording through the model,
with its own recovered randomness and nothing changed, reproduces exactly what
really happened. If that ever fails, every explanation built on the model is
suspect.

## Dataset

Mehling, C. W., Pieper, S. and Lueke, T. (2025). *causRCA: Real-World Dataset
for Causal Discovery and Root Cause Analysis in Machinery*. Zenodo.
https://doi.org/10.5281/zenodo.15876410 (Apache-2.0)
