# `__init__.py`

**Source:** [hunger_games/research/__init__.py](../../hunger_games/research/__init__.py)
**Depends on:** [research/telemetry.py](telemetry.md) (`BehaviorTelemetry`)
**Used by:** anything that writes `from hunger_games.research import BehaviorTelemetry`; the submodules are imported directly by [../runner.md](../runner.md), [../training/genetic.md](../training/genetic.md), [../training/reinforce.md](../training/reinforce.md), [../training/runs.md](../training/runs.md), [../ui/session.md](../ui/session.md), [../ui/app.md](../ui/app.md) and `tests/test_research.py`

## Purpose

This file is the front door of the `research` package. The package exists to answer one question: is a brain learning the right behaviours, not just winning more often? It has three files.

| File | Job | Page |
| --- | --- | --- |
| `telemetry.py` | Counts what tributes do against how thirsty, hungry, hurt and threatened they were, where they stood and how many tributes remained | [telemetry.md](telemetry.md) |
| `plots.py` | Draws one PNG per chart from those counts, from the CSV tables, or from a trainer's history | [plots.md](plots.md) |
| `experiments.py` | Runs parameter sweeps and writes timestamped run folders | [experiments.md](experiments.md) |

The `__init__.py` itself does very little. It imports `BehaviorTelemetry` so the most used class is one import away, and it declares `__all__`.

## Concepts you need

**A package.** A folder with an `__init__.py` file is a package. Python runs `__init__.py` when you import the folder name. Whatever that file imports or defines becomes available as `hunger_games.research.<name>`.

**Re-exporting.** `from hunger_games.research.telemetry import BehaviorTelemetry` inside `__init__.py` means a user can write the shorter `from hunger_games.research import BehaviorTelemetry`. The class still lives in `telemetry.py`. Only the name is copied.

**`__all__`.** A list of strings naming what `from hunger_games.research import *` should bring in. It is also a hint to readers: these are the names the package considers public.

**Why plots and experiments are not re-exported.** `plots.py` imports matplotlib and forces the file-only `Agg` backend the moment it is imported. `experiments.py` imports pandas and the batch runner. Keeping them out of `__init__.py` means importing the telemetry stays cheap and never touches matplotlib. Code that wants them imports the submodule directly: `from hunger_games.research import plots`.

**Hooks.** The telemetry does not change the game. It listens. `Game` keeps two lists, `decision_hooks` and `tick_hooks` (see [../game.md](../game.md)), and the telemetry appends one function to each. That is the whole integration.

## Walkthrough

### The module docstring

```python
"""research - measuring what tributes do, not just who wins.
    telemetry.py    counts actions against needs, health, danger and position while a game runs
    plots.py        one PNG per chart: the chapter 3 charts, heatmaps, behaviour and training curves
    experiments.py  parameter sweeps that write timestamped run folders
...
"""
```

A short map of the package. The last paragraph says who uses it: the trainers log behaviour every generation or epoch, and the dashboard's Research tab draws the plots.

### `from hunger_games.research.telemetry import BehaviorTelemetry`

**What it does.** Loads `telemetry.py` and copies the `BehaviorTelemetry` name into the package namespace.

**Why.** The collector is the one class almost every caller needs. The runner, both trainers and the dashboard session all create one per game.

**Side effect.** Importing `telemetry.py` imports `hunger_games.game`, which imports the arena, brains, players and so on. That is fine, because anyone using telemetry is about to run a game anyway.

### `__all__ = ["BehaviorTelemetry"]`

**Signature.** A plain list of one string.

**What it does.** Declares the public name. Star imports honour it. Documentation tools read it.

**Design reasoning.** Only the telemetry collector is listed. The plot functions and the sweep classes are reached through their submodules on purpose, as explained above.

## How the pieces fit together

The data flows in one direction. Nothing in `research` writes back into the simulation.

```
Game.step()
   |  decision_hooks -> BehaviorTelemetry.on_decision(player, perception, action)
   |  tick_hooks     -> BehaviorTelemetry.on_tick(game)  ->  on_game_end(game) when over
   v
BehaviorTelemetry.summary()   a dict of plain lists and numbers
   |
   |  BehaviorTelemetry.merge([...])   add up many games or workers
   v
plots.behaviour_plots(summary, folder)       one PNG per behaviour chart
plots.training_run_plots(rows, summaries, folder, method)   curves plus behaviour
experiments.Sweep(...).run()                 results.csv, summary.json, plots/
```

Where each caller plugs in:

| Caller | What it creates | Where the summary goes |
| --- | --- | --- |
| `runner.run_single_game` | One collector per game when `collect_telemetry=True` | `GameResult.telemetry`, merged into `Runner.telemetry_summary`, written as `telemetry.json` |
| `training/genetic.play_evaluation_game` | One collector per evaluation game | Merged per generation into `GenerationStats.telemetry` |
| `training/reinforce.play_rl_episode` | One collector per episode, tracking learners only | Merged per epoch into `EpochStats.telemetry` |
| `ui/session.py` | One collector per watched game | Merged across watched games for the dashboard charts and the export button |
| `experiments.Sweep` | Through `Runner` | One merged summary per swept value in `summary.json` |

## What each submodule exports

You will import these names directly from the submodules.

| Module | Public names | One line |
| --- | --- | --- |
| `telemetry` | `BehaviorTelemetry`, `bin_index`, `ACTION_NAMES`, `ACTION_INDEX`, `NEED_BIN_EDGES`, `NEED_BIN_LABELS`, `ALIVE_BIN_EDGES`, `ALIVE_BIN_LABELS`, `HEATMAP_CELLS`, `ARMED_THRESHOLD` | The collector and the bin definitions every chart shares |
| `plots` | `ACTION_COLORS`, eight chapter 3 charts, `heatmap`, `armed_vs_unarmed`, eight behaviour charts, `curves`, `stacked_area_over_training`, `death_needs_over_training`, `behaviour_metrics_over_training`, `timing`, `curve_gif`, and the bundles `training_run_plots`, `behaviour_plots`, `batch_plots` | One function per PNG |
| `experiments` | `make_run_dir`, `set_field`, `batch_metrics`, `SweepConfig`, `Sweep` | Sweeps and run folders |

## Words used across the package

| Word | Meaning here |
| --- | --- |
| Tick | One simulation step. Every living tribute senses, decides and acts once per tick. 24 ticks make a day by default |
| Decision | One call to a brain, with the `Perception` it saw and the `Action` it returned |
| Summary | The dictionary `BehaviorTelemetry.summary()` returns: plain lists and numbers, JSON-ready |
| Step | One generation (GA) or one epoch (RL). A training run keeps one history row and one summary per step |
| Run folder | `results/<name>_<timestamp>/` with `config.json`, `plots/` and the method's own files |
| Tracked | A tribute whose decisions and positions are counted. Everyone by default, learners only in RL |
| Alive fraction | Living tributes divided by the starting roster, from 1.0 down to 1/24 |

## Reading a run folder

Both trainers and the sweep write the same shape of folder, made by `experiments.make_run_dir`.

```
results/rl_20260902_150000/
    config.json      {"method": "reinforce", "simulation": {...}, "trainer": {...}}
    history.json     one row per epoch: policy_loss, value_loss, entropy, train_return, val_return, ...
    champion.json    the best policy in the champion file format (loadable from the dashboard)
    plots/           reward.png, losses.png, entropy.png, survival.png, win_kill_rate.png,
                     timing.png, reward.gif, action_distribution_over_training.png,
                     death_needs_over_training.png, behaviour_over_training.png,
                     and the twelve behaviour charts for the last epoch
```

A GA folder swaps `reward.png` and the loss charts for `fitness.png` and `fitness.gif`, and its rows carry `best_fitness`, `mean_fitness`, `worst_fitness` and `val_fitness`. A sweep folder has `results.csv` and `summary.json` instead of a history and a champion. The full layouts are in [experiments.md](experiments.md) and [../training/runs.md](../training/runs.md).

## How to use it / experiment

The shortest possible measurement of one game:

```python
from hunger_games.config import SimulationConfig
from hunger_games.game import Game
from hunger_games.research import BehaviorTelemetry

game = Game(SimulationConfig(seed=1))
telemetry = BehaviorTelemetry(game.arena.width, game.arena.height).attach(game)
game.run()
summary = telemetry.summary()
print(summary["action_names"])
print(summary["action_counts"])
print(f"entropy {summary['entropy']:.2f} nats, win rate {summary['win_rate']:.2f}")
```

Turning that summary into charts:

```python
from hunger_games.research import plots

paths = plots.behaviour_plots(summary, "output/one_game")
for path in paths:
    print(path)
```

Measuring many games at once, across CPU cores, with the runner doing the merging:

```python
from hunger_games.runner import Runner

runner = Runner(SimulationConfig(seed=100), num_games=50, workers=4, collect_telemetry=True)
eliminations, players, games = runner.run()
plots.behaviour_plots(runner.telemetry_summary, "output/batch")
plots.batch_plots(eliminations, players, games, "output/batch")
```

Asking a research question with a sweep:

```python
from hunger_games.research.experiments import Sweep, SweepConfig

sweep = SweepConfig(name="water", parameter="terrain.water_threshold", values=[0.1, 0.25, 0.4], games_per_value=30)
folder = Sweep(SimulationConfig(), sweep).run()
print(folder)   # results/water_20260902_153000
```

Ideas to try:

- Measure only some tributes by passing `tracked_ids={0, 5, 10}` to `BehaviorTelemetry`. That is how the RL trainer separates the learners from the hand-coded opponents.
- Compare two brains by running the same seeded games with `brain_name="voting"` and `brain_name="neural"` and drawing `behaviour_plots` for each into separate folders.
- Read `summary.json` from a sweep folder back with `json.load` and draw any chart from [plots.md](plots.md) on any value's summary.

## Where to go next

| If you want to | Read |
| --- | --- |
| Understand one tally or one summary key | [telemetry.md](telemetry.md), the two tables in its walkthrough |
| Find the chart that answers a question | [README.md](README.md), the three question tables |
| Know the exact file names a run writes | [plots.md](plots.md), the bundles section |
| Change what a sweep measures | [experiments.md](experiments.md), `batch_metrics` and the extension example |
| See how the trainers call all this | [../training/genetic.md](../training/genetic.md) and [../training/reinforce.md](../training/reinforce.md) |
| Drive it from the window | [../ui/README.md](../ui/README.md), the Train and Research tabs |

## Gotchas

- `from hunger_games.research import plots` works even though `plots` is not in `__all__`. Python imports submodules by name. Only `import *` is limited by `__all__`.
- Importing `plots` sets matplotlib to the `Agg` backend for the whole process. Windows will not open after that. Import it only in code that writes files.
- Importing the package imports `hunger_games.game` and everything under it. There is no circular import today because `game.py` does not import `research`. Keep it that way: the hooks are plain lists of callables so `game.py` never needs to know the telemetry exists.
- `BehaviorTelemetry` needs the arena width and height for the heatmaps. Create it after the `Game`, not before, so you can read `game.arena.width` and `game.arena.height`.
- The training packages also record telemetry, so a training run folder and a sweep folder both contain behaviour charts. The difference is who was measured: the RL trainer tracks learners only, the GA and the sweeps track everyone.
