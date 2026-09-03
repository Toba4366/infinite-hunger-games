# `runs.py`

**Source:** [hunger_games/training/runs.py](../../hunger_games/training/runs.py)
**Depends on:** `json`, `pathlib`, `dataclasses.asdict` (standard library); [hunger_games/research/experiments.py](../research/experiments.md) (`make_run_dir`); [hunger_games/research/plots.py](../research/plots.md) (`learning_curve_plots`, `training_run_plots`); duck-typed on every trainer: [training/genetic.py](genetic.md), [training/imitation.py](imitation.md), [training/neat.py](neat.md), [training/reinforce.py](reinforce.md), [training/ppo.py](ppo.md)
**Used by:** [training/__init__.py](init.md) (re-exports `save_run`); [hunger_games/ui/session.py](../ui/session.md) (`Session.save_training_run`); [research/comparison.py](../research/comparison.md) (`MethodComparison.train_all` saves every variant under `<run>/runs/`); [experiments/run_ga.py](../experiments/run_ga.md); [experiments/run_rl.py](../experiments/run_rl.md); `tests/test_methods.py`; `tests/test_research.py`; `tests/test_imitation.py`

## Purpose

Every trainer produces a history, a shared learning history, an event log and behaviour telemetry. `save_run` turns that into a results folder with the same layout a research sweep uses, so an imitation run, a GA run, a NEAT run, an RL run and a parameter sweep can sit side by side under `results/` and be read by the same tools. One function, one folder, one PNG per chart.

Since the five methods share `IterationStats`, every run folder also gets `learning.json` and the same set of learning-curve charts, whatever the method. That is what makes runs of different methods comparable.

## Concepts you need

**Run folder.** A timestamped directory, `results/<name>_<YYYYmmdd_HHMMSS>/`, made by `make_run_dir`. The timestamp means two runs with the same name never overwrite each other.

**Duck typing.** `save_run` does not check the trainer's class. It relies on the shape every trainer shares: a `settings` property that returns the trainer's own settings dataclass, plus `config`, `history`, `history_rows()`, `champion` and `save_champion`. It reads `learning_history` and `events` with `getattr` and `hasattr`, so an object without them still saves.

**Two histories.** `history_rows()` gives the trainer's own rows (`GenerationStats`, `EpochStats`, `ImitationStats`, or `IterationStats` for NEAT). `learning_history` gives the shared `IterationStats` rows. The first feeds the method's own charts, the second the shared curves.

**Telemetry.** Summaries are large dictionaries with heatmaps. They are kept only in memory, in each stats object's `telemetry` field, and drawn straight into charts.

## Walkthrough

### Imports

`json`, `Path`, then `make_run_dir`, `learning_curve_plots` and `training_run_plots` from the research package. `asdict` is imported inside the function. Importing `research.plots` pulls in matplotlib, which is why this module is the heaviest import in the training package.

### `save_run(trainer, method, name, results_dir="results")`

```python
def save_run(trainer, method: str, name: str, results_dir: str | Path = "results") -> Path:
```

Writes config, history, champion and plots. Returns the run folder.

| Parameter | Meaning |
| --- | --- |
| `trainer` | A trainer that has run at least one step |
| `method` | `"imitation"`, `"genetic"`, `"neat"`, `"reinforce"` or `"ppo"`. Recorded in `config.json` and chooses the method's own chart set |
| `name` | Prefix of the folder name |
| `results_dir` | Where run folders go; created if missing |

Steps, in order:

1. `folder = make_run_dir(results_dir, name)`. This creates `<results_dir>/<name>_<timestamp>/plots/`.
2. `trainer_config = asdict(trainer.settings)`. The `settings` property returns `TrainingConfig`, `RLConfig`, `PPOConfig`, `ImitationConfig` or `NeatTrainerConfig`, whichever the trainer holds. A `NeatTrainerConfig` nests its `NeatConfig`, and `asdict` converts both.
3. Write `config.json`: `{"method": method, "simulation": trainer.config.to_dict(), "trainer": trainer_config}` with `indent=2` and `default=str` (so tuples and any odd values still serialise).
4. Write `history.json`: `trainer.history_rows()`, one row per generation or epoch.
5. Write `learning.json`: `[stats.to_row() for stats in trainer.learning_history]`, the shared rows. An object without `learning_history` writes an empty list.
6. Write `events.txt`: `trainer.events.events` joined by newlines, or an empty file if there is no `events`.
7. Write `champion.json` through `trainer.save_champion`, if `trainer.champion is not None`.
8. Collect `summaries = [stats.telemetry for stats in trainer.history if stats.telemetry]`, one per step that recorded behaviour.
9. `training_run_plots(rows, summaries, folder / "plots", method)`, the method's own charts.
10. `learning_curve_plots(learning, folder / "plots")`, the shared charts.

Example:

```python
from hunger_games.training import save_run
folder = save_run(trainer, "ppo", "ppo_from_student")
print(folder)   # results/ppo_from_student_20260902_143015
```

Design reasoning: the config is saved first so a crash while plotting still leaves the settings on disk. The champion is written in the trainers' own file format, so the dashboard can load it back unchanged. Using `settings` and `save_champion` for every trainer means this function has no per-trainer branches; only the plots need to know the method.

### The run folder layout

```
results/<name>_<timestamp>/
    config.json      {"method", "simulation": SimulationConfig.to_dict(), "trainer": the settings dataclass}
    history.json     list of the trainer's own .to_row() rows
    learning.json    list of IterationStats.to_row() rows (the same shape for every method)
    events.txt       one EventLog line per row
    champion.json    trainer.save_champion output
    plots/           one PNG per chart, plus GIFs
```

**Shared charts, written for every method from `learning.json`:**

| File | Shows |
| --- | --- |
| `score.png` | mean, best and validation score per iteration |
| `entropy_shared.png` | policy entropy in nats |
| `game_length.png` | mean learner survival in ticks |
| `win_rate_shared.png` | win rate |
| `score_vs_time.png` | mean score against cumulative seconds |
| `curriculum.png` | opponents per iteration (flat without a curriculum) |
| `score.gif` | mean and validation score growing frame by frame |

**Charts for `method == "imitation"`:**

| File | Shows |
| --- | --- |
| `losses.png` | training and validation cross-entropy by epoch |
| `accuracy.png` | training and validation accuracy (how often the student picks the teacher's action) |
| `survival.png` | validation survival ticks |
| `win_rate.png` | validation win rate |
| `losses.gif` | the two loss curves growing frame by frame |
| `timing.png` | seconds per epoch (bars) and cumulative seconds (line) |

**Charts for `method == "genetic"`:**

| File | Shows |
| --- | --- |
| `fitness.png` | best, mean and validation fitness by generation |
| `fitness.gif` | the best and validation curves growing frame by frame |
| `timing.png` | seconds per generation (bars) and cumulative seconds (line) |

**Charts for `method == "neat"`:**

| File | Shows |
| --- | --- |
| `neat_structure.png` | species count and the champion's hidden node count by iteration |
| `fitness.png` | best, mean and validation score by iteration |
| `timing.png` | seconds per iteration and cumulative seconds |

**Charts for `method == "reinforce"`, `"ppo"` and any other value:**

| File | Shows |
| --- | --- |
| `reward.png` | training and validation return by epoch |
| `losses.png` | policy loss and value loss |
| `entropy.png` | policy entropy in nats |
| `survival.png` | training and validation survival ticks |
| `win_kill_rate.png` | training win rate, validation win rate, kills per game |
| `reward.gif` | the return curves growing frame by frame |
| `timing.png` | seconds per epoch and cumulative seconds |

**Behaviour charts, written for any method when at least one step has telemetry:**

| File | Shows |
| --- | --- |
| `action_distribution_over_training.png` | stacked area of action shares per step |
| `death_needs_over_training.png` | mean thirst, hunger and health at death per step |
| `behaviour_over_training.png` | survival, win rate, kill rate and entropy per step |
| `action_distribution.png`, `actions_by_thirst.png`, `actions_by_hunger.png`, `actions_by_health.png`, `instinct_curves.png`, `consumption_timing.png`, `fight_or_flight.png`, `proximity_vs_remaining.png`, `actions_by_remaining.png`, `position_heatmap.png`, `armed_vs_unarmed_heatmaps.png`, `deaths_by_cause.png` | the latest step's behaviour in detail |

The GA in `"self"` mode records telemetry only when `TrainingConfig.collect_telemetry` is true; in `"voting"` mode it always does, for the learners. The RL trainers always record it for the learners in training games. NEAT records it for the learners in evaluation games. The imitation trainer records it for the student in validation games, so it needs `validation_games >= 1`.

## How to use it / experiment

**From a script.** Both `experiments/run_ga.py` and `experiments/run_rl.py` end with `save_run(trainer, method, args.name, args.results)`. `experiments/run_comparison.py` saves every variant under `<comparison folder>/runs/`.

**From the dashboard.** The Train tab's Save run button calls `Session.save_training_run(name, results_dir)`, which wraps this function and puts the folder path in the status bar.

**Save mid-run.** `save_run` reads whatever is in the histories at the moment. Calling it every 10 iterations from a callback gives checkpoints:

```python
def checkpoint(stats):
    if stats.epoch % 10 == 9:
        save_run(trainer, "ppo", f"ppo_epoch{stats.epoch + 1}")
trainer.run(on_epoch=checkpoint)
```

**Compare methods from their folders.** Every `learning.json` has the same keys, so:

```python
import json
from hunger_games.research import plots

runs = {"reinforce": "results/rl_20260902_150000", "ppo": "results/ppo_20260902_151000"}
series = {}
for name, folder in runs.items():
    rows = json.load(open(f"{folder}/learning.json"))
    series[name] = ([r["iteration"] for r in rows], [r["val_score"] for r in rows])
plots.overlay_curves(series, "Validation score", "iteration", "score", "paper/val_by_method.png")
```

**Read the events.** `events.txt` is plain text; `grep curriculum results/*/events.txt` lists every promotion with its timestamp.

**Read a run back.** `json.load(open(folder / "history.json"))` gives a list of dicts ready for `pandas.DataFrame`. `GeneticTrainer.load_champion(folder / "champion.json")` restores the genome (an array, or a NEAT dictionary) and, when present, its `NeuralConfig`.

## Gotchas

- A `GeneticTrainer` or `NeatTrainer` that has not run yet has `champion is None`, so no `champion.json` is written, and both plot functions write nothing for empty histories. You get a folder with `config.json`, empty `history.json` and `learning.json` lists, an empty `events.txt` and an empty `plots/`. The other three trainers always have a champion (the current network), so they always write one.
- `method` must match the trainer. The method's own charts read specific row keys, so `save_run(ga_trainer, "reinforce", ...)` raises `KeyError` inside `training_run_plots` (no `train_return` in a GA row). Only `"imitation"`, `"genetic"` and `"neat"` are matched exactly; every other string gets the REINFORCE chart set, which is right for `"ppo"`.
- A PPO run's `champion.json` says `"method": "reinforce"`, because `save_policy` is inherited. `config.json` has the right method.
- For NEAT, `history.json` and `learning.json` hold the same rows, because `NeatTrainer.history` is its `learning_history`.
- Neither JSON file contains genomes, telemetry or recordings. The champion genome is in `champion.json`; per-step genomes and telemetry exist only in memory.
- The whole plotting stack runs in the calling thread. With hundreds of steps the two GIFs (`score.gif` plus the method's own) can take a while, because each frame is a separate matplotlib draw.
- `default=str` in the JSON dumps means an unexpected value type is written as its string form rather than raising. If a field looks like `"(64, 32)"` instead of `[64, 32]`, that is why.
- Two calls in the same second with the same `name` land in the same folder and overwrite each other's files.
- `events.txt` holds at most `EventLog.capacity` lines (500 by default). A long run loses its earliest events.
