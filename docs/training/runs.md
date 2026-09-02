# `runs.py`

**Source:** [hunger_games/training/runs.py](../../hunger_games/training/runs.py)
**Depends on:** `json`, `pathlib`, `dataclasses.asdict` (standard library); [hunger_games/research/experiments.py](../research/experiments.md) (`make_run_dir`); [hunger_games/research/plots.py](../research/plots.md) (`training_run_plots`); duck-typed on [training/genetic.py](genetic.md) (`GeneticTrainer`), [training/reinforce.py](reinforce.md) (`ReinforceTrainer`) and [training/imitation.py](imitation.md) (`ImitationTrainer`)
**Used by:** [training/__init__.py](init.md) (re-exports `save_run`); [hunger_games/ui/session.py](../ui/session.md) (`Session.save_training_run`); [experiments/run_ga.py](../experiments/run_ga.md); [experiments/run_rl.py](../experiments/run_rl.md); `tests/test_research.py`; `tests/test_imitation.py`

## Purpose

Every trainer produces a history and behaviour telemetry. `save_run` turns that into a results folder with the same layout a research sweep uses, so an imitation run, a GA run, an RL run and a parameter sweep can sit side by side under `results/` and be read by the same tools. One function, one folder, one PNG per chart.

## Concepts you need

**Run folder.** A timestamped directory, `results/<name>_<YYYYmmdd_HHMMSS>/`, made by `make_run_dir`. The timestamp means two runs with the same name never overwrite each other.

**Duck typing.** `save_run` does not check the trainer's class. It relies on the shape every trainer shares: a `settings` property that returns the trainer's own settings dataclass, plus `config`, `history`, `history_rows()`, `champion` and `save_champion`. Any object with those works.

**History rows versus telemetry.** `history_rows()` gives small JSON rows (numbers only). Telemetry summaries are large dictionaries with heatmaps and are kept only in memory, in each stats object's `telemetry` field, and drawn straight into charts.

## Walkthrough

### Imports

`json`, `Path`, then `make_run_dir` and `training_run_plots` from the research package. `asdict` is imported inside the function. Importing `research.plots` pulls in matplotlib, which is why this module is the heaviest import in the training package.

### `save_run(trainer, method, name, results_dir="results")`

```python
def save_run(trainer, method: str, name: str, results_dir: str | Path = "results") -> Path:
```

Writes config, history, champion and plots for a `GeneticTrainer`, `ReinforceTrainer` or `ImitationTrainer`. Returns the run folder.

| Parameter | Meaning |
| --- | --- |
| `trainer` | A trainer that has run at least one step |
| `method` | `"imitation"`, `"genetic"` or `"reinforce"`. Recorded in `config.json` and chooses the chart set |
| `name` | Prefix of the folder name |
| `results_dir` | Where run folders go; created if missing |

Steps, in order:

1. `folder = make_run_dir(results_dir, name)`. This creates `<results_dir>/<name>_<timestamp>/plots/`.
2. `trainer_config = asdict(trainer.settings)`. The `settings` property returns `TrainingConfig`, `RLConfig` or `ImitationConfig`, whichever the trainer holds.
3. Write `config.json`: `{"method": method, "simulation": trainer.config.to_dict(), "trainer": trainer_config}` with `indent=2` and `default=str` (so tuples and any odd values still serialise).
4. Write `history.json`: `trainer.history_rows()`, one row per generation or epoch.
5. Write `champion.json` through `trainer.save_champion`, if `trainer.champion is not None`. Every trainer has `save_champion` and every trainer writes the same file shape.
6. Collect `summaries = [stats.telemetry for stats in trainer.history if stats.telemetry]`, one per step that recorded behaviour.
7. `training_run_plots(rows, summaries, folder / "plots", method)`.

Example:

```python
from hunger_games.training import save_run
folder = save_run(trainer, "genetic", "voting_baseline")
print(folder)   # results/voting_baseline_20260902_143015
```

Design reasoning: the config is saved first so a crash while plotting still leaves the settings on disk. The champion is written in the trainers' own file format, so the dashboard can load it back unchanged. Using `settings` and `save_champion` for every trainer means this function has no per-trainer branches; only the plots need to know the method.

### The run folder layout

```
results/<name>_<timestamp>/
    config.json      {"method", "simulation": SimulationConfig.to_dict(), "trainer": ImitationConfig, TrainingConfig or RLConfig}
    history.json     list of ImitationStats, GenerationStats or EpochStats .to_row()
    champion.json    trainer.save_champion output
    plots/           one PNG per chart, plus one GIF
```

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

**Charts for `method == "reinforce"` (and any other value):**

| File | Shows |
| --- | --- |
| `reward.png` | training and validation return by epoch |
| `losses.png` | policy loss and value loss |
| `entropy.png` | policy entropy in nats |
| `survival.png` | training and validation survival ticks |
| `win_kill_rate.png` | training win rate, validation win rate, kills per game |
| `reward.gif` | the return curves growing frame by frame |
| `timing.png` | seconds per epoch and cumulative seconds |

**Charts written for any method when at least one step has telemetry:**

| File | Shows |
| --- | --- |
| `action_distribution_over_training.png` | stacked area of action shares per step |
| `death_needs_over_training.png` | mean thirst, hunger and health at death per step |
| `behaviour_over_training.png` | entropy, survival, win and kill rate per step |
| `action_distribution.png`, `actions_by_thirst.png`, `actions_by_hunger.png`, `actions_by_health.png`, `instinct_curves.png`, `consumption_timing.png`, `fight_or_flight.png`, `proximity_vs_remaining.png`, `actions_by_remaining.png`, `position_heatmap.png`, `armed_vs_unarmed_heatmaps.png`, `deaths_by_cause.png` | the latest step's behaviour in detail |

The GA records telemetry only when `TrainingConfig.collect_telemetry` is true. The RL trainer always records it for the learners in training games. The imitation trainer records it for the student in validation games, so it needs `validation_games >= 1`.

## How to use it / experiment

**From a script.** Both `experiments/run_ga.py` and `experiments/run_rl.py` end with `save_run(trainer, method, args.name, args.results)`.

**From the dashboard.** The Train tab's Save run button calls `Session.save_training_run(name, results_dir)`, which wraps this function and puts the folder path in the status bar.

**Save mid-run.** `save_run` reads whatever is in `trainer.history` at the moment. Calling it every 10 generations from an `on_generation` callback gives checkpoints:

```python
def checkpoint(stats):
    if stats.generation % 10 == 9:
        save_run(trainer, "genetic", f"ga_gen{stats.generation + 1}")
trainer.run(on_generation=checkpoint)
```

**Save the whole flow.** Save the student and the warm-started trainer as two folders, and compare their `survival.png` charts:

```python
save_run(student, "imitation", "student")
save_run(rl, "reinforce", "rl_from_student")
```

**Read a run back.** `json.load(open(folder / "history.json"))` gives a list of dicts ready for `pandas.DataFrame`. `GeneticTrainer.load_champion(folder / "champion.json")` restores the genome and its `NeuralConfig` for any method.

**Compare two runs.** Load both `history.json` files and plot `val_accuracy` (imitation), `val_fitness` (GA) or `val_return` (RL) on one axis. Those columns are the ones measured on fixed seeds or held-out data.

## Gotchas

- A `GeneticTrainer` that has not run yet has `champion is None`, so no `champion.json` is written, and `training_run_plots` writes nothing for an empty history. You get a folder with `config.json`, an empty `history.json` list and an empty `plots/`. The other two trainers always have a champion (the current network), so they always write one.
- `method` must match the trainer. The charts read specific row keys, so `save_run(ga_trainer, "reinforce", ...)` raises `KeyError` inside `training_run_plots` (no `train_return` in a GA row). Any `method` other than `"imitation"` and `"genetic"` gets the REINFORCE chart set.
- `history.json` never contains genomes or telemetry. The champion genome is in `champion.json`; per-step genomes and telemetry exist only in memory.
- The whole plotting stack runs in the calling thread. With hundreds of steps, the GIF can take a while, because each frame is a separate matplotlib figure.
- `default=str` in the JSON dumps means an unexpected value type is written as its string form rather than raising. If a field looks like `"(64, 32)"` instead of `[64, 32]`, that is why.
- Two calls in the same second with the same `name` land in the same folder and overwrite each other's files.
