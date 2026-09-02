# `experiments.py`

**Source:** [hunger_games/research/experiments.py](../../hunger_games/research/experiments.py)
**Depends on:** `copy`, `json`, `dataclasses`, `datetime`, `pathlib`, `collections.abc.Callable`; `numpy`; `pandas`; [../config.md](../config.md) (`SimulationConfig`); [plots.md](plots.md); [telemetry.md](telemetry.md) (`BehaviorTelemetry.merge`); [../runner.md](../runner.md) (`Runner`); [../scenario.md](../scenario.md) (`Scenario`)
**Used by:** [../training/runs.md](../training/runs.md) (`make_run_dir`); [../ui/session.md](../ui/session.md) (`Sweep`, `SweepConfig`); [../ui/app.md](../ui/app.md) (`SweepConfig`); `experiments/run_sweep.py`; `tests/test_research.py`

## Purpose

A researcher's question is usually "what happens to X if I change Y?". `Sweep` takes one config field, a list of values and a number of games per value, plays every value on the same seeded games, and writes a folder with a results table, a JSON file that includes behaviour telemetry, and one chart per metric. `make_run_dir` gives the trainers the same timestamped folder layout, so every experiment in `results/` looks alike.

## Concepts you need

**Controlled comparison.** Every value plays game N with seed `sweep.seed + N`, because `Game` adds the game id to the base seed. The terrain, the roster and the dice are the same across values, so differences come from the parameter and not from luck.

**Dotted paths.** `SimulationConfig` nests other dataclasses: `terrain`, `noise`, `neural`, `reward`. The string `"terrain.water_threshold"` means "the `water_threshold` field of the `terrain` object". `set_field` splits on dots and walks with `getattr`, then sets the last part with `setattr`.

**Deep copy.** `copy.deepcopy` copies an object and every object inside it. Without it, changing `updated.terrain.water_threshold` would change the original config's `terrain` too, because both would share the same nested object.

**Dataclass to dict.** `dataclasses.asdict(sweep)` turns the settings into a plain dictionary so they can be saved as JSON next to the results.

**Callbacks.** `run()` accepts two optional functions. `on_value(row)` is called after each value finishes; the dashboard uses it to print rows and the command-line script prints them. `on_progress(done, total)` updates a progress bar. Passing functions in is how a long job reports without knowing who is listening.

## Walkthrough

### `make_run_dir(base: str | Path, name: str) -> Path`

Builds `base/<name>_<YYYYMMDD_HHMMSS>/`, creates it along with a `plots/` subfolder, and returns the run folder. Two runs started in the same second would share a folder; `exist_ok=True` means that does not raise, the second simply writes into the same place.

```python
folder = make_run_dir("results", "ga")   # results/ga_20260902_153000
```

Both `Sweep.run` and `training/runs.save_run` use it, which is why a training run and a sweep sit side by side in `results/` with the same naming.

### `set_field(config: SimulationConfig, dotted: str, value) -> SimulationConfig`

Returns a deep copy of `config` with one field changed. `parts[:-1]` selects nested objects, `parts[-1]` is the field that gets set. No type checking or conversion happens; the value is stored as given.

```python
wet = set_field(SimulationConfig(), "terrain.water_threshold", 0.4)
quiet = set_field(SimulationConfig(), "gamemaker_enabled", False)
```

### `batch_metrics(eliminations, players, games) -> dict`

The headline numbers for one batch of games, computed from the three runner tables. Each is guarded so an empty table gives 0.0 instead of an error.

| Key | Formula | Meaning |
| --- | --- | --- |
| `games` | `len(games)` | Games played |
| `victor_rate` | share of `games["winner_id"]` that is not null | Games that ended with a sole survivor rather than a draw or wipe-out |
| `mean_days` | mean of `games["days"]` | Average game length |
| `player_vs_player_share` | share of eliminations with method `player_vs_player` | Fights |
| `natural_share` | share with method `natural_causes` | Thirst, hunger, bleeding |
| `gamemaker_share` | share with method `gamemaker` | The shrinking circle |
| `eliminations_per_point` | mean over player rows of `kills / training_score` | Kills relative to strength |
| `mean_interventions` | mean of `games["interventions"]` | How often the game makers stepped in |
| `eliminations_per_game` | `len(eliminations) / max(1, len(games))` | Deaths per game |

The three shares come from `value_counts(normalize=True)` on the `method` column, with `shares.get(name, 0.0)` so a missing method reads as zero.

### `class SweepConfig` (dataclass)

| Field | Default | Meaning |
| --- | --- | --- |
| `name` | required | Label for the run folder; the CLI defaults it to the parameter with dots replaced by underscores |
| `parameter` | required | Config field to change, dotted for nested fields |
| `values` | required | The values to try, in order |
| `games_per_value` | `50` | Games per value |
| `workers` | `1` | CPU cores handed to `Runner` |
| `seed` | `1000` | Base seed; game N of every value uses `seed + N` |
| `telemetry` | `True` | Collect behaviour telemetry (slower, gives the behaviour charts) |
| `results_dir` | `"results"` | Where run folders go |

### `class Sweep`

#### `__init__(self, config: SimulationConfig, sweep: SweepConfig, scenario: Scenario | None = None) -> None`

Stores the base config, the sweep settings and an optional scenario (the dashboard passes its painted map so every value plays on the same terrain). Starts with empty `rows` and `summaries`, `run_dir = None`, and the stop flag down.

#### `run(self, on_value=None, on_progress=None) -> Path`

1. Clears the stop flag and creates the run folder with `make_run_dir`.
2. Writes `config.json` with `{"base_config": config.to_dict(), "sweep": asdict(sweep)}`.
3. For each value, in order: stop if asked; build the value's config with `set_field`; set `config.seed = sweep.seed`; create a `Runner` with `num_games=games_per_value`, `workers`, `output_dir=run_dir/batches/<value>`, `collect_telemetry=sweep.telemetry`, and the scenario; call `runner.run(show_progress=False)`.
4. Build a row: `{"value": value, **batch_metrics(...)}`. If telemetry was collected, add `entropy`, `mean_survival_ticks` and `kill_rate` from `runner.telemetry_summary` and keep the whole summary in `self.summaries`.
5. Append the row, call `on_value(row)` and `on_progress(index + 1, len(values))` if given.
6. After the loop call `write()` and return `run_dir`.

`Runner.run` always calls `save()`, so each `batches/<value>/` folder receives `eliminations.csv`, `players.csv`, `games.csv`, `gifts.csv` and, with telemetry, `telemetry.json`. That is useful: you can redraw any chapter 3 chart for a single value later.

#### `stop(self) -> None`

Raises the stop flag. `run()` checks it at the top of each value, so the current value finishes and `write()` still runs with whatever was collected.

#### `write(self) -> None`

Does nothing without a run folder or rows. Otherwise:

- `results.csv`: one row per value from `pd.DataFrame(self.rows)`.
- `summary.json`: `{"rows": rows, "telemetry": summaries}`.
- `plots/<metric>.png`: one `plots.curves` chart per numeric column other than `value`, titled `"<metric> vs <parameter>"`. The x values are the swept values as strings.
- If any summaries exist: `plots/action_distribution_by_value.png` (a stacked area with the parameter as the x label) and a `plots/behaviour/` folder holding `plots.behaviour_plots` of **all** the summaries merged together.

### The run folder layout

```
results/water_threshold_20260902_153000/
    config.json                     base config and the sweep settings
    results.csv                     one row per value: value + every batch_metrics key
                                    + entropy, mean_survival_ticks, kill_rate when telemetry is on
    summary.json                    {"rows": [...], "telemetry": [one merged summary per value]}
    plots/
        victor_rate.png             one PNG per numeric metric, x = the swept values
        mean_days.png
        player_vs_player_share.png
        ...
        action_distribution_by_value.png
        behaviour/                  behaviour_plots of every value merged
            instinct_curves.png
            position_heatmap.png
            ...
    batches/
        0.1/                        the runner's CSV files and telemetry.json for that value
        0.25/
        0.4/
```

## How to use it / experiment

From the command line (see `experiments/run_sweep.py`):

```bash
python experiments/run_sweep.py --parameter chaos --values 0,0.25,0.5,0.75,1 --games 50 --workers 4
python experiments/run_sweep.py --parameter terrain.water_threshold --values 0.1,0.25,0.4
python experiments/run_sweep.py --parameter gamemaker_enabled --values false,true
```

The script parses each token as a bool, int, float or string, in that order.

From Python, with a painted map:

```python
from hunger_games.config import SimulationConfig
from hunger_games.research.experiments import Sweep, SweepConfig
from hunger_games.scenario import Scenario

settings = SweepConfig(name="vision", parameter="vision_radius", values=[4, 8, 12, 16], games_per_value=40, workers=4)
sweep = Sweep(SimulationConfig(), settings, scenario=Scenario.load("maps/island.json"))
folder = sweep.run(on_value=lambda row: print(row["value"], f"{row['victor_rate']:.0%}"))
```

From the dashboard: the Research tab has the same fields (parameter, comma-separated values, games per value, CPU workers, the telemetry checkbox), a Start button and a Stop button. It runs the sweep in a background thread on the painted map and prints where it saved.

Reading the results back:

```python
import pandas as pd, json
table = pd.read_csv(folder / "results.csv")
print(table[["value", "victor_rate", "mean_days", "natural_share"]])
telemetry = json.load(open(folder / "summary.json"))["telemetry"]
```

What `results.csv` looks like for a three-value sweep with telemetry on:

| value | games | victor_rate | mean_days | player_vs_player_share | natural_share | gamemaker_share | eliminations_per_point | mean_interventions | eliminations_per_game | entropy | mean_survival_ticks | kill_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 30 | 0.83 | 9.4 | 0.61 | 0.31 | 0.08 | 0.14 | 1.2 | 22.9 | 1.62 | 131.0 | 0.61 |
| 0.25 | 30 | 0.80 | 9.9 | 0.58 | 0.36 | 0.06 | 0.13 | 1.1 | 22.8 | 1.64 | 138.5 | 0.58 |
| 0.4 | 30 | 0.77 | 10.6 | 0.55 | 0.40 | 0.05 | 0.12 | 1.0 | 22.7 | 1.66 | 146.2 | 0.55 |

The numbers are illustrative. Each row is one value; each column becomes one PNG in `plots/`.

A two-parameter grid is two nested sweeps. `Sweep` handles one field, so fix the outer value with `set_field` and sweep the inner one:

```python
from hunger_games.research.experiments import Sweep, SweepConfig, set_field

for water in (0.1, 0.25, 0.4):
    base = set_field(SimulationConfig(), "terrain.water_threshold", water)
    inner = SweepConfig(name=f"chaos_water{water}", parameter="chaos", values=[0.0, 0.5, 1.0], games_per_value=30)
    Sweep(base, inner).run()
```

Each inner sweep gets its own folder, and `config.json` in each records the outer value under `base_config`.

Adding a metric of your own: `batch_metrics` returns a dictionary, so wrap it.

```python
from hunger_games.research import experiments

original = experiments.batch_metrics

def with_gifts(eliminations, players, games):
    row = original(eliminations, players, games)
    row["mean_gifts"] = float(players["gifts_received"].mean()) if len(players) else 0.0
    return row

experiments.batch_metrics = with_gifts   # Sweep.run looks the name up at call time
```

The new column appears in `results.csv` and gets its own `plots/mean_gifts.png` because `write()` charts every numeric column.

Questions this answers well:

- Does the shrinking circle create victors? Sweep `gamemaker_enabled` and look at `victor_rate.png`.
- Does more water make thirst deaths vanish? Sweep `terrain.water_threshold` and read `natural_share.png` plus `behaviour/deaths_by_cause.png`.
- Does chaos flatten the score-to-placement link? Sweep `chaos` and compare `batches/<value>/players.csv` with `plots.placement_by_score`.

## Gotchas

- `set_field` does not convert types. Enum fields like `shape` and `layout` need the enum value from Python, not a string. The CLI will hand you a string and the arena will reject it.
- A typo in the dotted path raises `AttributeError` from `getattr` or silently adds a new attribute with `setattr` if only the last part is wrong. Check `results.csv` reflects the change you meant.
- The swept values are plotted as evenly spaced labels, not on a numeric axis. `[0, 1, 10, 100]` will look linear.
- Every value replays `games_per_value` games. Five values at 50 games each is 250 games; with telemetry on, expect roughly twice the time of a plain batch.
- With `telemetry=True` and the default `brain_name="voting"`, the behaviour charts describe the hand-coded voting brain. To measure a trained brain, load its genome into the roster through a scenario first.
- The value is used as a folder name under `batches/`. Values with characters your file system dislikes will fail there.
- `workers > 1` uses processes. On macOS the script that starts the sweep needs the `if __name__ == "__main__":` guard, which `run_sweep.py` has.
- `stop()` is only checked between values. A long value keeps running until its games finish.
