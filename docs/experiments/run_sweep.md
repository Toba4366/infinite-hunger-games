# `run_sweep.py`

**Source:** [experiments/run_sweep.py](../../experiments/run_sweep.py)
**Depends on:** `argparse`, `sys`, `pathlib` (standard library); [hunger_games/config.py](../config.md) (`SimulationConfig`); [hunger_games/research/experiments.py](../../hunger_games/research/experiments.py) (`Sweep`, `SweepConfig`)
**Used by:** nobody imports it. It is run from the command line.

## Purpose

A command-line wrapper around the research package's `Sweep`. It changes one config value at a time, plays a batch of games for each value on the same seeds, and writes a results folder with one chart per metric. The question it answers is "what happens to the games when I change X?" It does no training.

```
python experiments/run_sweep.py --parameter chaos --values 0,0.25,0.5,0.75,1 --games 50 --workers 4
python experiments/run_sweep.py --parameter terrain.water_threshold --values 0.1,0.25,0.4
python experiments/run_sweep.py --parameter gamemaker_enabled --values false,true
```

## Concepts you need

**Parameter sweep.** Hold everything fixed, vary one setting across a list of values, measure the same metrics for each. Because game `N` of every value uses seed `--seed + N`, the values are compared on the same arenas, which removes most of the noise between them.

**Dotted field names.** `terrain.water_threshold` means the `water_threshold` field of the `terrain` object inside `SimulationConfig`. `set_field` in `research/experiments.py` walks the dots with `getattr` and sets the last part with `setattr` on a deep copy.

**`sys.path` and the `__main__` guard.** Same as the other two scripts: the repo root goes first on `sys.path` so the package imports without installation, and `main()` is guarded so `spawn`-started workers on macOS can import the file without starting their own sweep. Here the pool lives inside `Runner` (see [../runner.md](../runner.md)), but the rule is the same.

## Walkthrough

### Path setup

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

Then `SimulationConfig`, `Sweep` and `SweepConfig` are imported with `# noqa: E402`.

### `parse_value(text)`

```python
def parse_value(text: str):
```

Turns one command-line token into a Python value. `"true"` / `"false"` (any case) become booleans. Then `int(text)` is tried, then `float(text)`, and if both raise `ValueError` the string is returned as is. So `--values 0,0.25,1` gives `[0, 0.25, 1]` (an int, a float, an int) and `--values ring,cornucopia` gives two strings.

### `main()`

```python
def main() -> None:
```

**Flags.**

| Flag | Default | Goes to | Meaning |
| --- | --- | --- | --- |
| `--parameter` | required | `SweepConfig.parameter` | Config field, dotted for nested |
| `--values` | required | `SweepConfig.values` | Comma-separated values, each through `parse_value` |
| `--games` | `50` | `SweepConfig.games_per_value` | Games per value |
| `--workers` | `1` | `SweepConfig.workers` | CPU cores |
| `--seed` | `1000` | `SweepConfig.seed` | Base seed shared by every value |
| `--no-telemetry` | off | `SweepConfig.telemetry = not flag` | Skip behaviour measurement (faster, no behaviour charts) |
| `--size` | `120` | `SimulationConfig.width` and `height` | Arena size |
| `--name` | `None` | `SweepConfig.name` | Folder prefix; defaults to the parameter with dots replaced by underscores |
| `--results` | `results` | `SweepConfig.results_dir` | Where the folder goes |

**Configs.** `SimulationConfig(width=size, height=size)` is the base; everything else is default. `SweepConfig(name, parameter, values, games_per_value, workers, seed, telemetry, results_dir)`.

**Run.** `Sweep(config, sweep).run(on_value=...)`. After each value the callback prints the row as a dict with floats rounded to 3 places, for example:

```
{'value': 0.25, 'games': 50, 'victor_rate': 0.84, 'mean_days': 9.12, 'player_vs_player_share': 0.61, ...}
```

Then `print(f"saved to {folder}")`.

### The guard

```python
if __name__ == "__main__":
    main()
```

## How to use it / experiment

**Which knob matters most?** Sweep `chaos`, `num_players`, `vision_radius`, `sponsor_gift_chance`, `intervention_days` and `terrain.water_threshold` in turn, 50 games each, and look at `plots/victor_rate.png` and `plots/mean_days.png` for each.

**A fast check.**

```
python experiments/run_sweep.py --parameter num_players --values 12,24,36 --games 10 --size 80 --no-telemetry
```

**What the results folder contains.** `results/<name>_<timestamp>/`:

| Path | Contents |
| --- | --- |
| `config.json` | `{"base_config": SimulationConfig.to_dict(), "sweep": SweepConfig as a dict}` |
| `results.csv` | One row per value with the columns below |
| `summary.json` | `{"rows": the same rows, "telemetry": one merged telemetry summary per value}` |
| `plots/<metric>.png` | One line chart per numeric column of `results.csv`, metric against the parameter |
| `plots/action_distribution_by_value.png` | Stacked area of action shares across the values (telemetry on) |
| `plots/behaviour/` | The twelve behaviour charts for the telemetry merged over all values (telemetry on) |

The row columns come from `batch_metrics`: `games`, `victor_rate`, `mean_days`, `player_vs_player_share`, `natural_share`, `gamemaker_share`, `eliminations_per_point`, `mean_interventions`, `eliminations_per_game`. With telemetry on, three more: `entropy`, `mean_survival_ticks`, `kill_rate`.

Each value's `Runner` is pointed at `batches/<value>/` inside the folder, but `Sweep` never calls `Runner.save`, so no per-value CSVs are written and that folder does not appear.

**Read it back.** `pandas.read_csv(folder / "results.csv")` and plot any column against `value`.

**From the dashboard.** The Research tab runs the same `Sweep` on the painted map through `Session.start_sweep`.

## Gotchas

- With `--workers` above 1, run from a file. The spawn rule applies through `Runner`'s process pool.
- Values are parsed as bool, int, float or string. Enum fields such as `shape` and `layout` need `ArenaShape` / `LayoutName` members, and `set_field` will store the raw string instead, which breaks arena building. Sweep those from Python, not from this script.
- `--values 1,2` gives ints while `--values 1.0,2.0` gives floats. For fields typed as float that is harmless; for `num_players` pass ints.
- A typo in `--parameter` raises `AttributeError` from `set_field`, after the run folder and `config.json` have already been created.
- Every value plays `--games` games, so five values at 50 games is 250 full games. Telemetry roughly doubles the time; use `--no-telemetry` when you only need the headline metrics.
- The base `SimulationConfig` is all defaults except `--size`. To sweep on top of a different baseline (say `gamemaker_enabled=False`), edit the `SimulationConfig(...)` call in a copy of the script.
- `--name` with a dot in it is used verbatim; only the default name replaces dots.
