# `runner.py`

**Source:** [hunger_games/runner.py](../hunger_games/runner.py)
**Depends on:** [config.py](config.md) (`SimulationConfig`), [game.py](game.md) (`Game`), [records.py](records.md) (`GameResult`), `research/telemetry.py` (`BehaviorTelemetry`), [scenario.py](scenario.md) (`Scenario`), `pandas`, and the standard library (`json`, `concurrent.futures.ProcessPoolExecutor`, `pathlib.Path`).
**Used by:** [init.md](init.md) (re-exports `Runner`), [main.md](main.md) (the `simulate` command), `research/experiments.py` (one `Runner` per swept value, reading `telemetry_summary`).

## Purpose

This is the "infinite" part. Chapter 3 wanted hundreds of games to find reliable trends. `Runner` plays as many games as you ask, optionally across several CPU cores, and writes the results as CSV files: one row per elimination, one per player-game, one per game, and one per sponsor gift. With `collect_telemetry=True` it also measures behaviour in every game and writes a merged `telemetry.json`.

## Concepts you need

**Process pools.** `ProcessPoolExecutor(max_workers=n)` starts `n` Python processes. `pool.map(func, a, b, c, d)` calls `func(a[i], b[i], c[i], d[i])` in the workers and yields results in order. Everything sent to or from a worker must pickle, which is why `run_single_game` is a top-level function and results are plain dataclasses and dictionaries.

**pandas from lists of dictionaries.** `pd.DataFrame([{"a": 1}, {"a": 2}])` builds a table with one column per key. `to_csv(index=False)` writes it without the row numbers.

**Carriage return progress.** `print("\r...", end="")` moves the cursor to the start of the line so the counter overwrites itself.

**Merging summaries.** Each game produces its own telemetry dictionary. `BehaviorTelemetry.merge` sums the arrays and concatenates the lists so the batch has one summary.

## Walkthrough

### `run_single_game`

```python
def run_single_game(
    config: SimulationConfig, game_id: int, collect_telemetry: bool = False, scenario: Scenario | None = None
) -> GameResult
```

Builds `Game(config, game_id, scenario=scenario)`. If `collect_telemetry`, a `BehaviorTelemetry(arena.width, arena.height)` is attached (it registers a decision hook and a tick hook on the game). The game runs, and if telemetry was collected, `result.telemetry = telemetry.summary()`. Returns the `GameResult`.

It is a module-level function so the process pool can pickle a reference to it.

### `Runner.__init__`

```python
def __init__(
    self,
    config: SimulationConfig,
    num_games: int = 100,
    workers: int = 1,
    output_dir: str | Path = "output",
    collect_telemetry: bool = False,
    scenario: Scenario | None = None,
) -> None
```

| Attribute | From | Meaning |
| --- | --- | --- |
| `collect_telemetry` | argument | Measure behaviour in every game. |
| `scenario` | argument | Optional painted map, loot and roster used by every game. |
| `telemetry_summary` | `None` | Filled by `run()` when telemetry was collected. |
| `config` | argument | Shared settings. |
| `num_games` | argument | How many games. |
| `workers` | `max(1, workers)` | CPU cores. |
| `output_dir` | `Path(output_dir)` | Where files go. |

### `Runner.run`

```python
def run(self, show_progress: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
```

1. With `workers > 1`, a `ProcessPoolExecutor` maps `run_single_game` over `[config] * n`, `range(n)`, `[collect_telemetry] * n`, `[scenario] * n`. Results arrive in game-id order. Otherwise the games run in a plain loop.
2. After each game `_progress` prints the counter. A newline finishes the line if progress was shown.
3. Four tables are built: `eliminations` from `elimination_rows()`, `players` from `player_rows()`, `games` with the columns `game_id`, `seed`, `days`, `ticks`, `winner_id`, `winner_name`, `interventions`, and `gifts` from `result.gifts`.
4. If `collect_telemetry`, `self.telemetry_summary = BehaviorTelemetry.merge([...])` over every result that has a telemetry dictionary.
5. `save(...)` writes the files.
6. Returns `(eliminations, players, games)`. The gifts table is only saved, not returned.

### `Runner._progress`

```python
def _progress(self, done: int, show: bool) -> None
```

Prints `games finished: done/num_games` on one overwriting line when `show` is true.

### `Runner.save`

```python
def save(
    self, eliminations: pd.DataFrame, players: pd.DataFrame, games: pd.DataFrame, gifts: pd.DataFrame | None = None
) -> None
```

Creates `output_dir` if needed and writes:

| File | Contents | Always written |
| --- | --- | --- |
| `eliminations.csv` | One row per elimination. | Yes |
| `players.csv` | One row per player per game. | Yes |
| `games.csv` | One row per game. | Yes |
| `gifts.csv` | One row per sponsor gift (may be empty). | When `gifts` is given |
| `telemetry.json` | The merged behaviour summary. | When `telemetry_summary` is not `None` |

The columns are described in [output.md](output.md); the telemetry keys are the ones returned by `BehaviorTelemetry.summary()`.

### What `telemetry.json` holds

The file is the dictionary returned by `BehaviorTelemetry.merge`, which has the same keys as `BehaviorTelemetry.summary()`:

| Key | Shape | Meaning |
| --- | --- | --- |
| `games` | int | Games merged. |
| `action_names` | list of 9 strings | The action kinds, in column order. |
| `action_counts` | 9 numbers | Total count of each action. |
| `action_by_thirst`, `action_by_hunger`, `action_by_health`, `action_by_alive` | 5 by 9 | Action counts per need or alive-fraction bin. |
| `combat_by_health` | 5 by 2 | Attack versus flee per health bin when someone was in sight. |
| `position_heat`, `armed_heat`, `unarmed_heat` | 30 by 30 | Where tributes spent time. |
| `proximity_sum`, `proximity_count` | 5 numbers | Distance kept from the nearest tribute per alive bin. |
| `thirst_at_drink`, `hunger_at_eat`, `health_at_heal` | 10 numbers | Bar level at each drink, meal and heal. |
| `death_needs`, `death_count`, `mean_death_needs` | 3 numbers, int, 3 numbers | Bars at the moment of death. |
| `deaths_by_cause` | dict | Count per `cause_of_death` string. |
| `survival_ticks`, `kills`, `wins`, `placements`, `post_injury_ticks` | lists | One entry per tribute per game. |
| `entropy`, `mean_survival_ticks`, `win_rate`, `kill_rate` | numbers | Derived summaries, recomputed after merging. |

[analysis.md](analysis.md) reads this file to draw the behaviour charts, and `research/experiments.py` copies `entropy`, `mean_survival_ticks` and `kill_rate` into each sweep row.

## How to use it / experiment

A batch from Python:

```python
from hunger_games.config import SimulationConfig
from hunger_games.runner import Runner

runner = Runner(SimulationConfig(seed=1), num_games=20, workers=4, output_dir="output", collect_telemetry=True)
eliminations, players, games = runner.run()
print(games["winner_id"].notna().mean(), "of games had a victor")
print(runner.telemetry_summary["entropy"], "action entropy")
```

The same from the command line (note the `--gamemaker` flag, see Gotchas):

```
python -m hunger_games simulate --games 500 --workers 4 --seed 1 --gamemaker
python -m hunger_games analyze
```

Then [analysis.md](analysis.md) draws the charts, and reads `telemetry.json` for the behaviour charts if it exists.

For a scenario from the dashboard:

```python
from hunger_games.scenario import Scenario

runner = Runner(config, num_games=50, scenario=Scenario.load("my_arena.json"))
```

Parameter sweeps that build one `Runner` per value and gather `telemetry_summary` into `summary.json` live in `research/experiments.py`.

## Gotchas

- Multi-core runs need the script to be importable. On macOS and Windows put the `Runner` call under `if __name__ == "__main__":` or the worker processes will re-run your script.
- Seeds: `Game` uses `config.seed + game_id`, so games in a batch are different but reproducible. With `seed=None` every game is random and nothing is reproducible.
- `run()` always overwrites the CSVs in `output_dir`. Use a different `output_dir` per experiment.
- `run()` returns three tables but writes four files; read `gifts.csv` from disk if you need it.
- With `collect_telemetry=True` each game's `GameResult` carries a summary dictionary that includes a 30 by 30 heatmap and per-tribute lists. For thousands of games this is memory and pickling overhead; leave it off when you only need the CSVs.
- The command line's `simulate` command does not expose `collect_telemetry`; use Python or the dashboard for telemetry.
- The command line's `--gamemaker` / `--no-gamemaker` pair defaults to `SimulationConfig()`'s value (on), so batches from the command line and from Python match unless you pass `--no-gamemaker`.
