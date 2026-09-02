# `analysis.py`

**Source:** [hunger_games/analysis.py](../hunger_games/analysis.py)
**Depends on:** `pandas`, `matplotlib.pyplot`, the standard library (`json`, `pathlib.Path`), and `research/plots.py` (imported inside `make_report`, for the individual PNGs and the behaviour charts).
**Used by:** [main.md](main.md) (the `analyze` command calls `make_report`).

## Purpose

Chapter 3 of the video plotted eliminations per day, eliminations by method, weapons used, eliminations per training-score point and placing by score, then complained about only having three games of data. These functions do the same maths over as many games as [runner.md](runner.md) produced.

Each statistic is a small function that takes a table and returns a pandas `Series`, so you can use them in a notebook. `make_report` ties them together: it writes one PNG per chart under `output/plots/`, adds the behaviour charts when `telemetry.json` exists, draws a combined `report.png`, and prints the headline numbers.

## Concepts you need

**pandas groupby.** `eliminations.groupby("day").size()` counts rows per day. `players.groupby("training_score")["placement"].mean()` averages one column within each group. The result is a `Series` indexed by the group key.

**value_counts.** `series.value_counts()` counts each distinct value, most common first. `.sort_index()` orders by the value instead.

**Series plotting.** `series.plot.bar(ax=...)` and `series.plot(ax=..., marker="o")` draw straight onto a matplotlib axis.

**Subplot grids.** `plt.subplots(2, 3, figsize=(16, 9))` returns a figure and a 2 by 3 array of axes indexed `axes[row, col]`.

**Lazy imports.** `from hunger_games.research import plots` sits inside `make_report` so importing `analysis` for the small functions never pulls in the whole research package.

## Walkthrough

### `load_results`

```python
def load_results(output_dir: str | Path = "output") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
```

Reads `eliminations.csv`, `players.csv` and `games.csv` from the folder and returns them in that order. It does not read `gifts.csv` or `telemetry.json`.

### `eliminations_per_day`

```python
def eliminations_per_day(eliminations: pd.DataFrame, num_games: int) -> pd.Series
```

Rows per `day`, divided by the number of games: the average eliminations per game on each day. Chapter 2's exponential decay.

### `eliminations_by_method`

```python
def eliminations_by_method(eliminations: pd.DataFrame) -> pd.Series
```

`value_counts` of the `method` column: `player_vs_player`, `natural_causes`, `gamemaker`.

### `weapons_used`

```python
def weapons_used(eliminations: pd.DataFrame) -> pd.Series
```

Filters to `method == "player_vs_player"` and counts the `weapon` column.

### `kills_per_training_point`

```python
def kills_per_training_point(players: pd.DataFrame) -> float
```

The mean of `kills / training_score` across every player row. The video computed 0.134 from three games.

### `placement_by_training_score`

```python
def placement_by_training_score(players: pd.DataFrame) -> pd.Series
```

Mean `placement` per training score. Lower is better.

### `game_lengths`

```python
def game_lengths(games: pd.DataFrame) -> pd.Series
```

How many games lasted each number of days, sorted by length.

### `make_report`

```python
def make_report(output_dir: str | Path = "output", show: bool = False) -> Path
```

Step by step:

1. `load_results(output_dir)`.
2. Imports `hunger_games.research.plots`.
3. Arena size for the death heatmap: one more than the largest `x` and `y` in the eliminations table, or 120 by 120 if there are no eliminations.
4. `plots.batch_plots(eliminations, players, games, <output_dir>/plots, width, height)` writes one PNG per chart and returns the paths.
5. If `<output_dir>/telemetry.json` exists, it is parsed and `plots.behaviour_plots(summary, <output_dir>/plots/behaviour)` writes the behaviour charts too.
6. Prints `wrote N individual charts to <output_dir>/plots/`.
7. Draws the combined 2 by 3 figure and saves `<output_dir>/report.png` at 120 dpi.
8. Prints games, eliminations per training point, eliminations by method as percentages, and average interventions per game.
9. `plt.show()` if `show`.
10. Returns the path of `report.png`.

The individual charts written under `plots/`:

| File | Chart |
| --- | --- |
| `eliminations_per_day.png` | Average eliminations per game per day. |
| `eliminations_by_method.png` | The three categories. |
| `weapons_used.png` | Weapons in player-vs-player kills. |
| `placement_by_score.png` | Average placing per training score. |
| `kills_by_score.png` | Average kills per training score. |
| `game_lengths.png` | Games per length in days. |
| `deaths_by_district.png` | Eliminations per district. |
| `death_heatmap.png` | Where tributes died (only when there are eliminations). |

The behaviour charts under `plots/behaviour/`, only when `telemetry.json` exists:

| File | Chart |
| --- | --- |
| `action_distribution.png` | Share of each action overall. |
| `actions_by_thirst.png`, `actions_by_hunger.png`, `actions_by_health.png` | Action mix per need bin. |
| `instinct_curves.png` | Need-driven action curves. |
| `consumption_timing.png` | Bar level at each drink, meal and heal. |
| `fight_or_flight.png` | Attack versus flee by health. |
| `proximity_vs_remaining.png` | Distance kept from others as the field thins. |
| `actions_by_remaining.png` | Action mix by tributes remaining. |
| `position_heatmap.png` | Where tributes spend time. |
| `armed_vs_unarmed_heatmaps.png` | The same split by weapon. |
| `deaths_by_cause.png` | Deaths per cause. |

The combined `report.png` holds six panels: eliminations per day, by method, weapons, placing by score (y axis inverted so better is up), kills by score, and game lengths.

## How to use it / experiment

From the command line, after a batch:

```
python -m hunger_games simulate --games 200 --seed 1 --gamemaker
python -m hunger_games analyze --show
```

In a notebook, use the small functions directly:

```python
from hunger_games.analysis import (
    eliminations_by_method,
    eliminations_per_day,
    kills_per_training_point,
    load_results,
    placement_by_training_score,
)

eliminations, players, games = load_results("output")
print(eliminations_per_day(eliminations, len(games)))
print(eliminations_by_method(eliminations) / len(eliminations))
print(placement_by_training_score(players))
print(kills_per_training_point(players))
```

Compare two batches by running the runner into two folders and calling each function on both. To add a chart to `report.png`, add a panel to the grid in `make_report`; to add an individual PNG, add a function to `research/plots.py` and a line to `batch_plots`.

For behaviour charts you need telemetry, which the `simulate` command does not collect. Run the batch from Python with `Runner(..., collect_telemetry=True)` so `telemetry.json` is written, then `analyze` picks it up.

## Gotchas

- `make_report` needs all three CSVs. An empty `eliminations.csv` (no header, no rows) makes `pd.read_csv` raise; a batch with zero deaths is unusual but possible with tiny `max_days`.
- The heatmap size is inferred from the largest death coordinate, so a batch where nobody died near the far edge gets a slightly smaller grid than the arena. Only the binning changes.
- The percentages printed use `len(eliminations)` as the denominator, so a batch with no eliminations would divide by zero. `kills_per_training_point` only fails if a training score is zero (it never is; scores are 1..12).
- Every call rewrites `report.png` and every file under `plots/`. Copy them elsewhere before rerunning.
- `matplotlib` opens a window only with `show=True`; on a headless machine set the backend (`MPLBACKEND=Agg`) or leave `show` off.
- `simulate` keeps the game makers on by default (the config value); run it with `--no-gamemaker` and `eliminations_by_method` will show no `gamemaker` rows.
