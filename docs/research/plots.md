# `plots.py`

**Source:** [hunger_games/research/plots.py](../../hunger_games/research/plots.py)
**Depends on:** `pathlib`; `matplotlib` (with the `Agg` backend forced at import), `matplotlib.animation.FuncAnimation`; `numpy`; `pandas`; `pillow` (GIF writer); [telemetry.md](telemetry.md) (`ALIVE_BIN_LABELS`, `NEED_BIN_LABELS`)
**Used by:** [../training/runs.md](../training/runs.md) (`training_run_plots`); [experiments.md](experiments.md) (`curves`, `stacked_area_over_training`, `behaviour_plots`); [../analysis.md](../analysis.md) (`make_report` imports it inside the function and calls `batch_plots` and `behaviour_plots`, so `python -m hunger_games analyze` writes these PNGs too); [../ui/session.md](../ui/session.md) (`behaviour_plots` for the export button); `tests/test_research.py`

## Purpose

One function per chart. Each one draws exactly one figure, saves it to the path you give, closes it, and returns the path. Nothing is combined into a grid, so every chart can be dropped into a paper and cited on its own. [../analysis.md](../analysis.md) still draws its combined `report.png` for a quick look, but the individual PNGs under `output/plots/` come from here.

The functions read three kinds of data:

- The CSV tables the runner writes: `eliminations`, `players`, `games` (see [../output.md](../output.md)).
- A telemetry summary dictionary from [telemetry.md](telemetry.md).
- The history rows a trainer keeps (`trainer.history_rows()`).

## Concepts you need

**Backends.** matplotlib can draw into a window or into a file. `matplotlib.use("Agg")` at the top of this file picks the file-only backend, so plotting works on servers, in worker threads and in the dashboard's background threads.

**Figure and axes.** `fig, ax = plt.subplots()` makes a canvas (`fig`) with one drawing area (`ax`). Bars, lines and images go on `ax`. Titles and labels are set on `ax`. Saving happens on `fig`.

**Stacked bars and stacked areas.** To stack, each series is drawn on top of the previous total. The code keeps a running `bottom` array and adds each series to it after drawing. A stacked area does the same with `ax.stackplot`.

**Row shares.** Most behaviour charts divide each row of a table by its row total, so bars show percentages of the decisions made at that level. A level with no decisions is divided by 1 instead of 0 and stays empty.

**Nats.** Entropy here uses the natural log. Nine equally likely actions give `ln 9 = 2.20` nats.

## Walkthrough

### `ACTION_COLORS`

Nine colours in `ACTION_NAMES` order: rest grey, move blue, drink bright blue, eat yellow, hunt green, pick_up pink, heal orange, attack red, flee purple. Every action chart uses them, so an action keeps its colour across every figure in a paper.

### `_save(fig, path) -> Path`

Calls `fig.tight_layout()`, creates the parent folder, saves at 140 dpi, closes the figure to free memory, and returns the path. Every chart ends with it.

### Chapter 3 charts (from the CSV tables)

| Function | Draws | Needs | What a researcher wants to see |
| --- | --- | --- | --- |
| `eliminations_per_day(eliminations, num_games, path)` | Bar of average eliminations per day | `eliminations["day"]` | Chapter 2's decay: a bloodbath on day 1, then fewer each day |
| `eliminations_by_method(eliminations, path)` | Bar of the share of `player_vs_player`, `natural_causes`, `gamemaker` | `eliminations["method"]` | Whether fights or the environment do the killing |
| `weapons_used(eliminations, path)` | Bar of weapon names in player-versus-player deaths | `method`, `weapon` | Whether good weapons matter |
| `placement_by_score(players, path)` | Line of mean placing per training score, y axis inverted so better is higher | `training_score`, `placement` | A rising line: high scorers place better |
| `kills_by_score(players, path)` | Line of mean kills per training score | `training_score`, `kills` | Rising with score |
| `game_lengths(games, path)` | Bar of how many games lasted each number of days | `games["days"]` | Few games hitting the day cutoff |
| `death_heatmap(eliminations, width, height, path, cells=30)` | 2-D density of death positions via `np.histogram2d` | `x`, `y` | Where the danger is |
| `deaths_by_district(eliminations, path)` | Bar of deaths per district | `victim_district` | Careers dying less |

### Heatmaps

#### `heatmap(grid, title, path, cmap="magma") -> Path`

Takes any 2-D array, divides it by its total so colour means share of time, draws it with `origin="upper"` so row 0 is the top of the arena, adds a colour bar, hides the ticks. Used directly for `summary["position_heat"]` and by `death_heatmap`.

#### `armed_vs_unarmed(summary, path) -> Path`

Two panels side by side from `unarmed_heat` and `armed_heat`, each normalised on its own. Titles state the threshold (`weapon < 0.4` and `>= 0.4`). A researcher wants the armed panel to light up the centre and the unarmed panel to hug the edges, which is the ring layout's whole point.

### Behaviour charts (from one telemetry summary)

| Function | Draws | Summary keys | Good trend |
| --- | --- | --- | --- |
| `action_distribution(summary, path)` | Bar of the share of each action, entropy in the title | `action_counts`, `action_names`, `entropy` | Not one bar dominating; not all bars equal either |
| `action_by_need(summary, need, path)` | Stacked bars of actions at each level of `"thirst"`, `"hunger"` or `"health"` | `action_by_<need>` | The matching action grows as the bar empties |
| `need_action_curves(summary, path)` | Lines of P(drink given thirst), P(eat given hunger), P(heal given health) | the three `action_by_*` tables | Steep rise at the low end |
| `consumption_timing(summary, path)` | Three histograms of the bar level at every drink, meal and heal | `thirst_at_drink`, `hunger_at_eat`, `health_at_heal` | Mass at low levels, not spread flat |
| `fight_or_flight(summary, path)` | Stacked bars of attack versus flee by health, someone in sight | `combat_by_health` | Flee wins at low health, attack at high health |
| `proximity_vs_alive(summary, path)` | Line of mean distance to the nearest visible tribute, from most alive to final few | `proximity_sum`, `proximity_count` | Distance kept early, closing at the end |
| `action_by_alive(summary, path)` | Stacked bars of actions by tributes remaining | `action_by_alive` | Attack share rising toward the final few |
| `deaths_by_cause(summary, path)` | Bar per cause name | `deaths_by_cause` | Starvation and dehydration shrinking |

Details worth knowing:

- `action_by_need` builds the file name from the need, and the title reads "What tributes do at each thirst level".
- `need_action_curves` finds the columns by name with `names.index("drink")`, so it works on any summary whose `action_names` came from `ActionType`.
- `consumption_timing` puts bar centres at 5, 15, ... 95 percent with width 9.
- `proximity_vs_alive` and `action_by_alive` reverse the bin order so the x axis reads left to right from "most alive" to "final few".

### Over-training charts (from a list of summaries or history rows)

#### `curves(xs, series, title, xlabel, ylabel, path) -> Path`

The general multi-line chart. `series` maps a label to a list of y values. A legend appears only with more than one line. Every performance chart is built on it, and the sweep uses it once per metric.

#### `stacked_area_over_training(summaries, path, xlabel="generation") -> Path`

One row per training step from each summary's `action_counts`, converted to percentages, drawn with `ax.stackplot`. This is the chart that shows behaviour changing: near-random bands at the start and a structured pattern later. An empty list draws an empty `curves` chart rather than failing.

#### `death_needs_over_training(summaries, path, xlabel="generation") -> Path`

Three lines from `mean_death_needs`: thirst, hunger and health at death per step. If tributes stop dying of thirst, the thirst-at-death line rises because deaths move to fights, which end with a full thirst bar and empty health.

#### `behaviour_metrics_over_training(summaries, path, xlabel="generation") -> Path`

A 2 by 2 panel: `mean_survival_ticks`, `win_rate`, `kill_rate`, `entropy` per step. This is the one bundle function that puts several panels in one file, because the four numbers are read together.

#### `timing(history_rows, path, xlabel="step") -> Path`

Grey bars of `seconds` per step with a red cumulative line on a second y axis. Uses `cumulative_seconds` when present, otherwise a running sum. The x value is `generation` or `epoch`, whichever the row has.

#### `curve_gif(xs, series, title, xlabel, ylabel, path, fps=6) -> Path`

Animates a chart growing one point per frame, for slides. It creates one empty line per series, fixes the axis limits from all the data with 5 percent padding so the view never jumps, and in `draw(frame)` sets each line's data to the first `frame + 1` points. `FuncAnimation` drives it and `animation.save(..., writer="pillow", fps=fps)` writes the GIF. Frame count is `max(1, len(xs))`.

### Bundles

#### `training_run_plots(history_rows, summaries, folder, method) -> list[Path]`

Writes every chart a training run should have. `method` is `"genetic"` or anything else (treated as REINFORCE).

| Method | Files |
| --- | --- |
| genetic | `fitness.png` (best, mean, validation), `fitness.gif` |
| reinforce | `reward.png` (train and validation return), `losses.png` (policy and value), `entropy.png`, `survival.png`, `win_kill_rate.png`, `reward.gif` |
| both | `timing.png` |
| both, when `summaries` is not empty | `action_distribution_over_training.png`, `death_needs_over_training.png`, `behaviour_over_training.png`, then every `behaviour_plots` file for the **last** summary |

With no history rows it returns an empty list.

#### `behaviour_plots(summary, folder) -> list[Path]`

Every behaviour chart for one summary, in this order and with these names: `action_distribution.png`, `actions_by_thirst.png`, `actions_by_hunger.png`, `actions_by_health.png`, `instinct_curves.png`, `consumption_timing.png`, `fight_or_flight.png`, `proximity_vs_remaining.png`, `actions_by_remaining.png`, `position_heatmap.png`, `armed_vs_unarmed_heatmaps.png`, `deaths_by_cause.png`.

#### `batch_plots(eliminations, players, games, folder, width=120, height=120) -> list[Path]`

Every chapter 3 chart for a batch: `eliminations_per_day.png`, `eliminations_by_method.png`, `weapons_used.png`, `placement_by_score.png`, `kills_by_score.png`, `game_lengths.png`, `deaths_by_district.png`, and `death_heatmap.png` when there was at least one elimination.

### Which input feeds which chart

| Input | Where it comes from | Charts |
| --- | --- | --- |
| `eliminations`, `players`, `games` tables | `Runner.run()` return values, or `pd.read_csv` on the CSV files | the eight chapter 3 charts, `batch_plots` |
| One telemetry summary | `BehaviorTelemetry.summary()`, `Runner.telemetry_summary`, `telemetry.json`, one entry of a sweep's `summary.json`, `stats.telemetry` on a trainer step | the eight behaviour charts, `heatmap`, `armed_vs_unarmed`, `behaviour_plots` |
| A list of summaries, one per step | `[s.telemetry for s in trainer.history]` or a sweep's `summary.json["telemetry"]` | `stacked_area_over_training`, `death_needs_over_training`, `behaviour_metrics_over_training` |
| History rows | `trainer.history_rows()` or `history.json` | `timing`, and the performance `curves` inside `training_run_plots` |
| Any `xs` and `series` | anything you like | `curves`, `curve_gif` |

### Reading the figures

A few conventions hold across the file, so a reader can compare charts:

- Percent axes are percent of decisions in that row, not of all decisions. A tall `drink` bar at "0-20%" thirst says drinking is common when thirsty, even if thirsty moments are rare.
- Heatmaps always show a share of time, with a colour bar labelled that way, so two heatmaps from batches of different sizes are comparable.
- x axes that describe the field run from "most alive" on the left to "final few" on the right, so time reads left to right.
- Training charts use the step index on x (`generation` or `epoch`), never wall-clock time. Use `timing.png` to convert.
- A GIF and a PNG with the same stem show the same data; the GIF is for slides.

## How to use it / experiment

From a saved run folder, without rerunning anything:

```python
import json
from hunger_games.research import plots

rows = json.load(open("results/rl_20260902_150000/history.json"))
plots.curves([r["epoch"] for r in rows], {"entropy": [r["entropy"] for r in rows]},
             "Policy entropy", "epoch", "nats", "paper/entropy.png")
```

From a sweep's `summary.json`, one behaviour chart per value:

```python
data = json.load(open("results/chaos_20260902_153000/summary.json"))
for row, summary in zip(data["rows"], data["telemetry"]):
    plots.need_action_curves(summary, f"paper/instinct_chaos_{row['value']}.png")
```

From the runner's CSV files:

```python
import pandas as pd
plots.batch_plots(pd.read_csv("output/eliminations.csv"), pd.read_csv("output/players.csv"),
                  pd.read_csv("output/games.csv"), "paper/batch")
```

A chart that does not exist yet, survival after injury, in the house style:

```python
import matplotlib.pyplot as plt
from hunger_games.research.plots import _save

def post_injury(summary, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(summary["post_injury_ticks"], bins=20, color="#59a14f")
    ax.set_title("Ticks survived after first dropping below half health")
    ax.set_xlabel("ticks")
    ax.set_ylabel("tributes")
    return _save(fig, path)
```

Ideas:

- Write your own chart by copying `fight_or_flight`: read one key from the summary, divide by row totals, draw, return `_save(fig, path)`.
- Change `ACTION_COLORS` to a colour-blind safe palette before drawing; every action chart follows.
- Pass `cmap="viridis"` to `heatmap` if you need a perceptually uniform map for print.

## Gotchas

- Importing this file switches the whole process to the `Agg` backend. Do not import it in code that wants to open a matplotlib window. The `watch` command never imports it. `analysis.make_report` imports it inside the function, after the CSV files are loaded, which is why `analyze --show` is the one place a window and this module meet; if the window fails to open there, that is the reason.
- `curve_gif` needs pillow. Without it `animation.save` raises an error.
- `training_run_plots` writes the detailed behaviour charts for the last summary only. For an earlier generation call `behaviour_plots(summaries[i], folder)` yourself.
- `stacked_area_over_training` uses the row index as x, not `generation` or `epoch`. They agree because every step contributes one summary.
- `death_heatmap` bins with `range=[[0, height], [0, width]]`, so pass the real arena size or deaths near the edge land in the wrong cell.
- Charts made from a summary with zero decisions come out empty rather than raising. Check `summary["games"]` before reading anything into a result.
- `deaths_by_cause` uses the raw cause strings from `player.cause_of_death` as x labels and rotates them 45 degrees. Long weapon names still overlap when there are many causes.
- `timing` draws bars against `generation` or `epoch` values. With a single row the bar can look oddly wide; that is matplotlib picking a default width.
