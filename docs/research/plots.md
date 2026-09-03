# `plots.py`

**Source:** [hunger_games/research/plots.py](../../hunger_games/research/plots.py)
**Depends on:** `pathlib`; `matplotlib` (with the `Agg` backend forced at import), `matplotlib.animation.FuncAnimation`; `numpy`; `pandas`; `pillow` (GIF writer); [telemetry.md](telemetry.md) (`ALIVE_BIN_LABELS`, `NEED_BIN_LABELS`)
**Used by:** [../training/runs.md](../training/runs.md) (`training_run_plots`, `learning_curve_plots`); [comparison.md](comparison.md) (`overlay_curves`, `bars`); [experiments.md](experiments.md) (`curves`, `stacked_area_over_training`, `behaviour_plots`); [../analysis.md](../analysis.md) (`make_report` imports it inside the function and calls `batch_plots` and `behaviour_plots`); [../ui/session.md](../ui/session.md) (`behaviour_plots` for the export button); `tests/test_research.py`; `tests/test_methods.py` and `tests/test_imitation.py` (through `save_run`)

## Purpose

One function per chart. Each one draws exactly one figure, saves it to the path you give, closes it, and returns the path. Nothing is combined into a grid, so every chart can be dropped into a paper and cited on its own.

The functions read four kinds of data:

- The CSV tables the runner writes: `eliminations`, `players`, `games` (see [../output.md](../output.md)).
- A telemetry summary dictionary from [telemetry.md](telemetry.md).
- The history rows a trainer keeps (`trainer.history_rows()`), which differ by method.
- The shared learning rows every trainer keeps (`IterationStats.to_row()`), which are the same for every method.

## Concepts you need

**Backends.** `matplotlib.use("Agg")` at the top of this file picks the file-only backend, so plotting works on servers, in worker threads and in the dashboard's background threads.

**Figure and axes.** `fig, ax = plt.subplots()` makes a canvas (`fig`) with one drawing area (`ax`). Bars, lines and images go on `ax`; saving happens on `fig`.

**Stacked bars and stacked areas.** Each series is drawn on top of the previous total, with a running `bottom` array, or with `ax.stackplot`.

**Row shares.** Most behaviour charts divide each row of a table by its row total, so bars show percentages of the decisions made at that level.

**Nats.** Entropy here uses the natural log.

**History row keys.** Each trainer's `to_row()` has its own keys: `train_loss` and `val_accuracy` for imitation, `best_fitness` for the GA, `train_return` for REINFORCE and PPO, and the shared `mean_score` plus `extra_species` for NEAT. `training_run_plots` picks the chart set by the `method` string, so the method must match the rows.

**Learning rows.** `IterationStats.to_row()` from [../training/common.md](../training/common.md): `iteration`, `mean_score`, `best_score`, `entropy`, `mean_length`, `win_rate`, `val_score`, `seconds`, `cumulative_seconds`, `stage`, `opponents` and `extra_*`. `learning_curve_plots` reads these and nothing else.

## Walkthrough

### `ACTION_COLORS`

Nine colours in `ACTION_NAMES` order: rest grey, move blue, drink bright blue, eat yellow, hunt green, pick_up pink, heal orange, attack red, flee purple. Every action chart uses them.

### `_save(fig, path) -> Path`

Calls `fig.tight_layout()`, creates the parent folder, saves at 140 dpi, closes the figure, and returns the path. Every chart ends with it.

### Chapter 3 charts (from the CSV tables)

| Function | Draws | Needs | What a researcher wants to see |
| --- | --- | --- | --- |
| `eliminations_per_day(eliminations, num_games, path)` | Bar of average eliminations per day | `eliminations["day"]` | A bloodbath on day 1, then fewer each day |
| `eliminations_by_method(eliminations, path)` | Bar of the share of `player_vs_player`, `natural_causes`, `gamemaker` | `eliminations["method"]` | Whether fights or the environment do the killing |
| `weapons_used(eliminations, path)` | Bar of weapon names in player-versus-player deaths | `method`, `weapon` | Whether good weapons matter |
| `placement_by_score(players, path)` | Line of mean placing per training score, y axis inverted | `training_score`, `placement` | A rising line: high scorers place better |
| `kills_by_score(players, path)` | Line of mean kills per training score | `training_score`, `kills` | Rising with score |
| `game_lengths(games, path)` | Bar of how many games lasted each number of days | `games["days"]` | Few games hitting the day cutoff |
| `death_heatmap(eliminations, width, height, path, cells=30)` | 2-D density of death positions via `np.histogram2d` | `x`, `y` | Where the danger is |
| `deaths_by_district(eliminations, path)` | Bar of deaths per district | `victim_district` | Careers dying less |

### Heatmaps

#### `heatmap(grid, title, path, cmap="magma") -> Path`

Takes any 2-D array, divides it by its total so colour means share of time, draws it with `origin="upper"`, adds a colour bar, hides the ticks.

#### `armed_vs_unarmed(summary, path) -> Path`

Two panels side by side from `unarmed_heat` and `armed_heat`, each normalised on its own.

### Behaviour charts (from one telemetry summary)

| Function | Draws | Summary keys | Good trend |
| --- | --- | --- | --- |
| `action_distribution(summary, path)` | Bar of the share of each action, entropy in the title | `action_counts`, `action_names`, `entropy` | Not one bar dominating |
| `action_by_need(summary, need, path)` | Stacked bars of actions at each level of `"thirst"`, `"hunger"` or `"health"` | `action_by_<need>` | The matching action grows as the bar empties |
| `need_action_curves(summary, path)` | Lines of P(drink given thirst), P(eat given hunger), P(heal given health) | the three `action_by_*` tables | Steep rise at the low end |
| `consumption_timing(summary, path)` | Three histograms of the bar level at every drink, meal and heal | `thirst_at_drink`, `hunger_at_eat`, `health_at_heal` | Mass at low levels |
| `fight_or_flight(summary, path)` | Stacked bars of attack versus flee by health | `combat_by_health` | Flee at low health, attack at high |
| `proximity_vs_alive(summary, path)` | Line of mean distance to the nearest visible tribute, most alive to final few | `proximity_sum`, `proximity_count` | Distance kept early, closing at the end |
| `action_by_alive(summary, path)` | Stacked bars of actions by tributes remaining | `action_by_alive` | Attack share rising toward the final few |
| `deaths_by_cause(summary, path)` | Bar per cause name | `deaths_by_cause` | Starvation and dehydration shrinking |

### Over-training charts (from a list of summaries or history rows)

#### `curves(xs, series, title, xlabel, ylabel, path) -> Path`

The general multi-line chart. `series` maps a label to a list of y values. A legend appears only with more than one line. Every performance chart is built on it.

#### `stacked_area_over_training(summaries, path, xlabel="generation") -> Path`

One row per training step from each summary's `action_counts`, as percentages, drawn with `ax.stackplot`. An empty list draws an empty `curves` chart.

#### `death_needs_over_training(summaries, path, xlabel="generation") -> Path`

Three lines from `mean_death_needs`: thirst, hunger and health at death per step.

#### `behaviour_metrics_over_training(summaries, path, xlabel="generation") -> Path`

A 2 by 2 panel: `mean_survival_ticks`, `win_rate`, `kill_rate`, `entropy` per step.

#### `timing(history_rows, path, xlabel="step") -> Path`

Grey bars of `seconds` per step with a red cumulative line on a second y axis. Uses `cumulative_seconds` when present, otherwise a running sum. The x value is `generation` or `epoch`, whichever the row has, else the row index (which is what NEAT's `iteration` rows fall back to).

#### `curve_gif(xs, series, title, xlabel, ylabel, path, fps=6) -> Path`

Animates a chart growing one point per frame. Fixes the axis limits from all the data with 5 percent padding, and `FuncAnimation` plus `animation.save(..., writer="pillow", fps=fps)` writes the GIF.

### Comparison helpers

#### `overlay_curves(series, title, xlabel, ylabel, path) -> Path`

```python
def overlay_curves(series: dict[str, tuple[list, list]], title: str, xlabel: str, ylabel: str, path: str | Path) -> Path:
```

Several `(xs, ys)` lines on one chart, one per method or variant. Unlike `curves`, every entry brings its own x values, so runs of different lengths, or runs plotted against wall-clock time, overlay correctly. Entries with empty `xs` are skipped. A legend is drawn whenever `series` is not empty. The method comparison uses it for `score_by_method.png`, `score_by_time.png`, `validation_by_method.png`, `entropy_by_method.png` and `length_by_method.png`.

#### `bars(labels, values, title, ylabel, path) -> Path`

```python
def bars(labels: list[str], values: list[float], title: str, ylabel: str, path: str | Path) -> Path:
```

A simple bar chart, one slate-blue bar per label, x labels rotated 30 degrees. The method comparison uses it for the tournament charts (`tournament_mean_score.png`, `tournament_win_rate.png`, `tournament_mean_survival.png`, `tournament_mean_kills.png`), `lines_of_code.png` and `train_seconds.png`.

### Shared learning curves

#### `learning_curve_plots(learning_rows, folder) -> list[Path]`

```python
def learning_curve_plots(learning_rows: list[dict], folder: str | Path) -> list[Path]:
```

The curves every method shares, from `IterationStats.to_row()` rows. Returns an empty list for no rows. Otherwise, with `xs = iteration`:

| File | Series | Title |
| --- | --- | --- |
| `score.png` | `mean_score`, `best_score`, `val_score` | Score per iteration |
| `entropy_shared.png` | `entropy` | Policy entropy |
| `game_length.png` | `mean_length` | Average game length (learner survival) |
| `win_rate_shared.png` | `win_rate` | Win rate |
| `score_vs_time.png` | `mean_score` against `cumulative_seconds` | Score against wall-clock time |
| `curriculum.png` | `opponents` | Curriculum: opponents per iteration |
| `score.gif` | `mean_score` and `val_score` growing | Score per iteration |

The `_shared` suffixes keep these from overwriting the REINFORCE set's `entropy.png` and the imitation set's `win_rate.png` in the same folder. `save_run` calls this after `training_run_plots`.

### Bundles

#### `training_run_plots(history_rows, summaries, folder, method) -> list[Path]`

Writes every chart a training run should have and returns the paths. With no history rows it returns an empty list.

The x axis key is `"generation"` for `method == "genetic"`, `"iteration"` for `"neat"`, and `"epoch"` otherwise. Then the performance charts depend on `method`:

| Method | Files | Row keys read |
| --- | --- | --- |
| `"neat"` | `neat_structure.png` (species and hidden nodes, titled "NEAT structure"), `fitness.png` (best, mean, validation, titled "Fitness by generation") | `extra_species`, `extra_hidden_nodes` (with `.get`, default 0), `best_score`, `mean_score`, `val_score` |
| `"imitation"` | `losses.png`, `accuracy.png`, `survival.png`, `win_rate.png`, `losses.gif` | `train_loss`, `val_loss`, `train_accuracy`, `val_accuracy`, `val_survival`, `val_win_rate` |
| `"genetic"` | `fitness.png` (best, mean, validation), `fitness.gif` | `best_fitness`, `mean_fitness`, `val_fitness` |
| anything else (REINFORCE, PPO) | `reward.png`, `losses.png`, `entropy.png`, `survival.png`, `win_kill_rate.png`, `reward.gif` | `train_return`, `val_return`, `policy_loss`, `value_loss`, `entropy`, `train_survival`, `val_survival`, `win_rate`, `val_win_rate`, `kill_rate` |

The NEAT branch reads the shared `IterationStats` keys, because `NeatTrainer.history_rows()` returns those rows. It has no GIF of its own; `score.gif` from `learning_curve_plots` covers it.

After that, for every method: `timing.png`, and, when `summaries` is not empty, `action_distribution_over_training.png`, `death_needs_over_training.png`, `behaviour_over_training.png`, then every `behaviour_plots` file for the **last** summary.

#### `behaviour_plots(summary, folder) -> list[Path]`

Every behaviour chart for one summary: `action_distribution.png`, `actions_by_thirst.png`, `actions_by_hunger.png`, `actions_by_health.png`, `instinct_curves.png`, `consumption_timing.png`, `fight_or_flight.png`, `proximity_vs_remaining.png`, `actions_by_remaining.png`, `position_heatmap.png`, `armed_vs_unarmed_heatmaps.png`, `deaths_by_cause.png`.

#### `batch_plots(eliminations, players, games, folder, width=120, height=120) -> list[Path]`

Every chapter 3 chart for a batch, plus `death_heatmap.png` when there was at least one elimination.

### Which input feeds which chart

| Input | Where it comes from | Charts |
| --- | --- | --- |
| `eliminations`, `players`, `games` tables | `Runner.run()` return values, or `pd.read_csv` on the CSV files | the eight chapter 3 charts, `batch_plots` |
| One telemetry summary | `BehaviorTelemetry.summary()`, `telemetry.json`, `stats.telemetry` on a trainer step | the eight behaviour charts, `heatmap`, `armed_vs_unarmed`, `behaviour_plots` |
| A list of summaries, one per step | `[s.telemetry for s in trainer.history]` or a sweep's `summary.json["telemetry"]` | `stacked_area_over_training`, `death_needs_over_training`, `behaviour_metrics_over_training` |
| History rows | `trainer.history_rows()` or `history.json` | `timing`, and the performance `curves` inside `training_run_plots` |
| Learning rows | `[s.to_row() for s in trainer.learning_history]` or `learning.json` | `learning_curve_plots` |
| Several runs' learning rows | a comparison's `summary.json["learning"]` | `overlay_curves` |
| Labels and values | a comparison's `results.csv` or tournament | `bars` |
| Any `xs` and `series` | anything you like | `curves`, `curve_gif` |

### Reading the figures

- Percent axes are percent of decisions in that row, not of all decisions.
- Heatmaps always show a share of time, so two heatmaps from batches of different sizes are comparable.
- x axes that describe the field run from "most alive" on the left to "final few" on the right.
- Training charts use the step index on x, never wall-clock time, except `score_vs_time.png` and the comparison's `score_by_time.png`. Use `timing.png` to convert.
- A GIF and a PNG with the same stem show the same data; the GIF is for slides.
- `score.png` is the one chart that means the same thing for every method: the learner's episode return. Compare methods there, not on `fitness.png` or `reward.png`.

## How to use it / experiment

From a saved run folder, without rerunning anything:

```python
import json
from hunger_games.research import plots

rows = json.load(open("results/rl_20260902_150000/learning.json"))
plots.curves([r["iteration"] for r in rows], {"entropy": [r["entropy"] for r in rows]},
             "Policy entropy", "iteration", "nats", "paper/entropy.png")
```

Overlay several methods from their `learning.json` files:

```python
series = {}
for name in ("imitation", "genetic", "neat", "reinforce", "ppo"):
    rows = json.load(open(f"results/{name}_20260902/learning.json"))
    series[name] = ([r["iteration"] for r in rows], [r["val_score"] for r in rows])
plots.overlay_curves(series, "Validation score by method", "iteration", "score", "paper/val_by_method.png")
```

Tournament bars from a comparison's `summary.json`:

```python
data = json.load(open("results/comparison_20260902/summary.json"))
names = list(data["tournament"])
plots.bars(names, [data["tournament"][n]["win_rate"] for n in names], "Tournament win rate", "rate", "paper/wins.png")
```

From a sweep's `summary.json`, one behaviour chart per value:

```python
data = json.load(open("results/chaos_20260902_153000/summary.json"))
for row, summary in zip(data["rows"], data["telemetry"]):
    plots.need_action_curves(summary, f"paper/instinct_chaos_{row['value']}.png")
```

A chart that does not exist yet, in the house style:

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

## Gotchas

- Importing this file switches the whole process to the `Agg` backend. Do not import it in code that wants to open a matplotlib window.
- `curve_gif` needs pillow. Without it `animation.save` raises an error.
- `training_run_plots` trusts `method`. Rows from one trainer with another trainer's method name raise `KeyError` on the first missing key. Only `"imitation"`, `"genetic"` and `"neat"` are matched exactly; every other string gets the REINFORCE set.
- `learning_curve_plots` needs every shared key in every row. Rows from an old `history.json` (which lack `iteration` and `mean_score`) raise `KeyError`; use `learning.json`.
- `overlay_curves` draws nothing and no legend when every series is empty; it still writes a blank figure.
- `bars` with very long labels overlaps them even at a 30-degree rotation. Shorten variant names.
- `training_run_plots` writes the detailed behaviour charts for the last summary only.
- `stacked_area_over_training` uses the row index as x, not `generation` or `epoch`.
- `death_heatmap` bins with `range=[[0, height], [0, width]]`, so pass the real arena size.
- Charts made from a summary with zero decisions come out empty rather than raising.
- `timing` falls back to the row index for NEAT rows, whose step key is `iteration`; the bars still line up because iterations start at 0 and are consecutive.
