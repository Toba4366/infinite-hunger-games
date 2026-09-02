# `output/` folder

**Location:** [output/](../output/)
**Written by:** [runner.md](runner.md) (`Runner.save`), [analysis.md](analysis.md) (`make_report`), [renderer.md](renderer.md) (`Renderer.save`, `Renderer.snapshot`)
**Read by:** [analysis.md](analysis.md) (`load_results`), and you, with pandas or a spreadsheet

## Purpose

This folder is the "spreadsheet for future analysis" that chapter 3 of the
video wishes it had. Every game the runner plays is broken down into rows,
and the analysis command turns those rows into the same charts the video
draws from its three real games. Nothing in the folder is needed to run the
simulator; delete it and the next `simulate` command recreates it.

The folder is the default target of `python -m hunger_games simulate` and
the default source of `python -m hunger_games analyze`. Both accept
`--output DIR` so you can keep several experiments side by side, for
example `output_ring` and `output_cornucopia`.

## What is in it

| File | Made by | What it is |
| --- | --- | --- |
| `eliminations.csv` | `simulate` | One row per death across every game in the batch. |
| `players.csv` | `simulate` | One row per tribute per game. |
| `games.csv` | `simulate` | One row per game. |
| `gifts.csv` | `simulate` | One row per sponsor parachute. |
| `report.png` | `analyze` | Six charts in one image, for a quick look. |
| `plots/*.png` | `analyze` | The same charts one PNG each, plus a death heatmap and deaths by district, for papers. |
| `plots/behaviour/*.png` | `analyze` | Behaviour charts, written when `telemetry.json` exists. |
| `telemetry.json` | `simulate` with telemetry | Merged behaviour telemetry of the batch (see [research/telemetry.md](research/telemetry.md)); the runner writes it when asked for telemetry, and sweeps always do. |
| `sample_round_ring.gif` | `watch --save` | A 150-frame animation of a round-arena ring-layout game, seed 7. |
| `snapshot_open_start.png` | `Renderer.snapshot` | Frame zero of an open-field ring-layout game, seed 4. |
| `snapshot_cornucopia_start.png` | `Renderer.snapshot` | Frame zero of an open-field Cornucopia game, seed 8. |

The four CSV files are always overwritten together by one `simulate`
run, so they always describe the same batch. The shipped set is 200 games
of the default configuration (open field, ring layout, chaos 0.5, sponsors
on, game makers off, seed 1000, so game N used seed 1000 + N).

## `eliminations.csv`

One row per elimination. The columns come from the `Elimination`
dataclass in [records.md](records.md).

| Column | Type | Meaning |
| --- | --- | --- |
| `game_id` | int | Which game in the batch, starting at 0. |
| `day` | int | In-game day of the death, starting at 1. |
| `tick` | int | Exact simulation tick (24 ticks per day by default). |
| `victim_id` | int | The dead tribute's id, 0 to 23. |
| `victim_name` | text | Their display name, for example `Tribute 18 (D9)`. |
| `victim_district` | int | Their district, 1 to 12. |
| `victim_training_score` | int | Their 1 to 12 training score. |
| `method` | text | One of `player_vs_player`, `gamemaker`, `natural_causes`. Chapter 3's three categories. |
| `weapon` | text | The weapon name for fights (`fists`, `rock`, `knife`, `spear`, `sword`, `bow`), `arena hazard` for the game makers, `dehydration`, `starvation` or `untreated wound` for natural causes. |
| `killer_id` | float or empty | The killer's id for fights. Empty otherwise. pandas reads the column as float because of the blanks. |
| `killer_name` | text or empty | The killer's name for fights. |
| `x`, `y` | int | Where the death happened, as grid column and row. |
| `placement` | int | The victim's final placing. The first tribute out of 24 places 24th, the runner-up places 2nd. |

## `players.csv`

One row per tribute per game. The columns come from the `PlayerResult`
dataclass in [records.md](records.md).

| Column | Type | Meaning |
| --- | --- | --- |
| `game_id` | int | Which game. |
| `player_id` | int | The tribute's id within that game. |
| `name` | text | Display name. |
| `district` | int | District, 1 to 12. |
| `sex` | text | `F` or `M`. |
| `training_score` | int | 1 to 12. |
| `survival_score` | float | 0.05 to 0.95, the fixed hunting aptitude. |
| `brain` | text | Which brain drove them: `voting`, `random` or `neural`. |
| `favor` | float | The sponsors' final opinion of them, 0.0 to 1.0. |
| `gifts_received` | int | How many parachutes they received. |
| `placement` | int | Final placing, 1 for the victor. Survivors of a draw share the same number. |
| `kills` | int | How many tributes they eliminated. |
| `days_survived` | float | Death tick divided by ticks per day, or the game length if they survived. |
| `cause_of_death` | text or empty | Same vocabulary as `weapon` above. Empty if alive at the end. |
| `alive_at_end` | bool | `True` for the victor and for survivors of a draw. |

## `games.csv`

One row per game.

| Column | Type | Meaning |
| --- | --- | --- |
| `game_id` | int | Position in the batch. |
| `seed` | int | The exact seed used, so any single game can be replayed with `--seed`. |
| `days` | int | How many days it lasted. |
| `ticks` | int | How many ticks it lasted. |
| `winner_id` | float or empty | The victor's id. Empty for a draw (time ran out with more than one alive). |
| `winner_name` | text or empty | The victor's name. |
| `interventions` | int | How many separate times the game makers started shrinking the circle. |

## `gifts.csv`

One row per parachute. The columns come from `SponsorGift` in
[sponsors.md](sponsors.md).

| Column | Type | Meaning |
| --- | --- | --- |
| `game_id` | int | Which game. |
| `day` | int | The day the parachute landed (gifts arrive at the start of a day). |
| `tick` | int | The exact tick. |
| `player_id` | int | Who received it. |
| `player_name` | text | Their name. |
| `kind` | text | `medicine`, `food` or `water`. |
| `favor` | float | The receiver's sponsor favour at the time. |

## `report.png`

Six charts drawn by `make_report` in [analysis.md](analysis.md), each
matching a question chapter 3 of the video asks of its data.

| Position | Chart | Video question |
| --- | --- | --- |
| top left | Eliminations per day, averaged per game | Does the curve follow chapter 2's exponential decay? |
| top middle | Eliminations by method | How much of the killing do the game makers do? |
| top right | Weapons used in player-vs-player deaths | Is there a trend in weapons? |
| bottom left | Average placing by training score, better toward the top | Do high scorers last longer? |
| bottom middle | Average kills by training score | Do high scorers get more eliminations? |
| bottom right | Game length in days | How long do the games run? |

The `analyze` command also prints the "eliminations per training point"
number the video computes as 0.134 from its three real games.

## Reading the data yourself

```python
import pandas as pd

eliminations = pd.read_csv("output/eliminations.csv")
players = pd.read_csv("output/players.csv")
games = pd.read_csv("output/games.csv")

# Deaths on each day, averaged per game (chapter 2's decay curve).
print(eliminations.groupby("day").size() / len(games))

# Who does the killing: share of each method.
print(eliminations["method"].value_counts(normalize=True))

# Does a high training score help? Mean placing per score (lower is better).
print(players.groupby("training_score")["placement"].mean())

# Replay the longest game of the batch on screen.
longest = games.sort_values("days").iloc[-1]
print(f"python -m hunger_games watch --seed {int(longest.seed)}")
```

Note the last snippet: a replay with `--seed` only matches if you pass the
same shape, layout, size, chaos, brain and day limit the batch used.

## Regenerating everything

```bash
MPLBACKEND=Agg python -m hunger_games simulate --games 200 --workers 4 --seed 1000
MPLBACKEND=Agg python -m hunger_games analyze
MPLBACKEND=Agg python -m hunger_games watch --shape round --seed 7 --speed 2 --save output/sample_round_ring.gif
```

The snapshots were made from Python:

```python
from hunger_games import Game, SimulationConfig, LayoutName
from hunger_games.renderer import Renderer

Renderer(Game(SimulationConfig(seed=4))).snapshot("output/snapshot_open_start.png")
Renderer(Game(SimulationConfig(seed=8, layout=LayoutName.CORNUCOPIA))).snapshot("output/snapshot_cornucopia_start.png")
```

## Gotchas

- Columns with blanks (`killer_id`, `winner_id`) load as floats in pandas, so compare with `== 12.0` or convert with `.astype("Int64")`.
- `placement` in `players.csv` is not unique inside a drawn game: every survivor shares the same number.
- `days` counts the day the game ended on, so a game that ends at tick 251 reports day 11 even though day 11 is not complete.
- The GIF is 2 MB for 150 frames. Saving a full game at `--speed 1` can reach the 600-frame cap and several megabytes.
- `simulate` overwrites all four CSVs without asking. Use `--output` to keep an old batch.
- `gifts.csv` is empty (header only) when the batch ran with `--no-sponsors`.
