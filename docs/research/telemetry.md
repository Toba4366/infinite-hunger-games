# `telemetry.py`

**Source:** [hunger_games/research/telemetry.py](../../hunger_games/research/telemetry.py)
**Depends on:** `numpy`; [../actions.md](../actions.md) (`Action`, `ActionType`); [../game.md](../game.md) (`Game`, for the hook lists); [../perception.md](../perception.md) (`Perception`); [../player.md](../player.md) (`Player`, `Player.WOUND_THRESHOLD`)
**Used by:** [../runner.md](../runner.md) (one collector per game, merged into `telemetry.json`); [../training/genetic.md](../training/genetic.md) (one per evaluation game, merged per generation); [../training/reinforce.md](../training/reinforce.md) (one per episode, learners only, merged per epoch); [experiments.md](experiments.md) (`merge`); [plots.md](plots.md) (bin labels); [../ui/session.md](../ui/session.md) (watched games); [../ui/app.md](../ui/app.md) (`NEED_BIN_LABELS`); `tests/test_research.py`

## Purpose

A win count tells you a brain got better. It does not tell you why. This file tallies every decision a tribute makes against the state it was in: how empty its thirst, hunger and health bars were, whether someone was in sight, where it stood, and what fraction of the field was still alive. It also records where deaths happen, what the bars looked like at death, and how long tributes last after being badly hurt.

`BehaviorTelemetry` plugs into the two hook lists on `Game`. It never changes the game. `summary()` turns its numpy tallies into plain lists so they can be written as JSON, added up across games or CPU workers with `merge()`, and drawn by [plots.md](plots.md).

## Concepts you need

**Hooks.** `Game.step()` calls every function in `game.decision_hooks` right after a brain decides and before the action is carried out, as `hook(player, perception, action)`. At the end of the tick it calls every function in `game.tick_hooks` as `hook(game)`. Appending a bound method to those lists is the whole integration.

**Binning.** A bar level from 0.0 to 1.0 is put into one of five buckets so counts can be compared. `bin_index` walks the edges and returns the bucket number. The top edge is `1.0001` so a full bar (exactly 1.0) lands in the last bucket instead of falling off the end.

**Tallies as 2-D arrays.** `action_by_thirst[bin, action]` is a table with five rows (thirst levels) and nine columns (action kinds). Dividing a row by its total gives "what share of decisions at this thirst level were each action". That is the instinct curve.

**Heatmaps.** The arena is divided into 30 by 30 cells. Every living tracked tribute adds one count to its cell every tick. Rows are `y`, columns are `x`, so `position_heat[row][col]` matches the picture with row 0 at the top.

**Shannon entropy.** With probabilities `p` over the nine actions, entropy is `-sum(p * ln p)` in nats. All nine equally likely gives `ln 9 = 2.20`. Always the same action gives 0. It is a one-number measure of how varied the behaviour is.

**Alive fraction.** `perception.alive_fraction` is living tributes divided by the starting roster. The bins run from "final few" (under 15 percent, so 3 or fewer of 24) up to "most alive" (75 percent or more).

## Walkthrough

### Module constants

| Constant | Value | Meaning |
| --- | --- | --- |
| `ACTION_NAMES` | `["rest", "move", "drink", "eat", "hunt", "pick_up", "heal", "attack", "flee"]` | Column order of every action table, taken from `ActionType` in order |
| `ACTION_INDEX` | `{ActionType.REST: 0, ...}` | Lookup from an action kind to its column |
| `NEED_BIN_EDGES` | `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]` | Five bins for thirst, hunger and health |
| `NEED_BIN_LABELS` | `["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]` | Axis labels for those bins |
| `ALIVE_BIN_EDGES` | `[0.0, 0.15, 0.3, 0.5, 0.75, 1.0001]` | Five bins for the alive fraction |
| `ALIVE_BIN_LABELS` | `["final few", "<30%", "<50%", "<75%", "most alive"]` | Labels: bin 0 is under 15 percent alive, bin 4 is 75 percent or more |
| `HEATMAP_CELLS` | `30` | Cells per side of every heatmap |
| `ARMED_THRESHOLD` | `0.4` | A weapon of at least this quality counts as armed |

Note the `<30%` label means "15 to 30 percent", `<50%` means "30 to 50", and `<75%` means "50 to 75". The labels name the top of each bin.

### `bin_index(value: float, edges: list[float]) -> int`

Walks the edges and returns the first bin where `edges[i] <= value < edges[i + 1]`. A value off the top returns the last bin. A value below zero also returns the last bin, because no edge pair matches; health is clamped to zero before being stored so that never happens in practice.

```python
bin_index(0.35, NEED_BIN_EDGES)   # 1, the "20-40%" bin
bin_index(1.0, NEED_BIN_EDGES)    # 4, thanks to the 1.0001 top edge
```

### `class BehaviorTelemetry`

#### `__init__(self, width: int, height: int, tracked_ids: set[int] | None = None) -> None`

Creates empty tallies for an arena of the given size. `tracked_ids` limits measurement to some player ids; `None` means everyone. The RL trainer passes the learner ids so opponents driven by the hand-coded voting brain do not pollute the counts.

Every tally, its shape, and what it means:

| Attribute | Shape | Meaning |
| --- | --- | --- |
| `action_counts` | `(9,)` | Total decisions per action kind |
| `action_by_thirst` | `(5, 9)` | Decisions split by the thirst bin at the moment of choice |
| `action_by_hunger` | `(5, 9)` | Same, by hunger bin |
| `action_by_health` | `(5, 9)` | Same, by health bin |
| `action_by_alive` | `(5, 9)` | Same, by alive-fraction bin |
| `combat_by_health` | `(5, 2)` | Column 0 attacks, column 1 flees, by health bin, only when someone was in sight |
| `position_heat` | `(30, 30)` | Ticks spent in each cell by living tracked tributes |
| `armed_heat` | `(30, 30)` | The same, tributes with `weapon_quality >= 0.4` only |
| `unarmed_heat` | `(30, 30)` | The same, tributes below the threshold |
| `proximity_sum` | `(5,)` | Sum of distance to the nearest visible tribute, by alive bin |
| `proximity_count` | `(5,)` | How many samples went into each sum |
| `thirst_at_drink` | `(10,)` | Histogram of the thirst bar at every drink, ten bins of 10 percent |
| `hunger_at_eat` | `(10,)` | Hunger bar at every meal |
| `health_at_heal` | `(10,)` | Health bar at every medkit use |
| `death_needs` | `(3,)` | Sum of thirst, hunger and health at the moment of death |
| `death_count` | int | Deaths that went into `death_needs` |
| `deaths_by_cause` | dict | Cause name to count, using `player.cause_of_death` |
| `survival_ticks` | list | Ticks survived, one entry per tracked tribute per game |
| `kills` | list | Kills, one per tracked tribute per game |
| `wins` | list | 1 or 0, one per tracked tribute per game |
| `placements` | list | Placement, one per tracked tribute per game |
| `post_injury_ticks` | list | Ticks lived after first dropping below half health |
| `games` | int | Games that reached `on_game_end` |
| `_dead_seen` | set | Ids already counted as dead this game |
| `_injured_at` | dict | Id to the tick of the first serious injury this game |

#### `attach(self, game: Game) -> BehaviorTelemetry`

Appends `on_decision` to `game.decision_hooks` and `on_tick` to `game.tick_hooks`, then returns `self` so you can write `BehaviorTelemetry(w, h).attach(game)` on one line. One collector can be attached to several games in a row; the per-game bookkeeping resets itself at the end of each.

#### `tracks(self, player: Player) -> bool`

True when `tracked_ids` is `None` or contains `player.player_id`. Every hook checks this first.

#### `on_decision(self, player: Player, perception: Perception, action: Action) -> None`

Called once per decision. In order:

1. Skip untracked players.
2. Look up the action column `a`.
3. Add one to `action_counts[a]` and to the matching cell of `action_by_thirst`, `action_by_hunger`, `action_by_health` and `action_by_alive`, using the bars and alive fraction from the perception.
4. If `perception.nearest_threat` is not `None`, add that tribute's distance to `proximity_sum[alive_bin]` and one to `proximity_count[alive_bin]`. If the action is `ATTACK` add one to `combat_by_health[health_bin, 0]`; if `FLEE`, to column 1.
5. If the action is `DRINK`, `EAT` or `HEAL`, add one to the timing histogram at `min(9, int(bar * 10))`.

The perception is the one the brain saw, so the tallies describe the state at the moment of choice, not after the action landed. An `ATTACK` chosen with nobody in sight (possible for a neural brain) counts in the action tables but not in `combat_by_health`.

#### `on_tick(self, game: Game) -> None`

Called at the end of every tick, after `game.tick` has been advanced. For every tracked player:

- **Alive:** convert `(x, y)` to a heatmap cell with `int(x * 30 / width)` capped at 29, add one to `position_heat` and to `armed_heat` or `unarmed_heat`. If health is below `Player.WOUND_THRESHOLD` (0.5) and this id is not yet in `_injured_at`, record the current tick.
- **Newly dead** (not yet in `_dead_seen`): add the id, add `(thirst, hunger, max(0, health))` to `death_needs`, count the death, bump `deaths_by_cause[cause]` (or `"unknown"`), and if the tribute had been injured, append `tick - injured_at` to `post_injury_ticks`.

Finally, if `game.is_over`, call `on_game_end(game)`.

#### `on_game_end(self, game: Game) -> None`

Counts the game and records one outcome per tracked tribute: `survival_ticks` from `game.death_ticks` (or the current tick for survivors), `kills`, `wins` (1 only for a sole survivor), `placements` (the game's placing, or the shared survivor placing computed as the number of survivors when the game has not assigned it yet), and for injured survivors a `post_injury_ticks` entry. Then it clears `_dead_seen` and `_injured_at` so the collector can serve the next game.

#### `entropy(self) -> float`

Shannon entropy of `action_counts` in nats. Returns 0.0 with no decisions. Zero-count actions are dropped before taking the log so there is no `log(0)`.

#### `summary(self) -> dict`

Every tally converted with `.tolist()` plus derived numbers. This is the format every plot function reads and every JSON file stores.

| Key | Type | Meaning |
| --- | --- | --- |
| `games` | int | Games counted |
| `action_names` | list of 9 str | Column names for the tables below |
| `action_counts` | list of 9 | Total per action |
| `action_by_thirst` | 5 x 9 | Actions by thirst bin |
| `action_by_hunger` | 5 x 9 | Actions by hunger bin |
| `action_by_health` | 5 x 9 | Actions by health bin |
| `action_by_alive` | 5 x 9 | Actions by alive-fraction bin |
| `combat_by_health` | 5 x 2 | Attack and flee counts by health bin |
| `position_heat` | 30 x 30 | Time spent per cell, all tracked tributes |
| `armed_heat` | 30 x 30 | Time spent per cell, armed only |
| `unarmed_heat` | 30 x 30 | Time spent per cell, unarmed only |
| `proximity_sum` | list of 5 | Distance sums by alive bin |
| `proximity_count` | list of 5 | Sample counts by alive bin |
| `thirst_at_drink` | list of 10 | Drink timing histogram |
| `hunger_at_eat` | list of 10 | Eat timing histogram |
| `health_at_heal` | list of 10 | Heal timing histogram |
| `death_needs` | list of 3 | Summed thirst, hunger, health at death |
| `death_count` | int | Deaths counted |
| `deaths_by_cause` | dict | Cause to count |
| `survival_ticks` | list | One per tribute per game |
| `kills` | list | One per tribute per game |
| `wins` | list | One per tribute per game |
| `placements` | list | One per tribute per game |
| `post_injury_ticks` | list | One per injured tribute |
| `entropy` | float | Action entropy in nats |
| `mean_survival_ticks` | float | Mean of `survival_ticks`, 0.0 if empty |
| `win_rate` | float | Mean of `wins` |
| `kill_rate` | float | Mean of `kills` |
| `mean_death_needs` | list of 3 | `death_needs / max(1, death_count)` |

#### `merge(summaries: list[dict]) -> dict` (static)

Adds several summaries into one. Array keys (the fifteen tables and histograms) are summed element by element. The five per-tribute lists are concatenated. `games` and `death_count` are added. `deaths_by_cause` dictionaries are added key by key. Then `entropy`, `mean_survival_ticks`, `win_rate`, `kill_rate` and `mean_death_needs` are recomputed from the merged totals, so they are true totals and not averages of averages. An empty list returns the empty summary of a 1 by 1 collector.

```python
merged = BehaviorTelemetry.merge([result.telemetry for result in results])
```

Merging is what lets the runner combine worker processes, the trainers combine a generation's games, and the sweep combine every value.

## How to use it / experiment

```python
from hunger_games.config import SimulationConfig
from hunger_games.game import Game
from hunger_games.research.telemetry import BehaviorTelemetry, NEED_BIN_LABELS
import numpy as np

game = Game(SimulationConfig(seed=3))
telemetry = BehaviorTelemetry(game.arena.width, game.arena.height).attach(game)
game.run()
s = telemetry.summary()

# P(drink | thirst) by hand, the same maths as plots.need_action_curves.
thirst = np.asarray(s["action_by_thirst"])
drink = s["action_names"].index("drink")
for label, row in zip(NEED_BIN_LABELS, thirst):
    print(label, f"{row[drink] / max(1, row.sum()):.2%}")
```

Things to try:

- Track only the careers: `BehaviorTelemetry(w, h, tracked_ids={0, 1, 2, 3, 6, 7})` (districts 1, 2 and 4 with the default roster).
- Save a summary with `json.dump(s, open("one_game.json", "w"))` and draw it later with any function in [plots.md](plots.md).
- Raise `ARMED_THRESHOLD` in your own subclass to see whether only well-armed tributes head for the centre.
- Compare `mean_death_needs` between two brains. A brain that stops dying of thirst should show a higher thirst bar at death, because deaths shift to other causes.

## Gotchas

- `on_game_end` fires from the tick hook before `Game._finish()` assigns survivors their placing, so the collector computes the survivors' shared placing itself (the number of survivors, 1 for a sole victor). It matches what `game.result()` reports afterwards.
- A game that is over before its first tick never triggers `on_game_end`, because the hook only runs inside `step()`.
- `merge` builds its result from the first summary and adds into that summary's `deaths_by_cause` dictionary in place. Pass copies if you need the originals unchanged.
- The heatmap is 30 by 30 whatever the arena size, so on a 120-cell arena each heat cell is a 4 by 4 block. Plots normalise by the total so the colour means share of time, not ticks.
- Only decisions where `nearest_threat` is not `None` feed `combat_by_health` and the proximity tallies. A tribute that never sees anyone leaves those tables empty.
- `on_tick` records positions after actions resolve and after needs drain, not where the tribute stood when it chose.
- With `tracked_ids=None` the collector measures every tribute, including any driven by the voting brain. In a sweep or a GA run that means part of what you see is hand-coded behaviour, not learned behaviour.
- The summary does not store the arena size. Keep it in `config.json` next to the summary if you need it later.
