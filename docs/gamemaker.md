# `gamemaker.py`

**Source:** [hunger_games/gamemaker.py](../hunger_games/gamemaker.py)
**Depends on:** [arena.py](arena.md) (`Arena`, for `radius` and `distance_from_center`), [config.py](config.md) (`SimulationConfig`).
**Used by:** [game.py](game.md) only, which builds one per game, calls `update`, `is_lethal` and `hazard_distance` every tick, reads `shrinking` for the perception, `interventions` for the results, and the constants `DAMAGE_PER_TICK` and `WEAPON_LABEL`. [recorder.py](recorder.md) captures `safe_radius` and `is_active` into each frame so [renderer.py](renderer.md) and `ui/canvas.py` can draw the circle.

## Purpose

Chapter 1 of the video complains that Seneca Crane keeps herding tributes together, and chapter 3 counts those interventions as a category of elimination. `Gamemaker` reproduces that. It watches for quiet stretches with no eliminations and then shrinks a safe circle toward the centre. Anyone outside loses health every tick.

The circle is on by default but slow. The reason is measured, not aesthetic. Over 20 seeded games with a strict 24-day cutoff:

| Setting | Games ending with a victor | Game maker deaths |
| --- | --- | --- |
| Neither the circle nor the endgame instinct | 0 / 20 | none |
| The slow circle alone (default) | 19 / 20 | 4% of eliminations |
| The endgame instinct alone | 18 / 20 | none |

Without some push, the last two tributes wander opposite corners until time runs out. The circle ends almost every game while killing very rarely, so it became the default. The instinct (`endgame_instinct`) is the alternative for anyone who wants to test whether the ring layout can finish games without help.

## Concepts you need

**Radius, not a box.** The safe area is a circle measured from the arena centre. A cell is lethal when `distance_from_center > safe_radius`.

**Ticks versus days.** Config values are in days; this class converts them to ticks once in `__init__` using `ticks_per_day`.

**Commitment.** Once the circle starts closing it keeps going for at least one day even if someone dies. Otherwise one lucky kill would cancel the intervention and the games would stall again.

**State in a class.** `safe_radius`, `shrinking`, `shrink_until_tick` and `interventions` change over time. `update` is the only method that changes them.

## Walkthrough

### `Gamemaker` class constants

| Constant | Value | Meaning |
| --- | --- | --- |
| `DAMAGE_PER_TICK` | `0.08` | Health lost per tick outside the circle. About 12 ticks from full health to death. |
| `MIN_SAFE_RADIUS` | `8.0` | The circle never shrinks below this, leaving room for a final fight. |
| `WEAPON_LABEL` | `"arena hazard"` | The `weapon` text written to the eliminations CSV. |

`Game._environment_tick` applies `DAMAGE_PER_TICK` itself; the game maker only says where the edge is.

### `Gamemaker.__init__`

```python
def __init__(self, config: SimulationConfig, arena: Arena) -> None
```

| Attribute | Value | Meaning |
| --- | --- | --- |
| `enabled` | `config.gamemaker_enabled` | Whether interventions are allowed at all. |
| `quiet_ticks` | `int(quiet_days_before_intervention * ticks_per_day)` | Ticks of silence that trigger an intervention (24 by default). |
| `arena` | the arena | For distance checks. |
| `safe_radius` | `arena.radius * 1.5` | Starts covering the whole grid, corners included. |
| `shrink_per_tick` | `arena.radius / (intervention_days * ticks_per_day)` | Cells removed per tick of shrinking. |
| `minimum_shrink_ticks` | `ticks_per_day` | Once triggered, shrink for at least this long. |
| `shrink_until_tick` | `-1` | The tick the current commitment runs to. |
| `interventions` | `0` | How many separate interventions have started. |
| `shrinking` | `False` | Whether the circle is closing right now. |

With defaults (120 by 120 grid) `arena.radius` is 59.0, so `shrink_per_tick` is 59 / 144, about 0.41 cells per tick, and the circle takes six days of shrinking to go from the arena edge to the centre.

### `Gamemaker.update`

```python
def update(self, tick: int, last_elimination_tick: int, alive_count: int) -> None
```

Called once per tick from `Game._environment_tick`, before bars drain. The logic:

1. If disabled, or one or zero tributes remain: `shrinking = False`, return.
2. `quiet_for = tick - last_elimination_tick`.
3. If `quiet_for < quiet_ticks` and no commitment is running (`tick >= shrink_until_tick`): `shrinking = False`, return.
4. Otherwise, if not already shrinking, this is a new intervention: `interventions += 1`, `shrinking = True`, `shrink_until_tick = tick + minimum_shrink_ticks`.
5. `safe_radius = max(MIN_SAFE_RADIUS, safe_radius - shrink_per_tick)`.

So the circle closes while the games are quiet, and for at least one day after each trigger. When a kill lands and the commitment expires, shrinking pauses where it is; it never grows back.

### `Gamemaker.is_active`

```python
@property
def is_active(self) -> bool
```

`safe_radius < arena.radius * 1.5`: has the circle ever shrunk? The recorder stores this as `circle_visible` so the renderer only draws a circle once there is one.

### `Gamemaker.hazard_distance`

```python
def hazard_distance(self, x: int, y: int) -> float
```

`safe_radius - distance_from_center(x, y)`. Positive inside, negative outside. `Game.step` passes it into `Player.perceive`, where the vector clamps it to -1..1 in units of vision radius, so tributes see the fog coming before it reaches them.

### `Gamemaker.is_lethal`

```python
def is_lethal(self, x: int, y: int) -> bool
```

`distance_from_center(x, y) > safe_radius`. Used both for the perception's `in_danger_zone` and for applying damage.

### A worked timeline with defaults

Default settings: 120 by 120 grid, `arena.radius` 59.0, `ticks_per_day` 24, `quiet_days_before_intervention` 1.0, `intervention_days` 6.0. Suppose the last elimination happened on tick 100.

| Tick | `quiet_for` | What `update` does |
| --- | --- | --- |
| 101 to 123 | 1 to 23 | Below `quiet_ticks` (24) and no commitment: `shrinking` stays `False`. |
| 124 | 24 | First intervention: `interventions` becomes 1, `shrinking` is `True`, `shrink_until_tick` is 148. Radius drops by about 0.41. |
| 125 to 147 | 25 to 47 | Keeps shrinking whether or not a kill lands (the commitment). |
| 148 onward | | If someone died at tick 140, `quiet_for` is small and the commitment has expired: shrinking pauses. Otherwise it continues, one radius step per tick. |

From 88.5 (1.5 times the radius) to the 8-cell minimum is about 80.5 cells, or roughly 196 ticks of shrinking: a little over eight days if the games stay quiet the whole time. In practice kills interrupt it and the count in `interventions` records each restart.

## How to use it / experiment

Watch the circle close in a quiet game:

```python
from hunger_games.config import SimulationConfig
from hunger_games.game import Game

config = SimulationConfig(width=60, height=60, seed=11, max_days=10, intervention_days=2.0)
game = Game(config)
while not game.is_over:
    game.step()
    if game.tick % 24 == 0:
        gm = game.gamemaker
        print(f"day {game.day_number}: radius {gm.safe_radius:.1f}, shrinking={gm.shrinking}, interventions={gm.interventions}")
print(game.result().interventions, "interventions")
```

Experiments worth running:

- `gamemaker_enabled=False, endgame_instinct=True` reproduces the "instinct alone" row of the table above.
- Halve `intervention_days` and check how the share of `gamemaker` eliminations changes in `eliminations.csv` (see [analysis.md](analysis.md)).
- Raise `quiet_days_before_intervention` to 2.0 to see whether tributes finish the games themselves before the circle starts.
- Sweep `intervention_days` with `research/experiments.py` and plot victors per batch.

## Gotchas

- The command line (`python -m hunger_games`) sets `gamemaker_enabled` from the `--gamemaker` flag, so the circle is off there unless you pass the flag, even though the config default is `True`. The dashboard and direct `SimulationConfig()` use the default.
- `safe_radius` starts at 1.5 times the arena radius. On the square open field the corners are about 1.41 radii out, so the first shrink bites within about half a day. On the round arena nothing is outside the circle until it has shrunk by half a radius, which takes three days of shrinking at the default `intervention_days`.
- `last_elimination_tick` starts at 0. If the bloodbath produces no kill in the first day, the circle starts on tick 24.
- The circle never widens. Once shrunk, that ground is lost for the rest of the game.
- Hazard damage is applied in `Game._environment_tick`, and a tribute killed there is recorded with method `gamemaker` and weapon `arena hazard`, before the natural-causes check.
- `MIN_SAFE_RADIUS` is 8 cells regardless of arena size. On a tiny test arena that is a large share of the grid.
- `update` runs with the tick number before `Game.step` increments it, so an intervention that starts "on tick 124" is visible to tributes from the perception built on tick 125.
