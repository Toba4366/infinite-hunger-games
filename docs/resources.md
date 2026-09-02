# `resources.py`

**Source:** [hunger_games/resources.py](../hunger_games/resources.py)
**Depends on:** [config.py](config.md) (`LayoutName`), `numpy`, and the standard library (`abc`, `enum`, `math`, `typing`). It mentions `Arena` from [arena.py](arena.md) in type hints only.
**Used by:** [arena.py](arena.md) (`ResourceGrid`), [game.py](game.md) (`ResourceKind`, `build_layout`, `weapon_name`), [player.py](player.md) (`ResourceKind`, `weapon_reach`), [perception.py](perception.md), [brain/voting.py](brain/voting.md), [renderer.py](renderer.md), the dashboard (`ui/app.py`, `ui/session.py`, `ui/canvas.py`), and the tests (`test_arena.py`, `test_scenario.py`, `test_brains.py`, `test_ui_session.py`).

## Purpose

`resources.py` is everything about supplies: what kinds exist, how they are stored on the map, how far a weapon can strike, and how the game makers scatter loot before the games begin. Chapter 4 of the video tracks resources as three grids laid over the map rather than as a list of items: what type of thing is in each cell, how many, and how good. `ResourceGrid` is exactly those three grids.

The two layout classes are the two arena designs from the video. `CornucopiaLayout` is the original 74th games: a golden horn in the middle stuffed with weapons and food, almost nothing elsewhere, podiums in a tight ring around it. `RingLayout` is the redesign from chapters 2 and 4: lots of poor supplies around the edge, better and rarer supplies toward the centre, a cache of top weapons in the middle, and podiums on the outer edge so everyone must travel inward.

One design choice cuts across both layouts: medicine is rare. The Cornucopia pile is 5% medkits and the ring gives a non-weapon cell only a 3% chance of one. The video points out that in the arena even a cut is deadly, and in the films help comes by parachute rather than from a chest on the ground. So medkits on the map are a lucky find, and [sponsors.md](sponsors.md) is the main way a wounded tribute heals.

In the pipeline, `Arena.__init__` creates an empty `ResourceGrid`. `Game.__init__` calls `build_layout(config.layout)` and `layout.apply(arena, rng)` to fill it (unless a scenario turns layout loot off), then `layout.spawn_positions(arena, count)` to place the podiums. During the game `Player` calls `arena.resources.peek` and `take`, and the renderer calls `cells_of_kind` to draw icons.

## Concepts you need

**IntEnum.** `ResourceKind` members are integers, so they fit in a numpy grid. `int(ResourceKind.FOOD)` is `1` and `ResourceKind(1)` goes back.

**Parallel arrays.** Cell `(x, y)` is described by `kind[y, x]`, `quantity[y, x]` and `quality[y, x]` together. Three grids of the same shape are faster and simpler to draw than one grid of objects.

**numpy `[y, x]` indexing.** Grids are indexed row first. Every method takes `(x, y)` and does `[y, x]` inside so callers never have to remember.

**Abstract base classes.** `ResourceLayout` inherits from `ABC` and marks `apply` and `spawn_positions` with `@abstractmethod`. Any subclass must implement both, or Python refuses to instantiate it. `Game` uses either layout without knowing which one it has. This is polymorphism.

**`TYPE_CHECKING`.** `arena.py` imports this file, so importing `arena.py` back would be circular. The `if TYPE_CHECKING:` block runs only for type checkers, and the hint is written as the string `"Arena"`.

**`np.random.Generator`.** `rng.random()` gives a float in `0..1`, `rng.uniform(a, b)` a float in `a..b`, `rng.integers(a, b)` an int from `a` to `b - 1`, and `rng.normal(mean, sd)` a bell-curve sample. Same seed, same rolls.

**Trigonometry.** A point at `angle` on a circle of radius `r` is `(cos(angle) * r, sin(angle) * r)`. Spreading `count` points evenly means angles of `2 * pi * i / count`.

**Factory function.** `build_layout` maps a setting to a class and constructs it.

## Walkthrough

### `ResourceKind`

```python
class ResourceKind(IntEnum):
    NONE = 0
    FOOD = 1
    WEAPON = 2
    MEDICINE = 3
```

| Member | Value | Meaning |
| --- | --- | --- |
| `NONE` | `0` | Nothing here. Zero so a fresh grid is empty. |
| `FOOD` | `1` | Rations. Eating one restores some of the hunger bar. |
| `WEAPON` | `2` | A weapon. Its quality decides how deadly it is and how far it reaches. |
| `MEDICINE` | `3` | A medkit. Using one restores some of the health bar. |

These integer values are also what `LootSpec.kind` stores in a saved scenario (see [scenario.md](scenario.md)).

### `WEAPON_TIERS`

```python
WEAPON_TIERS = [
    (0.0, "fists"),
    (0.2, "rock"),
    (0.4, "knife"),
    (0.6, "spear"),
    (0.8, "sword"),
    (0.9, "bow"),
]
```

`(minimum_quality, name)` pairs from worst to best. It must stay sorted for `weapon_name` to work.

### `weapon_name`

```python
def weapon_name(quality: float) -> str
```

Walks up `WEAPON_TIERS`, remembering the name of every tier the quality reaches. The last one remembered wins.

| Quality | Name |
| --- | --- |
| below `0.2` | fists |
| `0.2` to below `0.4` | rock |
| `0.4` to below `0.6` | knife |
| `0.6` to below `0.8` | spear |
| `0.8` to below `0.9` | sword |
| `0.9` and up | bow |

```python
from hunger_games.resources import weapon_name
print(weapon_name(0.0), weapon_name(0.65), weapon_name(0.95))  # fists spear bow
```

### `weapon_reach`

```python
def weapon_reach(quality: float) -> int
```

How many cells away a weapon can strike. Two `if` checks from the top down.

| Quality | Reach | Weapons |
| --- | --- | --- |
| `0.9` and up | `3` | bow |
| `0.6` to below `0.9` | `2` | spear, sword |
| below `0.6` | `1` | fists, rock, knife |

`Player.reach` is a property that returns `weapon_reach(self.weapon_quality)`, and `Player.attack` uses it to decide whether a target is close enough. A bow lets a tribute strike before a knife-holder can close the distance, which is why the central weapon cache matters so much.

### `ResourceGrid`

```python
class ResourceGrid:
    def __init__(self, width: int, height: int) -> None
```

Creates three empty grids of shape `(height, width)`:

| Attribute | dtype | Meaning |
| --- | --- | --- |
| `kind` | `np.int8` | A `ResourceKind` value per cell. |
| `quantity` | `np.int16` | How many of that thing. |
| `quality` | `float` | How good, `0.0` to `1.0`. |

### `ResourceGrid.place`

```python
def place(self, x: int, y: int, kind: ResourceKind, quantity: int, quality: float) -> None
```

Writes a stack into a cell, replacing whatever was there. Quality is clamped to `0..1` with `np.clip`. This is what layouts and hand-placed scenario loot both call.

### `ResourceGrid.has`

```python
def has(self, x: int, y: int) -> bool
```

True if the cell's kind is anything other than `NONE`.

### `ResourceGrid.peek`

```python
def peek(self, x: int, y: int) -> tuple[ResourceKind, int, float]
```

Reads a cell without changing it and returns `(kind, quantity, quality)` as plain Python types.

### `ResourceGrid.take`

```python
def take(self, x: int, y: int) -> tuple[ResourceKind, int, float]
```

Calls `peek`, then zeroes all three grids at that cell, then returns what was there. `Player.pick_up` uses this.

### `ResourceGrid.cells_of_kind`

```python
def cells_of_kind(self, kind: ResourceKind) -> tuple[np.ndarray, np.ndarray]
```

Returns `(xs, ys)` arrays of every cell holding `kind`, via `np.nonzero`. Used by the renderer to scatter icons in one call.

### `ResourceLayout`

```python
class ResourceLayout(ABC):
    @abstractmethod
    def apply(self, arena: "Arena", rng: np.random.Generator) -> None
    @abstractmethod
    def spawn_positions(self, arena: "Arena", count: int) -> list[tuple[int, int]]
```

The contract. `apply` fills `arena.resources`; `spawn_positions` returns one podium per player.

### `CornucopiaLayout`

```python
class CornucopiaLayout(ResourceLayout):
    PILE_RADIUS = 4
    PODIUM_RADIUS = 10
```

| Constant | Value | Meaning |
| --- | --- | --- |
| `PILE_RADIUS` | `4` | Cells within this distance of the centre are the pile. |
| `PODIUM_RADIUS` | `10` | Distance from the centre to the podium circle. |

### `CornucopiaLayout.apply`

```python
def apply(self, arena: "Arena", rng: np.random.Generator) -> None
```

Visits every cell, skipping anything that is not dry land (`arena.is_land`). Inside the pile (`distance <= PILE_RADIUS`) one roll picks the item:

| Roll | Share | Item |
| --- | --- | --- |
| below `0.5` | 50% | One weapon, quality `0.6..1.0`. |
| `0.5` to below `0.95` | 45% | Food, `4..8` rations, quality `0.6..1.0`. |
| `0.95` and up | 5% | One medkit, quality `0.6..1.0`. |

Outside the pile each land cell has a 2% chance of `1..2` poor rations (quality `0.1..0.3`). Note that the pile ignores water only because water cells are skipped first; a lake at the centre makes the pile smaller.

### `CornucopiaLayout.spawn_positions`

```python
def spawn_positions(self, arena: "Arena", count: int) -> list[tuple[int, int]]
```

Spreads `count` podiums evenly around a circle of radius `PODIUM_RADIUS`, then passes each through `arena.snap_to_podium`. That method honours `config.allow_water_podiums`: with the default `True` a podium may stand in water, otherwise it is nudged to the nearest dry cell (see [arena.md](arena.md)).

### `RingLayout`

```python
class RingLayout(ResourceLayout):
    CENTER_FRACTION = 0.06
```

| Constant | Value | Meaning |
| --- | --- | --- |
| `CENTER_FRACTION` | `0.06` | Cells closer than 6% of the arena radius count as the centre cache. |

### `RingLayout.apply`

```python
def apply(self, arena: "Arena", rng: np.random.Generator) -> None
```

For every land cell, `distance = arena.normalized_distance_from_center(x, y)` runs from `0.0` at the middle to `1.0` at the edge. Then:

1. **Centre cache** (`distance < 0.06`): a 50% chance of one weapon with quality `0.8..1.0` (swords and bows). Nothing else spawns here.
2. **Density**: `density = 0.015 + 0.09 * distance`. That is 1.5% of cells near the centre and 10.5% at the edge. If `rng.random() >= density` the cell stays empty.
3. **Quality**: `clip(1.0 - distance + normal(0, 0.08), 0.05, 1.0)`. High near the centre, low near the edge, with a little wobble.
4. **Weapon roll**: `weapon_chance = (1.0 - distance) ** 2`. Near the centre almost every occupied cell is a weapon; at the edge almost none. The square makes the drop-off steep.
5. **Otherwise medicine**: a 3% roll for one medkit.
6. **Otherwise food**: `1 + int(4 * distance)` rations, so one at the centre and up to five at the edge.

This gives exactly the shape the video asks for: quantity rises toward the edge, quality rises toward the centre, and healing is scarce everywhere.

### `RingLayout.spawn_positions`

```python
def spawn_positions(self, arena: "Arena", count: int) -> list[tuple[int, int]]
```

Delegates to `arena.edge_positions(count)`, because the arena knows whether it is round or rectangular.

### `build_layout`

```python
def build_layout(name: LayoutName) -> ResourceLayout
```

Looks the enum up in a dictionary and constructs the matching class. Adding a layout means one new class and one new dictionary line.

## How to use it / experiment

Count what a layout produced:

```python
import numpy as np
from hunger_games.config import SimulationConfig
from hunger_games.game import Game
from hunger_games.resources import ResourceKind

game = Game(SimulationConfig(seed=5))
kinds = game.arena.resources.kind
for kind in (ResourceKind.FOOD, ResourceKind.WEAPON, ResourceKind.MEDICINE):
    print(kind.name, int((kinds == int(kind)).sum()))
```

Peek and take by hand:

```python
grid = game.arena.resources
xs, ys = grid.cells_of_kind(ResourceKind.WEAPON)
x, y = int(xs[0]), int(ys[0])
print(grid.peek(x, y))
print(grid.take(x, y), grid.has(x, y))  # (...), False
```

Compare the two layouts on the same seed:

```python
from hunger_games.config import LayoutName
for layout in LayoutName:
    game = Game(SimulationConfig(seed=5, layout=layout))
    print(layout.value, int((game.arena.resources.kind == 3).sum()), "medkits")
```

Write your own layout by subclassing `ResourceLayout`, implementing both methods, and adding it to the dictionary in `build_layout` (plus a new `LayoutName` member).

## Gotchas

- **Medicine is rare on purpose.** Do not "fix" the 5% and 3% figures without also thinking about [sponsors.md](sponsors.md); `test_medicine_is_rare_in_layouts` checks that medkits are under 6% of all loot.
- **`place` overwrites.** Placing food on a cell that held a weapon loses the weapon. Scenario loot placed on top of layout loot does exactly this.
- **Layouts only spawn on dry land.** Water cells and `VOID` are skipped, but *podiums* may still be in water when `allow_water_podiums` is `True`.
- **`rng.integers(4, 9)` is `4..8`.** The upper bound is exclusive, as with `range`.
- **Chapter 4's quality has a wobble.** `rng.normal(0.0, 0.08)` can push quality below zero, which is why it is clipped to at least `0.05`.
- **Layout randomness comes from the game's generator.** Every `rng.random()` call advances the shared stream, so changing the arena size changes every later roll in the game too.
- **`weapon_reach` and `weapon_name` use different cut-offs.** A sword (`0.8`) and a spear (`0.6`) both reach `2`; only the bow reaches `3`.
