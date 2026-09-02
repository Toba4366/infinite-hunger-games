# `terrain.py`

**Source:** [hunger_games/terrain.py](../hunger_games/terrain.py)
**Depends on:** [config.py](config.md) (`TerrainConfig`) and `numpy`.
**Used by:** [arena.py](arena.md), `perception.py`, `brain/voting.py`, `renderer.py`, and `tests/test_arena.py`.

## Purpose

`terrain.py` turns a grid of heights into a grid of ground types. A height map from [noise.py](noise.md) is just numbers between `0.0` and `1.0`. This file decides which of those numbers mean water, which mean sand, which mean grass and which mean rock. That is the "interpret the heights" step of chapter 4 of the video.

The file also holds three small lookup tables that give each terrain type its gameplay meaning and its colour. `HUNT_DIFFICULTY` says how hard it is to find food on each ground. `MOVE_SUCCESS` says how likely a step onto each ground succeeds. `TERRAIN_COLORS` says what colour the renderer paints it.

In the pipeline: `Arena.__init__` in [arena.md](arena.md) calls `classify_heights(heights, config.terrain)` right after generating the height map. The `Arena` then reads `HUNT_DIFFICULTY` and `MOVE_SUCCESS` whenever a player hunts or moves. The renderer reads `TERRAIN_COLORS`. The `Perception` object and the voting brain use `TerrainType` to describe and reason about the ground a player stands on.

The key idea from the video is that thresholds are relative. The water threshold is a fixed number, but sand and grass are given as *sizes* that stack on top of the previous band. Shrink `sand_size` to zero and sand disappears while grass slides down to fill the gap. No other setting needs to change.

## Concepts you need

**IntEnum.** Like an `Enum`, but every member is also an integer. `TerrainType.GRASS == 3` is `True`, and `int(TerrainType.GRASS)` is `3`. This matters because numpy grids store numbers, not Python objects. We store `3` in the grid and turn it back into `TerrainType.GRASS` when we read it.

**Dictionaries keyed by enum members.** `HUNT_DIFFICULTY[TerrainType.WATER]` looks up `0.6`. Because `IntEnum` members hash like their integer values, `HUNT_DIFFICULTY[1]` also works, but the enum form is clearer.

**numpy boolean masks.** `heights < 0.25` produces a grid of `True`/`False` the same shape as `heights`. Writing `terrain[mask] = value` sets every cell where the mask is `True`. This replaces a double `for` loop over every cell with one line.

**`np.full(shape, value, dtype)`.** Makes a new array of the given shape, filled with one value. `dtype=np.int8` stores each cell as a single byte, which is plenty for five terrain types.

**Painting from the top down.** The classifier starts with everything as rock, then overwrites lower and lower bands. The order matters: each later assignment covers a subset of the earlier one.

## Walkthrough

### `TerrainType`

```python
class TerrainType(IntEnum):
    VOID = 0
    WATER = 1
    SAND = 2
    GRASS = 3
    ROCK = 4
```

The kinds of ground a cell can be.

| Member | Value | Meaning |
| --- | --- | --- |
| `VOID` | `0` | Outside the arena entirely. Used to carve a round arena out of the square grid. |
| `WATER` | `1` | Lakes and rivers. Drinkable, hard to hunt in, slow to cross. |
| `SAND` | `2` | Beaches. Poor hunting, nothing to hide behind. |
| `GRASS` | `3` | Meadows and forest. The easiest hunting in the arena. |
| `ROCK` | `4` | Mountains. Very hard hunting, but the high ground. |

The values go from lowest ground (`WATER`) to highest (`ROCK`), which matches the height bands. `VOID` is `0` so that a freshly zeroed grid means "nothing".

```python
from hunger_games.terrain import TerrainType

print(int(TerrainType.GRASS))       # 3
print(TerrainType(1))               # TerrainType.WATER
print(TerrainType.ROCK > TerrainType.SAND)   # True, they compare like ints
```

### `HUNT_DIFFICULTY`

```python
HUNT_DIFFICULTY = {
    TerrainType.VOID: 1.0,
    TerrainType.WATER: 0.6,
    TerrainType.SAND: 0.8,
    TerrainType.GRASS: 0.2,
    TerrainType.ROCK: 0.9,
}
```

How hard it is to catch food on each terrain, from `0.0` (trivial) to `1.0` (impossible).

| Terrain | Difficulty | Note |
| --- | --- | --- |
| `VOID` | `1.0` | Nothing to hunt outside the arena. |
| `WATER` | `0.6` | From chapter 4 of the video. |
| `SAND` | `0.8` | Added by this project. |
| `GRASS` | `0.2` | From chapter 4 of the video. |
| `ROCK` | `0.9` | Added by this project. |

`Player.hunt` compares its survival score against this difficulty. `Arena.hunt_difficulty_at(x, y)` is the usual way to read it.

### `MOVE_SUCCESS`

```python
MOVE_SUCCESS = {
    TerrainType.VOID: 0.0,
    TerrainType.WATER: 0.5,
    TerrainType.SAND: 0.85,
    TerrainType.GRASS: 1.0,
    TerrainType.ROCK: 0.6,
}
```

The chance that a step *into* each terrain succeeds.

| Terrain | Chance | Note |
| --- | --- | --- |
| `VOID` | `0.0` | You can never step outside the arena. |
| `WATER` | `0.5` | Wading is slow: half of all steps fail. |
| `SAND` | `0.85` | Slightly slow. |
| `GRASS` | `1.0` | Always succeeds. |
| `ROCK` | `0.6` | Scrambling is slow. |

This is how chases get resolved. A tribute fleeing across grass outruns one wading through a lake. `Arena.move_success_at(x, y)` reads it, and `Player.move` rolls a random number against it.

### `TERRAIN_COLORS`

```python
TERRAIN_COLORS = {
    TerrainType.VOID: (0.05, 0.05, 0.08),
    TerrainType.WATER: (0.16, 0.42, 0.80),
    TerrainType.SAND: (0.86, 0.78, 0.52),
    TerrainType.GRASS: (0.30, 0.62, 0.28),
    TerrainType.ROCK: (0.50, 0.50, 0.50),
}
```

The RGB colour of each terrain type, each channel from `0.0` to `1.0` (matplotlib style, not `0..255`).

| Terrain | (R, G, B) | Looks like |
| --- | --- | --- |
| `VOID` | `(0.05, 0.05, 0.08)` | Near-black with a hint of blue. |
| `WATER` | `(0.16, 0.42, 0.80)` | Lake blue. |
| `SAND` | `(0.86, 0.78, 0.52)` | Beach tan. |
| `GRASS` | `(0.30, 0.62, 0.28)` | Meadow green. |
| `ROCK` | `(0.50, 0.50, 0.50)` | Mid grey. |

Only the renderer uses this table. It also shades each colour by height, so hills look lighter than valleys.

### `classify_heights`

```python
def classify_heights(heights: np.ndarray, config: TerrainConfig) -> np.ndarray
```

Converts a 0-to-1 height map into a grid of `TerrainType` values.

**Parameters**

- `heights`: a numpy array of any shape with values in `0.0..1.0`. In practice it is `(height, width)`.
- `config`: a `TerrainConfig` with `water_threshold`, `sand_size` and `grass_size`.

**Returns** a numpy array of the same shape with `dtype=np.int8`. Each cell holds the integer value of a `TerrainType` (`1` to `4`). It never produces `VOID`; `Arena` adds that later for round arenas.

**Step by step**

1. `water_threshold = config.water_threshold` (default `0.25`).
2. `sand_threshold = water_threshold + config.sand_size` (default `0.35`).
3. `grass_threshold = sand_threshold + config.grass_size` (default `0.85`).
4. Make a grid the same shape as `heights`, filled with `ROCK`.
5. Where `heights < grass_threshold`, write `GRASS`.
6. Where `heights < sand_threshold`, write `SAND`. This overwrites some of the grass.
7. Where `heights < water_threshold`, write `WATER`. This overwrites some of the sand.
8. Return the grid.

With the defaults the bands are:

| Height range | Terrain |
| --- | --- |
| `0.00` up to (not including) `0.25` | `WATER` |
| `0.25` up to `0.35` | `SAND` |
| `0.35` up to `0.85` | `GRASS` |
| `0.85` to `1.00` | `ROCK` |

Because `PerlinNoise.grid()` equalizes heights, these ranges are also *fractions of cells*: about 25% water, 10% sand, 50% grass, 15% rock.

```python
import numpy as np
from hunger_games.config import TerrainConfig
from hunger_games.terrain import TerrainType, classify_heights

heights = np.array([[0.1, 0.3, 0.5, 0.95]])
terrain = classify_heights(heights, TerrainConfig(water_threshold=0.25, sand_size=0.1, grass_size=0.3))
print(terrain.tolist())
# [[1, 2, 3, 4]]  -> WATER, SAND, GRASS, ROCK
print([TerrainType(v) for v in terrain[0]])
```

Why paint from the top down instead of testing each band with two comparisons? Painting needs only one comparison per band and reads naturally: "everything below here is at most grass, everything below here is at most sand". It also means a band of size zero simply gets overwritten completely by the band below it, which is how the video's "set the size to zero to delete a terrain type" trick works for free.

## How to use it / experiment

**Count the cells of each type** on a real arena.

```python
import numpy as np
from hunger_games.arena import Arena
from hunger_games.config import SimulationConfig
from hunger_games.terrain import TerrainType

arena = Arena(SimulationConfig(seed=1), np.random.default_rng(1))
for kind in TerrainType:
    share = (arena.terrain == int(kind)).mean()
    print(f"{kind.name:6} {share:.2%}")
```

**Delete sand**, exactly as the video does.

```python
from hunger_games.config import TerrainConfig
terrain = classify_heights(heights, TerrainConfig(sand_size=0.0))
assert not (terrain == TerrainType.SAND).any()
```

**Make a mountain world.** Small `grass_size` leaves most of the map as rock.

```python
TerrainConfig(water_threshold=0.15, sand_size=0.05, grass_size=0.2)
```

**Change the gameplay tables.** The dictionaries are plain module-level objects, so you can edit them in a script before building an arena. This is a quick way to test "what if water were fast to cross?"

```python
import hunger_games.terrain as terrain
terrain.MOVE_SUCCESS[terrain.TerrainType.WATER] = 0.9
```

Note that `Arena` imports the dictionaries by name (`from hunger_games.terrain import MOVE_SUCCESS`), so mutate the dictionary in place as above. Rebinding `terrain.MOVE_SUCCESS = {...}` would not affect the copy `arena.py` already holds.

**Future projects.**

- *New terrain type* (say, `SWAMP` between sand and grass): add `SWAMP = 5` to `TerrainType`, give it an entry in all three tables, add `swamp_size` to `TerrainConfig` in [config.py](config.md), and add one more threshold and one more mask line in `classify_heights`. Also check `perception.py`, which lists the four real terrains when building its vector, and the renderer.
- *Neural network brain:* `Perception.to_vector()` one-hot encodes `TerrainType`. Adding a terrain type changes the vector length, so retrain any saved network.
- *Genetic algorithm:* nothing here changes, but note that the voting brain compares `terrain_here is TerrainType.GRASS`, so brains are already terrain-aware.
- *New layout:* layouts use `Arena.is_land`, which is built on `TerrainType.VOID` and `TerrainType.WATER`. New terrain types are automatically "land" unless you change `is_land`.

## Gotchas

- **Grids hold integers, not enum members.** `arena.terrain[y, x]` returns a numpy `int8`, not a `TerrainType`. Use `Arena.terrain_at(x, y)` or `TerrainType(int(value))` to get the member back. Comparing with `== TerrainType.GRASS` still works because `IntEnum` compares like an int, but `is TerrainType.GRASS` on a raw grid value is always `False`.
- **numpy indexing is `[row, column]`, so `[y, x]`.** `classify_heights` does not care (it works element-wise), but everything that reads its output does.
- **Comparisons are strict `<`.** A height exactly equal to a threshold belongs to the band *above* it. With equalized heights this only affects one or two cells.
- **Thresholds must stay in `0..1`.** If `water_threshold + sand_size + grass_size >= 1.0`, there is no rock. If `water_threshold >= 1.0`, everything is water.
- **`classify_heights` never produces `VOID`.** Only `Arena._carve_circle` writes `VOID`, and only for round arenas. An open-field arena has no `VOID` at all.
- **`VOID` has `MOVE_SUCCESS` of `0.0`**, but players never even roll for it, because `Player.move` refuses to step onto non-walkable cells first. The `0.0` is a safety net.
- **Sizes are relative to the previous band, not absolute cut-offs.** `grass_size=0.5` does not mean "grass ends at 0.5". It means "grass is 0.5 tall, starting wherever sand ends".
- **The output dtype is `int8`.** That is fine for values `0..4`, but if you ever add more than 127 terrain types (you will not) it would overflow.
