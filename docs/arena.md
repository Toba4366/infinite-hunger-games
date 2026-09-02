# `arena.py`

**Source:** [hunger_games/arena.py](../hunger_games/arena.py)
**Depends on:** [config.py](config.md) (`ArenaShape`, `SimulationConfig`), [noise.py](noise.md) (`PerlinNoise`), [resources.py](resources.md) (`ResourceGrid`), [terrain.py](terrain.md) (`HUNT_DIFFICULTY`, `MOVE_SUCCESS`, `TerrainType`, `classify_heights`), `numpy`, and the standard library (`collections.deque`, `math`).
**Used by:** [game.py](game.md) (builds one per game), [gamemaker.py](gamemaker.md), [player.py](player.md) (`Arena`, `sign`), the dashboard's `ui/painter.py` (`Arena._heights_from_terrain`), and [tests/test_arena.md](tests/test_arena.md).

## Purpose

An `Arena` is the world. It owns the height map, the terrain grid, the supply grid, and two pre-computed navigation maps ("how far to the nearest water, and which way") that tributes use to find a drink or a hunting ground. It also knows its own geometry, so the 74th-games open field and the 75th-games round arena are both just a setting.

This is the chapter 4 pipeline: Perlin noise makes a height map, the height map is classified into water, sand, grass and rock, and a round arena is carved out of the square grid by marking the corners `VOID`. The arena then builds breadth-first-search distance fields so a player anywhere can take one step toward water without doing any pathfinding of their own.

New in this version is the **painted terrain override**. The dashboard lets a game maker paint a map by hand. `Arena(config, rng, terrain=grid)` skips generation and adopts the painted grid, then gives it plausible heights from `PAINTED_HEIGHTS` plus a little noise so "downhill" still means something. `Game` passes `scenario.terrain` here (see [scenario.md](scenario.md)).

Also new: `snap_to_walkable` and `snap_to_podium`, which support the `allow_water_podiums` setting, and `move_success_at`, which lets `Player.move` fail a step into water or rock.

## Concepts you need

**numpy grids indexed `[y, x]`.** Row first. Every public method takes `(x, y)` and flips it internally.

**Boolean masks.** `self.terrain[outside] = 0` writes to every cell where the boolean array `outside` is `True`. `heights[terrain == 3] = 0.5` is the same trick.

**`np.indices` and `np.hypot`.** `np.indices((h, w))` gives two grids holding each cell's row and column. `np.hypot(a, b)` is `sqrt(a*a + b*b)` for whole arrays at once.

**Chebyshev rings.** `max(abs(dx), abs(dy)) == ring` picks the square ring of cells exactly `ring` steps out. The snap methods search these rings from small to large.

**Breadth-first search (BFS).** Start from every target cell at distance 0, expand to unvisited neighbours one layer at a time using a queue. The first time BFS reaches a cell it has found the shortest path. `deque.popleft()` is the O(1) "take from the front" that makes it a queue.

**Class attributes and `classmethod`.** `PAINTED_HEIGHTS` lives on the class, not on each instance. `_heights_from_terrain` is a `@classmethod` so the dashboard's painter can call it without building an `Arena`.

**Leading underscore.** `_carve_circle`, `_heights_from_terrain` and `_distance_field` are internal helpers. Nothing stops you calling them, but they are not the public interface.

## Walkthrough

### `NEIGHBOUR_STEPS`

```python
NEIGHBOUR_STEPS = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
```

The eight surrounding cells as `(dx, dy)` steps. Used by `downhill_direction`.

### `CROSS_STEPS`

```python
CROSS_STEPS = [(0, -1), (-1, 0), (1, 0), (0, 1)]
```

The four orthogonal neighbours. The distance-field BFS uses these so "distance" counts whole steps.

### `sign`

```python
def sign(value: float) -> int
```

Returns `-1`, `0` or `1`. The trick `(value > 0) - (value < 0)` subtracts two booleans. `Player` imports it too.

### `Arena.PAINTED_HEIGHTS`

```python
PAINTED_HEIGHTS = {
    TerrainType.VOID: 0.0,
    TerrainType.WATER: 0.1,
    TerrainType.SAND: 0.3,
    TerrainType.GRASS: 0.5,
    TerrainType.ROCK: 0.8,
}
```

| Terrain | Base height |
| --- | --- |
| `VOID` | `0.0` |
| `WATER` | `0.1` |
| `SAND` | `0.3` |
| `GRASS` | `0.5` |
| `ROCK` | `0.8` |

A gentle height for each painted terrain type. Water is lowest and rock highest, in the same order the generated thresholds produce, so `downhill_direction` still leads toward water on a hand-made map.

### `Arena.__init__`

```python
def __init__(self, config: SimulationConfig, rng: np.random.Generator, terrain: np.ndarray | None = None) -> None
```

Builds the world. Step by step:

1. Stores `config`, `width`, `height`, `center_x = width // 2`, `center_y = height // 2` and `radius = min(width, height) / 2.0 - 1.0`.
2. Draws a noise seed from `rng` and builds a `PerlinNoise`.
3. Computes `persistence = config.noise.persistence + 0.25 * config.chaos`, so more chaos means rougher land.
4. Generates the `0..1` height map with `noise.grid(width, height, scale, octaves, persistence, lacunarity)`.
5. **If `terrain` is given:** copies it into an `int8` grid and replaces the heights with `_heights_from_terrain(terrain, heights)`. No classification and no circle carving happen; the painted map is taken as is.
6. **Otherwise:** classifies the heights with `classify_heights(heights, config.terrain)` and, if `config.shape is ArenaShape.ROUND`, calls `_carve_circle`.
7. Creates an empty `ResourceGrid` (a layout fills it later).
8. Pre-computes `water_distance, water_direction` and `grass_distance, grass_direction` with `_distance_field`.

The noise is generated even for a painted map. That keeps the random stream identical either way and gives the painted map its texture.

### `Arena._carve_circle`

```python
def _carve_circle(self) -> None
```

Builds row and column index grids, measures every cell's distance from the centre with `np.hypot`, and marks cells beyond `radius` as `VOID` with height `0.0`. Only called for generated round arenas.

### `Arena._heights_from_terrain`

```python
@classmethod
def _heights_from_terrain(cls, terrain: np.ndarray, noise: np.ndarray) -> np.ndarray
```

Gives a painted map plausible heights. Starts flat, writes each type's `PAINTED_HEIGHTS` base into the matching cells, adds `0.1 * (noise - 0.5)` so slopes exist within a terrain type, forces `VOID` back to `0.0`, and clips to `0..1`. `ui/painter.py` calls this directly so the dashboard preview shades hills the same way a game will.

### `Arena.in_bounds`

```python
def in_bounds(self, x: int, y: int) -> bool
```

`0 <= x < width and 0 <= y < height`.

### `Arena.terrain_at`

```python
def terrain_at(self, x: int, y: int) -> TerrainType
```

The cell's `TerrainType`, or `VOID` if off the grid. Because off-grid is `VOID`, callers rarely need to check bounds themselves.

### `Arena.is_walkable`

```python
def is_walkable(self, x: int, y: int) -> bool
```

Anything but `VOID`, so water counts as walkable.

### `Arena.is_water`

```python
def is_water(self, x: int, y: int) -> bool
```

True when the cell is `WATER`.

### `Arena.is_land`

```python
def is_land(self, x: int, y: int) -> bool
```

Walkable and not water. Layouts only scatter loot on land.

### `Arena.height_at`

```python
def height_at(self, x: int, y: int) -> float
```

The `0..1` height. No bounds check, so keep `x, y` inside the grid.

### `Arena.hunt_difficulty_at`

```python
def hunt_difficulty_at(self, x: int, y: int) -> float
```

Looks the terrain up in `HUNT_DIFFICULTY` (grass `0.2`, water `0.6`, sand `0.8`, rock `0.9`, void `1.0`).

### `Arena.move_success_at`

```python
def move_success_at(self, x: int, y: int) -> float
```

Looks the terrain up in `MOVE_SUCCESS`: the chance a step *into* this cell succeeds.

| Terrain | Chance |
| --- | --- |
| `GRASS` | `1.0` |
| `SAND` | `0.85` |
| `ROCK` | `0.6` |
| `WATER` | `0.5` |
| `VOID` | `0.0` |

`Player.move` rolls `rng.random() > arena.move_success_at(nx, ny)` and stays put on a failure. Wading and climbing are slow, which is how chases get resolved.

### `Arena.distance_from_center`

```python
def distance_from_center(self, x: int, y: int) -> float
```

Straight-line distance in cells, via `math.hypot`.

### `Arena.normalized_distance_from_center`

```python
def normalized_distance_from_center(self, x: int, y: int) -> float
```

Distance divided by `radius`, capped at `1.0` so the corners of a square arena never exceed the edge. `RingLayout` builds its whole gradient from this.

### `Arena.direction_to_center`

```python
def direction_to_center(self, x: int, y: int) -> tuple[int, int]
```

`(sign(center_x - x), sign(center_y - y))`: one diagonal-allowed step toward the middle.

### `Arena.downhill_direction`

```python
def downhill_direction(self, x: int, y: int) -> tuple[int, int]
```

Checks all eight walkable neighbours and returns the step to the lowest one, or `(0, 0)` if none is lower than where you stand. Water lies in the valleys, so this is a cheap way to look for it without a map.

### `Arena.direction_to_water` and `Arena.distance_to_water`

```python
def direction_to_water(self, x: int, y: int) -> tuple[int, int]
def distance_to_water(self, x: int, y: int) -> float
```

Read the pre-computed maps. Distance is in steps; it is `inf` if no water is reachable. Direction is `(0, 0)` on a water cell or an unreachable one.

### `Arena.direction_to_grass` and `Arena.distance_to_grass`

```python
def direction_to_grass(self, x: int, y: int) -> tuple[int, int]
def distance_to_grass(self, x: int, y: int) -> float
```

The same for grass, the best hunting ground.

### `Arena.snap_to_land`

```python
def snap_to_land(self, x: int, y: int) -> tuple[int, int]
```

Returns `(x, y)` if it is already dry land. Otherwise searches Chebyshev rings of growing size and returns the first land cell found. Falls back to the centre if the arena has no land at all.

### `Arena.snap_to_walkable`

```python
def snap_to_walkable(self, x: int, y: int) -> tuple[int, int]
```

Identical search, but accepts any non-`VOID` cell including water. Used when a podium is allowed to stand in the sea.

### `Arena.snap_to_podium`

```python
def snap_to_podium(self, x: int, y: int) -> tuple[int, int]
```

The single place that reads `config.allow_water_podiums`: `True` calls `snap_to_walkable`, `False` calls `snap_to_land`. Both layouts and `Game._place_players` (for roster podiums from a scenario) go through this, so the setting is honoured everywhere.

### `Arena.edge_positions`

```python
def edge_positions(self, count: int) -> list[tuple[int, int]]
```

Evenly spaced podiums along the outer edge. For `ROUND` arenas it walks a circle of radius `radius - 3.0`. For open fields it walks the perimeter of a rectangle inset `3` cells from the border, converting a distance `t` along the perimeter into a point on the top, right, bottom or left edge in turn. Every point is passed through `snap_to_podium`.

### `Arena._distance_field`

```python
def _distance_field(self, target: TerrainType) -> tuple[np.ndarray, np.ndarray]
```

Breadth-first search from every cell of `target` type at once. Returns two grids: `distance` (`float`, `inf` where unreachable) and `direction` (shape `(height, width, 2)`, `int8`). When BFS steps from `(x, y)` to a neighbour with `(dx, dy)`, the neighbour's direction is recorded as `(-dx, -dy)`, the reverse step, because that is the way *back* toward the source. Only walkable cells are expanded, so paths never cross `VOID`.

## How to use it / experiment

Generate an arena and inspect it:

```python
import numpy as np
from hunger_games.arena import Arena
from hunger_games.config import ArenaShape, SimulationConfig
from hunger_games.terrain import TerrainType

config = SimulationConfig(width=60, height=60, shape=ArenaShape.ROUND, seed=1)
arena = Arena(config, np.random.default_rng(1))
print(arena.terrain_at(0, 0).name)                     # VOID
print(arena.distance_to_water(arena.center_x, arena.center_y))
print(arena.direction_to_water(arena.center_x, arena.center_y))
```

Build an arena from a painted map:

```python
painted = np.full((40, 40), int(TerrainType.GRASS), dtype=np.int8)
painted[15:25, 15:25] = int(TerrainType.WATER)
arena = Arena(SimulationConfig(width=40, height=40), np.random.default_rng(0), terrain=painted)
print(arena.is_water(20, 20), arena.height_at(20, 20) < arena.height_at(5, 5))  # True True
```

See the podium setting in action:

```python
dry = Arena(SimulationConfig(width=40, height=40, allow_water_podiums=False, seed=2), np.random.default_rng(2))
assert all(dry.is_land(x, y) for x, y in dry.edge_positions(24))
```

Test how terrain slows movement on the painted map above:

```python
print(arena.move_success_at(20, 20))   # 0.5, a water cell
print(arena.move_success_at(5, 5))     # 1.0, a grass cell
```

## Gotchas

- **A painted map is never carved.** `config.shape` is ignored when `terrain` is given. If you want a round painted arena, paint the `VOID` corners yourself.
- **A painted map must match `config.width` and `config.height`.** The arena reads its size from the config, not from the array. A mismatch breaks the distance fields and the renderer.
- **`is_walkable` includes water.** Use `is_land` when you mean dry ground.
- **`height_at` does no bounds check.** Off-grid coordinates raise `IndexError`; `terrain_at` is the safe one.
- **`distance_to_water` can be `inf`.** An arena with no water, or a painted island of `VOID`, gives infinite distance and a `(0, 0)` direction. The voting brain treats `inf` as "no water known".
- **Chaos changes the map.** `persistence` gets `0.25 * chaos` added, so `chaos` is not just about dice.
- **The distance fields are built once.** If you edit `arena.terrain` after construction, `water_distance` is stale. Build a new arena instead.
- **`radius` is measured from the shorter side.** In a non-square grid the round arena is a circle, not an ellipse, and `normalized_distance_from_center` caps at `1.0` past it.
