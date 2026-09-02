# `painter.py`

**Source:** [hunger_games/ui/painter.py](../../hunger_games/ui/painter.py)
**Depends on:** `numpy`; project modules [../arena.md](../arena.md) (`Arena`), [../config.md](../config.md) (`SimulationConfig`), [../noise.md](../noise.md) (`PerlinNoise`), [../terrain.md](../terrain.md) (`TerrainType`)
**Used by:** [session.md](session.md) (`Session.painter`), [app.md](app.md) (the Map tab reads `MapPainter.PRESETS` and calls stamp and fill methods directly)

## Purpose

`painter.py` is the part of the dashboard that lets a game maker draw the arena by hand. It holds a terrain grid (water, sand, grass, rock, and void for "outside the arena") and a matching height grid, and it offers the editing operations the Map tab needs: a round brush, circle and rectangle and ring stamps, a whole-grid fill, "carve round", and five named presets including the 75th games' clock island.

The file has no Dear PyGui code at all. That is on purpose. A `MapPainter` can be created, painted and inspected from a plain Python script or a unit test with no window open. The dashboard's canvas ([canvas.md](canvas.md)) only reads `terrain`, `heights` and `version`.

## Concepts you need

- **numpy grids.** The terrain is a 2D array shaped `(height, width)`. Row comes first, so cell `(x, y)` is `terrain[y, x]`. The values are the integers of `TerrainType` (0 void, 1 water, 2 sand, 3 grass, 4 rock), stored as `int8` to keep them small.
- **Boolean masks.** `np.indices((h, w))` gives two grids holding every cell's row and column. `np.hypot(xs - cx, ys - cy) <= radius` turns that into a grid of `True`/`False` for "inside the circle". Assigning through a mask, `terrain[inside] = 1`, paints only those cells. That is how every stamp works, with no loops over pixels.
- **Slicing.** `terrain[top:bottom + 1, left:right + 1] = value` fills a rectangle. The `+ 1` is because Python slices stop before the end index.
- **Version counters.** Instead of telling the canvas "I changed", the painter bumps `self.version`. The canvas remembers the version it last drew and re-uploads the texture only when the number moved. This is a simple and reliable way to avoid redrawing a big image every frame.
- **`IntEnum`.** `TerrainType` is an `IntEnum`, so `int(TerrainType.GRASS)` is `3` and `TerrainType[name.upper()]` looks one up by name. The painter always stores the integer.
- **Re-running `__init__`.** `resize` calls `self.__init__(width, height)` on an existing object. That is allowed in Python: `__init__` is an ordinary method that sets attributes, so calling it again simply resets them.

## Walkthrough

### `class MapPainter`

"A terrain grid with brushes, stamps and presets."

#### `MapPainter.PRESETS`

```python
PRESETS = ("perlin", "flat_field", "flat_round", "quarter_quell", "lake_island")
```

The names accepted by `apply_preset` and listed in the Map tab's preset combo. The order here is the order in the combo.

#### `MapPainter.__init__`

```python
def __init__(self, width: int, height: int) -> None
```

Starts with a flat grass field. Sets `width` and `height`, fills `terrain` with `GRASS`, fills `heights` with `0.5`, sets `version = 0`, and precomputes `_noise`, a Perlin grid from `PerlinNoise(0).grid(width, height, scale=20.0, octaves=3)`. The noise is seeded with `0` so a painted map always gets the same subtle texture.

```python
from hunger_games.ui.painter import MapPainter
painter = MapPainter(120, 120)
painter.terrain.shape   # (120, 120)
```

#### `MapPainter._changed`

```python
def _changed(self) -> None
```

Adds one to `version`. Every method that touches the grid calls this last. The leading underscore marks it as internal.

#### `MapPainter.load`

```python
def load(self, terrain: np.ndarray, heights: np.ndarray | None = None) -> None
```

Replaces the whole map. Copies `terrain` as `int8`, reads the new `height, width` from its shape, regenerates `_noise` for that size, and takes `heights` if given or derives them with `derive_heights()`. Used when a scenario or a replay is loaded and when the Perlin preset adopts a fresh `Arena`.

#### `MapPainter.derive_heights`

```python
def derive_heights(self) -> np.ndarray
```

Returns heights for a painted map by calling `Arena._heights_from_terrain(self.terrain, self._noise)`: a base height per terrain type plus a little noise. Reusing the arena's rule means the relief shading in the dashboard matches what a game built from this map will use.

#### `MapPainter.resize`

```python
def resize(self, width: int, height: int) -> None
```

Starts over at a new size by calling `__init__` again. Everything painted is lost; the result is a flat grass field.

#### `MapPainter.in_bounds`

```python
def in_bounds(self, x: int, y: int) -> bool
```

`True` when `0 <= x < width` and `0 <= y < height`. The session checks this before every brush stroke and podium move.

#### `MapPainter.paint`

```python
def paint(self, x: int, y: int, terrain: TerrainType, radius: int = 1) -> None
```

The brush. It is exactly `stamp_circle(x, y, radius, terrain)`. With `radius = 0` only the centre cell is painted, because `hypot(0, 0) <= 0` is true for that one cell alone.

#### `MapPainter.stamp_circle`

```python
def stamp_circle(self, cx: int, cy: int, radius: float, terrain: TerrainType) -> None
```

Fills every cell whose distance from `(cx, cy)` is at most `radius`. The centre may be off the grid; only the cells that exist are touched, because the mask is built over the grid itself.

#### `MapPainter.stamp_rectangle`

```python
def stamp_rectangle(self, x0: int, y0: int, x1: int, y1: int, terrain: TerrainType) -> None
```

Fills the rectangle between two corners, inclusive, in any order. Corners are sorted with `min`/`max` and clipped to the grid before slicing.

#### `MapPainter.stamp_ring`

```python
def stamp_ring(self, cx: int, cy: int, inner: float, outer: float, terrain: TerrainType) -> None
```

Fills the cells whose distance is between `inner` and `outer` inclusive. With `inner = 0.0` it is a filled circle; the quarter_quell preset uses that for its sea.

#### `MapPainter.fill`

```python
def fill(self, terrain: TerrainType) -> None
```

Sets every cell to one type with `self.terrain[:] = int(terrain)`. The slice assignment writes into the existing array instead of replacing it.

#### `MapPainter.carve_round`

```python
def carve_round(self) -> None
```

Turns the square grid into a round arena. The centre is `(width // 2, height // 2)`, the radius is `min(width, height) / 2.0 - 1.0`, and every cell outside that circle becomes `VOID`. Games treat void as "not part of the arena".

#### `MapPainter.finish`

```python
def finish(self) -> None
```

Recomputes `heights` from the terrain and bumps the version. The brush does not do this itself because recomputing heights over the whole grid on every mouse move would be wasteful; the dashboard calls `finish` once when the mouse button is released.

#### `MapPainter.apply_preset`

```python
def apply_preset(self, name: str, config: SimulationConfig, seed: int | None = None) -> None
```

Loads one preset by name.

| Name | What it builds |
| --- | --- |
| `perlin` | A normal generated arena: `Arena(config, rng)` with `seed`, else `config.seed`, else random. Adopts its terrain and heights and returns early. |
| `flat_field` | A plain square grass meadow. |
| `flat_round` | Grass, then `carve_round()`. |
| `quarter_quell` | The 75th games: round arena, sea from the centre out to 45 % of the radius, a sand island at 12 %, a sand beach ring from 45 % to 52 %, and twelve spokes of single rock cells from 14 % to 43 % so tributes can wade out. |
| `lake_island` | Square field with a lake at 45 % of the radius, a sand shore to 50 %, a grass island at 12 %, and a rock quarter-circle (30 %) in each corner. |

Every preset except `perlin` starts with `fill(GRASS)` and ends with `finish()`. An unknown name raises `KeyError` listing the valid choices.

```python
from hunger_games.config import SimulationConfig
painter = MapPainter(120, 120)
painter.apply_preset("quarter_quell", SimulationConfig())
painter.coverage()   # {'water': 0.18, 'sand': 0.08, 'grass': 0.73, 'rock': 0.01} (rounded)
```

#### `MapPainter.coverage`

```python
def coverage(self) -> dict[str, float]
```

The fraction of the arena covered by each terrain type, ignoring void cells in both the numerator and the denominator. Keys are lower-case names: `water`, `sand`, `grass`, `rock`. The Map tab shows this as a one-line summary.

## How to use it / experiment

**Paint a map without the GUI.**

```python
from hunger_games.config import SimulationConfig
from hunger_games.terrain import TerrainType
from hunger_games.ui.painter import MapPainter

p = MapPainter(100, 100)
p.fill(TerrainType.WATER)
p.stamp_circle(50, 50, 30, TerrainType.GRASS)
p.stamp_ring(50, 50, 30, 34, TerrainType.SAND)
p.finish()
from hunger_games.scenario import Scenario
Scenario(terrain=p.terrain.tolist(), title="my island").save("island.json")
```

**Add a new preset.** Add its name to `PRESETS`, then add an `elif name == "...":` branch in `apply_preset` between the existing ones and the final `else`. Use `stamp_circle`, `stamp_ring`, `stamp_rectangle` and direct assignment to `self.terrain[y, x]`. Do not call `finish()` inside your branch; the method already does that at the end. The Map tab's combo picks the name up from `PRESETS` automatically.

**Add a new stamp shape.** Build a boolean mask over `np.indices((self.height, self.width))`, assign the terrain through it, and call `self._changed()`. A diamond, for example, is `abs(xs - cx) + abs(ys - cy) <= radius`.

## Gotchas

- `terrain[y, x]`, not `terrain[x, y]`. Row first. `stamp_rectangle` takes `x0, y0, x1, y1` in x-then-y order and converts internally.
- `resize` destroys the painting. `load` keeps it and changes the size to match the given array.
- Brush strokes leave `heights` stale until `finish()` runs. If you script the painter, call `finish()` before saving or before reading `heights`.
- `apply_preset("perlin", ...)` ignores the painter's current size and takes the size from `config.width` and `config.height`. `Session.generate_arena` resizes the painter first so they agree.
- `coverage()` divides by the number of non-void cells, so a fully void map reports every type as `0.0` rather than raising.
- Painting `void` with the brush is allowed (the Map tab lists it). Podiums and hand-placed loot cannot be put on void cells, and `carve_round` is the tidy way to make a round arena.
