# `test_arena.py`

**Source:** [tests/test_arena.py](../../tests/test_arena.py)
**Tests:** [../arena.md](../arena.md) (`Arena`), [../terrain.md](../terrain.md) (`classify_heights`, `TerrainType`), [../resources.md](../resources.md) (`build_layout`), [../config.md](../config.md) (`SimulationConfig`, `TerrainConfig`, `ArenaShape`, `LayoutName`)

## Purpose

The arena is the world the tributes fight in. `Arena` takes a `SimulationConfig` and a random generator, builds a Perlin height map, turns it into terrain, optionally carves it into a circle, and pre-computes maps that tell every cell how far it is from water and grass. The two `ResourceLayout` classes then scatter supplies across it. This file guards all of those steps.

The first two tests check `classify_heights` on tiny hand-made grids, where you can predict the answer by eye. They prove the chapter 4 rule that each terrain threshold is relative to the one before it, and that setting a band's size to zero removes that terrain type entirely. The next two check the two arena shapes: a round arena must have void corners, and an open field must have none. Then come the podiums. Since the 75th games had podiums standing in the sea, water podiums are allowed by default, so one test checks that every podium is inside the arena and distinct, and a second checks that turning `allow_water_podiums` off really does move every podium onto dry land. Finally the water distance field (zero at water, and one step along the arrow brings you one step closer) and the supply layouts (both put something on the map).

The bugs these catch are the kind that silently ruin a simulation rather than crash it. Wrong thresholds give the wrong mix of terrain. A missing `_carve_circle` call gives a square arena labelled round. Podiums in the void put tributes outside the world. A distance field pointing the wrong way sends thirsty players uphill.

## Concepts you need

**Test discovery.** pytest runs every `test_*` function in every `test_*.py` file. Functions without the `test_` prefix, such as `make_arena`, are ignored by discovery and only run when a test calls them.

**Helper functions.** `make_arena(**overrides)` is a small factory used by six tests. `**overrides` collects any keyword arguments and passes them on to `SimulationConfig`, so `make_arena(shape=ArenaShape.ROUND)` changes one setting and keeps the rest.

**Plain `assert`.** Each `assert` is one check. If it fails, pytest stops that test and reports it. Other tests keep running.

**Looping inside a test.** Some tests loop over every member of an enum, such as `for shape in ArenaShape:`. Every iteration must pass. pytest treats the whole loop as one test, so a failure message shows which iteration broke by printing the values involved.

**Boolean array tricks.** `(terrain == TerrainType.SAND).any()` is True if any cell is sand. `np.nonzero(mask)` returns the row and column indices of every True cell, in `(ys, xs)` order.

**Running a subset.** `python -m pytest tests/test_arena.py -k podium` runs only the two podium tests.

## Walkthrough

### `make_arena(**overrides) -> Arena`

```python
def make_arena(**overrides) -> Arena:
    config = SimulationConfig(width=60, height=60, seed=3, **overrides)
    return Arena(config, np.random.default_rng(3))
```

Builds a 60 by 60 arena with a fixed seed. `Arena` draws its noise seed from the generator you pass in, so the same `default_rng(3)` gives the same terrain every run. 60 by 60 is half the default size and generates in a few milliseconds. Anything not overridden keeps the `SimulationConfig` default, including `shape=OPEN_FIELD`, `allow_water_podiums=True` and the default `TerrainConfig`. The third `Arena` parameter, `terrain`, is left as `None`, so the map is generated rather than painted.

### `test_relative_thresholds()`

**Setup.** A one-row height map `[[0.1, 0.3, 0.5, 0.95]]` and a `TerrainConfig(water_threshold=0.25, sand_size=0.1, grass_size=0.3)`. From these, `classify_heights` computes three cut-offs: water below 0.25, sand from 0.25 to 0.35, grass from 0.35 to 0.65, rock above.

**`assert terrain.tolist() == [[WATER, SAND, GRASS, ROCK]]`.** Height 0.1 is below 0.25 so it is water. Height 0.3 is between 0.25 and 0.35 so it is sand. Height 0.5 is between 0.35 and 0.65 so it is grass. Height 0.95 is above everything so it is rock. `.tolist()` converts the numpy array to a plain list so the comparison reads naturally. Because `TerrainType` is an `IntEnum`, the integers in the grid compare equal to the enum members. A failure here would mean the thresholds were being treated as absolute values instead of cumulative sizes, which is the exact mistake chapter 4 warns about.

**Why these numbers.** Each height sits inside one band with room on both sides, so a small change to the maths is still caught cleanly.

### `test_zero_size_removes_a_terrain_type()`

**Setup.** A 20 by 20 grid of uniform random heights from `default_rng(0)`, and a config with `sand_size=0.0`. That makes the sand threshold equal to the water threshold, so no height can be at or above 0.25 and below 0.25 at the same time.

**`assert not (terrain == TerrainType.SAND).any()`.** With 400 random heights, some would certainly land in a sand band if one existed. None do. A failure would mean `classify_heights` was giving sand some fixed minimum width, or using `<=` where it should use `<`.

**Why these numbers.** The seed is fixed so the test is repeatable, but any spread of heights would do.

### `test_round_arena_has_void_corners()`

**Setup.** `make_arena(shape=ArenaShape.ROUND)`. In `Arena.__init__`, this shape triggers `_carve_circle`, which marks every cell farther than `radius` from the centre as `VOID`. For a 60 by 60 grid the radius is `min(60, 60) / 2 - 1 = 29.0` and the centre is `(30, 30)`.

**`assert arena.terrain_at(0, 0) is TerrainType.VOID`.** The corner is about 42 cells from the centre, well outside the circle. `terrain_at` converts the stored integer back to a `TerrainType`, so `is` works on the enum member.

**`assert arena.terrain_at(arena.center_x, arena.center_y) is not TerrainType.VOID`.** The centre is inside the circle, so it must keep its real terrain. This guards against a carve that voids the inside instead of the outside.

**`assert not arena.is_walkable(0, 0)`.** `is_walkable` is how `Player.move` decides whether a step is legal. Void must be unwalkable, or tributes could leave the circle.

### `test_open_field_has_no_void()`

**Setup.** `make_arena(shape=ArenaShape.OPEN_FIELD)`.

**`assert not (arena.terrain == TerrainType.VOID).any()`.** The open field uses the whole square, so no cell should be void. A failure would mean the carve was being applied regardless of shape.

### `test_edge_positions_are_inside_the_arena_and_distinct()`

**Setup.** Loops over both shapes, builds an arena for each, and asks `arena.edge_positions(24)` for 24 podiums. Round arenas place them on a circle three cells inside the boundary (radius 26 here). Open fields walk around a rectangle three cells inside the edge. Both pass every candidate through `snap_to_podium`, which, because `allow_water_podiums` is `True` by default, calls `snap_to_walkable`: a podium already inside the arena stays put, water included, and only a podium in the void is nudged inward.

**`assert len(podiums) == 24`.** One podium per tribute.

**`assert len(set(podiums)) == 24`.** A set drops duplicates, so if two tributes were given the same cell the set would be smaller. Two podiums could collide if the rounding on the circle or the snap nudged neighbours onto the same cell. A failure would mean two players start stacked on top of each other.

**`assert all(arena.is_walkable(x, y) for x, y in podiums)`.** `is_walkable` means any real terrain, water included. This is deliberately weaker than `is_land`: the 75th games' arena had podiums in the sea, and the default config allows that. A failure would mean a podium was left in the void, for example if the ring radius reached past the carved circle.

**Why 24.** That is the default `num_players` and matches the films.

### `test_edge_positions_avoid_water_when_asked()`

**Setup.** `make_arena(allow_water_podiums=False)`. Same seed, same 60 by 60 open field, but now `snap_to_podium` calls `snap_to_land` instead of `snap_to_walkable`.

**`assert all(arena.is_land(x, y) for x, y in arena.edge_positions(24))`.** `is_land` means walkable and not water. With seed 3 and the default thresholds, a quarter of cells are water, and some of the 24 rectangle positions land in it, so this test really does exercise the nudge. A failure would mean the config flag was ignored, or `snap_to_land` returned water.

**Why this test exists.** The default behaviour changed from "always dry land" to "water allowed". This test keeps the old behaviour available and working behind the flag.

### `test_water_distance_field()`

**Setup.** A default open-field arena. `np.nonzero(arena.terrain == TerrainType.WATER)` gives every water cell. The test takes the first one.

**`assert len(xs) > 0`.** The default thresholds must produce some water. With `equalize=True` in the noise, exactly a quarter of cells are water, which is 900 here.

**`assert arena.distance_to_water(int(xs[0]), int(ys[0])) == 0.0`.** The breadth-first search in `_distance_field` seeds every water cell at distance 0. If a water cell reported anything else, the search had the wrong sources.

**`dx, dy = arena.direction_to_water(x, y)` on the first grass cell, then `assert arena.distance_to_water(x + dx, y + dy) == arena.distance_to_water(x, y) - 1`.** The direction map stores, for each cell, the step back toward the cell that reached it first in the search. Following that step must bring you exactly one step closer to water. This is the property `VotingBrain` relies on when it votes to move along `water_direction`. A failure would mean the stored direction was reversed (the classic `(dx, dy)` versus `(-dx, -dy)` bug) or the distances were off by one.

**Why the first grass cell.** Grass is never water, so its distance is at least 1 and there is always a step to take. Using index 0 keeps the test deterministic.

### `test_layouts_place_supplies()`

**Setup.** Loops over `LayoutName.CORNUCOPIA` and `LayoutName.RING`. For each, builds a fresh arena and calls `build_layout(name).apply(arena, np.random.default_rng(1))`.

**`assert (arena.resources.kind != 0).sum() > 0`.** `kind` is the grid of `ResourceKind` integers, where 0 means empty. Counting non-zero cells and requiring at least one proves the layout put something down. A failure would mean the layout skipped every cell, perhaps because `is_land` returned False everywhere. The docstring mentions the ring keeping weapons central, but that is not asserted here. See the extension ideas below.

## How to run and extend

```bash
python -m pytest tests/test_arena.py
python -m pytest tests/test_arena.py::test_edge_positions_avoid_water_when_asked
python -m pytest tests/test_arena.py -k "void or podium"
python -m pytest tests/test_arena.py -v
```

Ideas for new tests in this area:

**1. Water podiums really happen by default.** Prove the default is not accidentally dry.

```python
def test_default_podiums_can_stand_in_water():
    arena = make_arena()
    podiums = arena.edge_positions(24)
    assert any(arena.is_water(x, y) for x, y in podiums)
```

**2. `snap_to_podium` follows the flag.** Feed it a void cell and a water cell under both settings.

```python
def test_snap_to_podium_respects_the_flag():
    wet = make_arena()
    ys, xs = np.nonzero(wet.terrain == TerrainType.WATER)
    x, y = int(xs[0]), int(ys[0])
    assert wet.snap_to_podium(x, y) == (x, y)
    dry = make_arena(allow_water_podiums=False)
    assert dry.is_land(*dry.snap_to_podium(x, y))
```

**3. A painted map replaces the generated one.** `Arena` accepts a `terrain` grid.

```python
def test_painted_terrain_is_adopted():
    painted = np.full((60, 60), int(TerrainType.SAND), dtype=np.int8)
    config = SimulationConfig(width=60, height=60, shape=ArenaShape.ROUND)
    arena = Arena(config, np.random.default_rng(0), terrain=painted)
    assert (arena.terrain == int(TerrainType.SAND)).all()
    assert 0.0 <= arena.heights.min() and arena.heights.max() <= 1.0
```

**4. The ring layout keeps the best weapons near the centre.** Weapons inside the central cache have quality 0.8 or more.

```python
from hunger_games.resources import ResourceKind

def test_ring_layout_puts_best_weapons_in_centre():
    arena = make_arena()
    build_layout(LayoutName.RING).apply(arena, np.random.default_rng(1))
    xs, ys = arena.resources.cells_of_kind(ResourceKind.WEAPON)
    qualities = arena.resources.quality[ys, xs]
    distances = np.array([arena.normalized_distance_from_center(x, y) for x, y in zip(xs, ys)])
    assert qualities[distances < 0.06].min() >= 0.8
```

## Gotchas

**Two seeds, not one.** `make_arena` fixes `seed=3` in the config, but `Arena` never reads `config.seed`. It draws its noise seed from the generator you pass, so `np.random.default_rng(3)` is the one that matters. `Game` is what turns `config.seed` into a generator.

**`is_walkable` is not `is_land`.** The default podium test uses `is_walkable` on purpose. If you "tighten" it to `is_land` it will fail, because water podiums are allowed by default. Use `allow_water_podiums=False` when you want dry podiums.

**Default config values.** `test_water_distance_field` and `test_layouts_place_supplies` rely on the default `TerrainConfig` giving some water and some grass, and on `noise.grid` using `equalize=True`. Changing `water_threshold` to 0.0 would make the first assertion in the water test fail.

**Indexing order.** numpy grids are `[y, x]`. `np.nonzero` returns `(ys, xs)`. Every method on `Arena` takes `(x, y)`. The test converts with `int(xs[0]), int(ys[0])` for this reason.

**Enum comparisons.** `terrain_at` returns a `TerrainType`, so `is` works. The raw `arena.terrain` grid holds `int8` values, so compare those with `==`.

**Speed.** The slowest test here is `test_layouts_place_supplies` at about 0.05 seconds, because both layouts visit all 3600 cells in Python loops. Doubling the arena size quadruples that time.
