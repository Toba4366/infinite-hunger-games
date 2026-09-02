# `test_noise.py`

**Source:** [tests/test_noise.py](../../tests/test_noise.py)
**Tests:** [../noise.md](../noise.md) (`hunger_games/noise.py`, the `PerlinNoise` class)

## Purpose

Every arena in the simulator starts as a height map. `PerlinNoise.grid()` produces that map: a 2D numpy array of numbers between 0.0 and 1.0, one per cell. `terrain.py` then slices those heights into water, sand, grass and rock using the chapter 4 thresholds (water below 0.25, then a band of sand, then a band of grass, rock on top). If the noise is wrong, every terrain map is wrong, and every game played on it is wrong too.

This file checks the three promises the rest of the package relies on. First, the grid has the right shape and really does run from 0.0 to 1.0, so the thresholds in `TerrainConfig` mean what they say. Second, the same seed always gives the same map, which is what makes a whole game replayable from one seed. Third, the noise is smooth. Neighbouring cells should differ by small amounts, so the map looks like rolling hills instead of static.

The bugs each test would catch are concrete. A swapped `width` and `height` would make the arena transposed. A broken `equalize` step would make a 0.25 threshold give far less than a quarter water. A missing or ignored seed would make games impossible to reproduce. A broken `fade` or `lerp` function would make the terrain jagged, with lakes and cliffs scattered one cell at a time.

## Concepts you need

**Test discovery.** pytest looks for files named `test_*.py` and, inside them, functions named `test_*`. It runs each one. You never call the test functions yourself.

**Plain `assert`.** A test passes if it reaches the end without an `assert` failing. There is no special assertion library. `assert heights.min() == 0.0` means "stop and report a failure if this is not true".

**numpy helpers used here.** `np.array_equal(a, b)` is True only if two arrays have the same shape and every value matches. `np.isclose(x, 1.0)` allows a tiny floating-point difference. `np.diff(arr, axis=1)` subtracts each cell from the one to its right.

**What a failure looks like.** pytest re-evaluates the failing expression and shows both sides. For example, if the shape assertion in the first test were wrong you would see:

```
>       assert heights.shape == (64, 48)
E       assert (48, 64) == (64, 48)
E         At index 0 diff: 48 != 64
```

**Running a subset.** `-k` filters tests by name. `python -m pytest tests -k seed` runs only tests whose name contains "seed".

## Walkthrough

### `test_grid_is_normalised_to_unit_range()`

```python
def test_grid_is_normalised_to_unit_range():
```

**Setup.** Builds a `PerlinNoise` with `seed=1` and asks for a grid 64 cells wide and 48 cells tall with `scale=16.0`. The other arguments keep their defaults: `octaves=4`, `persistence=0.5`, `lacunarity=2.0`, `equalize=True`.

**`assert heights.shape == (48, 64)`.** numpy arrays are indexed `[row, column]`, which is `[y, x]`. So a map 64 wide and 48 tall has shape `(48, 64)`, not `(64, 48)`. This is the single most common mistake when working with grids. A failure here would mean `grid()` built the coordinate arrays in the wrong order, and every `arena.terrain[y, x]` lookup in the package would be reading the wrong cell.

**`assert heights.min() == 0.0`.** With `equalize=True`, `grid()` replaces every height with its rank among all cells, then divides by `ranks.size - 1`. The lowest cell has rank 0, so its value is exactly `0 / (N - 1) = 0.0`. This is an exact comparison on purpose. If it failed, the equalize step had been changed or skipped, and the water threshold of 0.25 would no longer put a quarter of cells under water.

**`assert np.isclose(heights.max(), 1.0)`.** The highest cell has rank `N - 1`, so its value is `(N - 1) / (N - 1)`. That should be exactly 1.0, but the test uses `np.isclose` to be safe against floating-point division. A failure would mean the top of the range had been clipped or scaled.

**Why these numbers.** 64 by 48 is deliberately not square, so a transposed shape is caught. `scale=16.0` makes hills 16 cells wide, giving several hills in a small grid. `seed=1` is arbitrary; any seed would do.

### `test_same_seed_same_terrain()`

```python
def test_same_seed_same_terrain():
```

**Setup.** Builds two separate `PerlinNoise(seed=42)` objects and asks each for a 32 by 32 grid with default settings. Then builds a third with `seed=43`.

**`assert np.array_equal(first, second)`.** Two generators built from the same seed must produce identical grids, cell for cell. The seed feeds `np.random.default_rng(seed)`, which shuffles the 256-entry permutation table in `__init__`. Same seed, same shuffle, same gradients, same noise. A failure would mean some randomness was leaking in from outside the seed, and `Game` could no longer replay a game from its seed.

**`assert not np.array_equal(first, PerlinNoise(seed=43).grid(32, 32))`.** Different seeds must give different maps. This guards against the opposite bug: a generator that ignores its seed entirely and always produces the same map. Without this line, a `PerlinNoise` that returned a constant grid would pass the first assertion.

**Why these numbers.** 42 and 43 are adjacent so the test also proves that a change of one in the seed is enough to get a different map. 32 by 32 is small enough to be fast.

### `test_noise_is_smooth_not_static()`

```python
def test_noise_is_smooth_not_static():
```

**Setup.** Builds a 100 by 100 grid with `seed=7`, `scale=30.0` and `octaves=1`. A single octave means one layer of noise with no fine detail added, so this is the cleanest possible test of the core `noise()` method.

**`horizontal_steps = np.abs(np.diff(heights, axis=1)).mean()`.** `np.diff` along `axis=1` gives the difference between each cell and its right-hand neighbour. Taking the absolute value and averaging gives one number: the typical height change per step across the map.

**`assert horizontal_steps < 0.05`.** On a 0-to-1 scale, an average step under 0.05 means you need about 20 steps to climb from the lowest to the highest height. That is a hill, not static. Pure random numbers would give an average step of about 0.33. With these settings the real value is about 0.017. A failure would mean the `fade` curve, the `lerp` blend, or the gradient lookup was broken, and the map had lost its smoothness.

**Why these numbers.** `scale=30.0` gives wide hills, so the expected step is small and the threshold of 0.05 leaves a comfortable margin. `octaves=1` removes the extra detail layers, which would roughly double the average step (about 0.037 with four octaves) and make the threshold tighter than it needs to be.

## How to run and extend

```bash
# Run just this file
python -m pytest tests/test_noise.py

# Run one test by its full id
python -m pytest tests/test_noise.py::test_same_seed_same_terrain

# Run tests whose name contains a word
python -m pytest tests -k smooth

# Verbose: one line per test with PASSED or FAILED
python -m pytest tests/test_noise.py -v
```

Ideas for new tests in this area:

**1. `equalize=True` gives a flat distribution.** A quarter of cells should sit below 0.25.

```python
def test_equalized_quarter_below_threshold():
    heights = PerlinNoise(seed=1).grid(80, 80)
    fraction = (heights < 0.25).mean()
    assert abs(fraction - 0.25) < 0.01
```

**2. `equalize=False` still spans 0 to 1.** The non-equalized path stretches the raw values instead of ranking them.

```python
def test_raw_grid_is_stretched_to_unit_range():
    heights = PerlinNoise(seed=1).grid(40, 40, equalize=False)
    assert np.isclose(heights.min(), 0.0)
    assert np.isclose(heights.max(), 1.0)
```

**3. More octaves means rougher terrain.** The average step should grow when detail layers are added.

```python
def test_more_octaves_are_rougher():
    smooth = PerlinNoise(seed=7).grid(100, 100, scale=30.0, octaves=1)
    rough = PerlinNoise(seed=7).grid(100, 100, scale=30.0, octaves=4)
    step = lambda h: np.abs(np.diff(h, axis=1)).mean()
    assert step(rough) > step(smooth)
```

**4. Vertical smoothness too.** The existing test only checks left-to-right steps.

```python
def test_noise_is_smooth_vertically():
    heights = PerlinNoise(seed=7).grid(100, 100, scale=30.0, octaves=1)
    assert np.abs(np.diff(heights, axis=0)).mean() < 0.05
```

## Gotchas

**Seeds are everything.** Every test here passes a seed. If you add a test and forget the seed, `PerlinNoise()` uses fresh entropy and your test may pass one day and fail the next.

**Exact versus approximate.** `heights.min() == 0.0` is an exact comparison and works only because equalize produces `0 / (N - 1)`. If you test the `equalize=False` path, use `np.isclose`, because `(values - low) / (high - low + 1e-12)` never reaches exactly 1.0.

**Shape order.** `grid(width, height)` returns shape `(height, width)`. Write your expected shape as `(rows, columns)`.

**Default arguments.** `test_same_seed_same_terrain` relies on the defaults `scale=40.0` and `octaves=4`. Changing those defaults in `noise.py` will not break the test, but it will change every arena in the package.

**Speed.** All three tests finish in well under a tenth of a second. The 100 by 100 grid is the largest and is still instant because the maths is vectorised in numpy.
