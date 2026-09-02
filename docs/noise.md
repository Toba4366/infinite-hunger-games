# `noise.py`

**Source:** [hunger_games/noise.py](../hunger_games/noise.py)
**Depends on:** `numpy` (third-party). No project modules.
**Used by:** [arena.py](arena.md) and `tests/test_noise.py`.

## Purpose

`noise.py` builds the height map that every arena is made from. It is a from-scratch implementation of 2D Perlin noise, the technique Minecraft uses to shape its worlds. Chapter 4 of the video describes wanting "semi-random terrain that follows a clear pattern": not pure static, but rolling hills with a bit of randomness. Perlin noise is exactly that.

The single class, `PerlinNoise`, does three jobs in increasing order of usefulness. `noise()` samples one smooth layer at any set of coordinates. `fractal()` stacks several layers of different sizes on top of each other for natural detail. `grid()` wraps it all up and returns a ready-to-use height map of shape `(height, width)` with values from `0.0` to `1.0`.

In the pipeline, [arena.py](arena.md) creates a `PerlinNoise` with a seed, calls `grid()` with the settings from `NoiseConfig` in [config.py](config.md), and passes the result to `classify_heights()` in [terrain.py](terrain.md), which turns heights into water, sand, grass and rock.

The idea in one paragraph, taken from the source: lay an invisible grid over the map and give every grid corner a random arrow (a gradient). For any point, look at the four corners of the square it sits in, measure how well the point lines up with each corner's arrow, and blend those four measurements together smoothly. Nearby points share corners, so they get similar values, which is why the result looks like hills instead of TV static.

## Concepts you need

**numpy arrays and vectorisation.** Instead of looping over every cell, numpy does the same maths on a whole grid at once. `np.floor(x)` floors every element of `x`. `a * b` multiplies element by element. Every method here accepts whole arrays and returns whole arrays.

**Broadcasting and fancy indexing.** `self.perm[some_array]` looks up many table entries at once and returns an array of the same shape. `g00[..., 0]` takes the first component of every arrow in an array of arrows. The `...` means "all the leading dimensions".

**`np.random.default_rng(seed)`.** A random number generator. Given the same seed it produces the same sequence every time, which is how the same seed reproduces the same map.

**Dot product.** For two arrows `(a, b)` and `(c, d)`, the dot product is `a*c + b*d`. It is large and positive when the arrows point the same way, negative when they point opposite ways, and zero when they are at right angles. Perlin noise uses it to measure how well a point's offset from a corner lines up with that corner's random arrow.

**Linear interpolation (lerp).** Sliding smoothly from value `a` to value `b` as `t` goes from `0` to `1`: `a + t * (b - a)`.

**Fade curve.** Plain lerp gives visible creases at grid lines. Ken Perlin's curve `6t^5 - 15t^4 + 10t^3` is an S-shape that starts and ends flat, so the blend has no seams.

**Octaves.** Musical octaves double the frequency. Here each octave doubles (by default) how fine the noise is and halves how loud it is. Adding octaves adds detail.

**`@staticmethod`.** A method that does not need `self`. `fade` and `lerp` are pure maths, so they are static.

**Rank equalization.** Replacing each value by its rank (its position in sorted order) spreads the values out evenly. `argsort().argsort()` is the standard numpy trick for computing ranks.

## Walkthrough

### `PerlinNoise`

```python
class PerlinNoise:
```

Generates repeatable, smooth, random-looking 2D noise. One instance holds one shuffled lookup table, so one instance means one unique world.

#### `GRADIENTS` (class attribute)

```python
GRADIENTS = np.array(
    [[1, 1], [-1, 1], [1, -1], [-1, -1], [1, 0], [-1, 0], [0, 1], [0, -1]],
    dtype=float,
)
```

The eight arrows a grid corner can point in: the four diagonals and the four axis directions. Every corner picks one of these. It is a class attribute, shared by all instances, because the set of arrows never changes.

#### `__init__`

```python
def __init__(self, seed: int | None = None) -> None
```

1. Makes a random generator from `seed`.
2. Shuffles the numbers `0..255` into a random order. This is the classic Perlin "permutation table".
3. Stores the table twice in a row as `self.perm` (length 512), so indices up to 511 work without wrapping.

`seed=None` gives a different table every time.

```python
from hunger_games.noise import PerlinNoise

a = PerlinNoise(seed=1)
b = PerlinNoise(seed=1)
print((a.perm == b.perm).all())   # True: same seed, same table
```

#### `fade`

```python
@staticmethod
def fade(t: np.ndarray) -> np.ndarray
```

Returns `6t^5 - 15t^4 + 10t^3`, computed as `t*t*t*(t*(t*6 - 15) + 10)`. Input `t` is expected in `0..1`. Output is also `0..1`, but it starts slowly, speeds up in the middle and ends slowly. At `t=0` it returns `0`, at `t=1` it returns `1`, at `t=0.5` it returns `0.5`.

```python
import numpy as np
print(PerlinNoise.fade(np.array([0.0, 0.25, 0.5, 0.75, 1.0])))
# [0.       0.103516 0.5      0.896484 1.      ]
```

#### `lerp`

```python
@staticmethod
def lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray
```

Returns `a + t * (b - a)`: `a` when `t=0`, `b` when `t=1`, halfway when `t=0.5`.

```python
print(PerlinNoise.lerp(10.0, 20.0, 0.25))   # 12.5
```

#### `_corner_gradient`

```python
def _corner_gradient(self, ix: np.ndarray, iy: np.ndarray) -> np.ndarray
```

Looks up the arrow assigned to grid corner `(ix, iy)`. Both inputs are integer arrays.

1. `self.perm[ix % 256]` hashes the x coordinate to a number in `0..255`.
2. Adding `iy` and taking `% 256` mixes in the y coordinate, then `self.perm[...]` hashes again.
3. `hashed % 8` picks one of the eight `GRADIENTS`.

The result has one extra trailing dimension of size 2 (the arrow), so for input shape `(H, W)` you get `(H, W, 2)`. The leading underscore means "internal helper; not part of the public interface".

Why hash instead of storing an arrow per corner? Hashing means the arrow for any corner, at any coordinate, can be computed on demand with no memory, and the same corner always gets the same arrow.

#### `noise`

```python
def noise(self, x: np.ndarray, y: np.ndarray) -> np.ndarray
```

Classic single-layer Perlin noise at coordinates `(x, y)`. Coordinates are in "grid units": one unit is one cell of the invisible corner grid.

Step by step:

1. Convert `x` and `y` to float arrays.
2. `x0, y0 = floor(x), floor(y)`: the bottom-left corner of the square each point is in. `x1, y1` are one step right and one step up.
3. `dx, dy = x - x0, y - y0`: how far into the square the point is, each in `0..1`.
4. Fetch the arrows at the four corners (`g00`, `g10`, `g01`, `g11`).
5. For each corner, compute the dot product of its arrow with the offset from that corner to the point. The offsets are `(dx, dy)`, `(dx-1, dy)`, `(dx, dy-1)` and `(dx-1, dy-1)`. These give `n00`, `n10`, `n01`, `n11`.
6. Smooth the blend amounts: `u = fade(dx)`, `v = fade(dy)`.
7. Blend the two bottom corners left-to-right, the two top corners left-to-right, then blend bottom-to-top.

The output is roughly in `-0.7..0.7`. It is exactly `0` at every integer grid point, because there `dx = dy = 0` and the dot product with a zero offset is zero.

```python
import numpy as np
p = PerlinNoise(seed=3)
print(p.noise(np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 0.0])))
# first and last are 0.0 (integer corners); the middle is some value in -0.7..0.7
```

#### `fractal`

```python
def fractal(
    self,
    x: np.ndarray,
    y: np.ndarray,
    octaves: int = 4,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
) -> np.ndarray
```

Stacks several layers of `noise()`.

1. Start `total` at zeros, `amplitude = 1.0`, `frequency = 1.0`, `max_amplitude = 0.0`.
2. For each octave: add `noise(x * frequency, y * frequency) * amplitude` to `total`. Add `amplitude` to `max_amplitude`. Then multiply `amplitude` by `persistence` and `frequency` by `lacunarity`.
3. Return `total / max_amplitude`.

With the defaults the layers have amplitudes `1, 0.5, 0.25, 0.125` and frequencies `1, 2, 4, 8`. The first layer makes the big hills, the last adds pebbles. Dividing by `max_amplitude` (here `1.875`) keeps the result in the same range as a single layer.

| Parameter | Default | Effect |
| --- | --- | --- |
| `octaves` | `4` | Number of layers. |
| `persistence` | `0.5` | Volume multiplier per layer. Higher means rougher. |
| `lacunarity` | `2.0` | Frequency multiplier per layer. |

#### `grid`

```python
def grid(
    self,
    width: int,
    height: int,
    scale: float = 40.0,
    octaves: int = 4,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    equalize: bool = True,
) -> np.ndarray
```

Produces a height map of shape `(height, width)` with values from `0.0` to `1.0`.

1. `xs = arange(width) / scale` and `ys = arange(height) / scale`. Dividing by `scale` means one grid unit of noise (one "hill") spans `scale` cells.
2. `np.meshgrid(xs, ys)` turns the two 1D lists into two 2D coordinate grids of shape `(height, width)`.
3. Call `fractal()` on the whole grid at once.
4. If `equalize` is `True`: rank every cell (`argsort().argsort()` on the flattened array), divide by `size - 1`, and reshape. The lowest cell becomes exactly `0.0`, the highest exactly `1.0`, and every value in between is spread evenly.
5. If `equalize` is `False`: stretch the raw values so the minimum is `0.0` and the maximum is `1.0`. The `1e-12` guards against dividing by zero on a flat map.

Why equalize? Raw fractal noise bunches up around the middle. A threshold of `0.25` would then give far less than a quarter water. With equalization, a threshold of `0.25` puts exactly a quarter of the cells below it, so the chapter 4 thresholds mean what they say.

```python
heights = PerlinNoise(seed=1).grid(64, 48, scale=16.0)
print(heights.shape)            # (48, 64): rows first, then columns
print(heights.min(), heights.max())   # 0.0 1.0
print((heights < 0.25).mean())  # about 0.25 because of equalization
```

## How to use it / experiment

**Look at a height map.**

```python
import matplotlib.pyplot as plt
from hunger_games.noise import PerlinNoise

heights = PerlinNoise(seed=5).grid(120, 120, scale=40.0, octaves=5)
plt.imshow(heights, cmap="terrain")
plt.colorbar()
plt.show()
```

**Compare one octave with many.** Run the snippet above with `octaves=1` and then `octaves=6`. One octave is smooth blobs; six octaves adds fine texture.

**Change `scale`.** `scale=10` gives many small hills. `scale=100` gives one or two huge ones across a 120-cell map.

**Turn off equalization** and see how the raw distribution bunches in the middle.

```python
raw = PerlinNoise(seed=5).grid(120, 120, equalize=False)
print((raw < 0.25).mean())   # much less than 0.25
```

**Check smoothness**, like the test suite does.

```python
import numpy as np
h = PerlinNoise(seed=7).grid(100, 100, scale=30.0, octaves=1)
print(np.abs(np.diff(h, axis=1)).mean())   # small, well under 0.05
```

**Sample a single point** rather than a grid, for example to drive a 1D effect.

```python
p = PerlinNoise(seed=0)
print(p.fractal(np.linspace(0, 4, 9), np.zeros(9), octaves=3))
```

**Future projects.**

- *New terrain type:* nothing changes here. Heights stay `0..1`; add the new band in [terrain.py](terrain.md).
- *Rivers, cliffs or islands:* you could post-process the `grid()` output, for example by multiplying by a radial falloff to make an island, before it reaches `classify_heights`.
- *Genetic algorithm or neural brain:* neither touches this file. They only care that the same seed reproduces the same map, which `PerlinNoise(seed)` guarantees.
- *Different noise:* a `SimplexNoise` class with the same `grid()` signature could be swapped into `Arena.__init__` without any other change.

## Gotchas

- **Output shape is `(height, width)`**, rows first. `heights[y, x]` is the cell at column `x`, row `y`. Every grid in this project uses that order.
- **Coordinates are in grid units, not cells.** `noise(x, y)` at integer inputs always returns `0`. `grid()` divides cell indices by `scale` so cells sit between corners.
- **Single-layer `noise()` is not in `0..1`.** It is roughly `-0.7..0.7`. Only `grid()` normalises.
- **`grid()` defaults to 4 octaves but `NoiseConfig` defaults to 5.** `Arena` passes the config value, so real games use 5.
- **Equalized maps always contain the full range.** Even a nearly flat noise field is stretched to `0.0..1.0`, so thresholds always produce every band (unless a band size is `0`).
- **Ties in equalization** are broken arbitrarily by `argsort`. This only matters for exactly equal raw values, which is rare with floats.
- **`persistence > 1` makes each layer louder than the last.** The result is still normalised, but the fine detail swamps the hills.
- **Very large `x` or `y`** still work because of `% 256`, but the pattern repeats every 256 grid units.
