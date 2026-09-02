"""noise.py - a from-scratch implementation of 2D Perlin noise.

Perlin noise is the "semi-random terrain that follows a clear pattern" from
chapter 4. It is the same technique Minecraft uses to build its worlds.

The idea in one paragraph: lay an invisible grid over the map and give every
grid corner a random arrow (a "gradient"). For any point on the map, look at
the four corners of the grid square it sits in, measure how well the point
lines up with each corner's arrow, and blend those four measurements
together smoothly. Nearby points share corners, so they get similar values,
which is why the result looks like rolling hills instead of TV static.
"""

# numpy does fast maths on whole grids at once instead of one cell at a time.
import numpy as np


class PerlinNoise:
    """Generates repeatable, smooth, random-looking 2D noise."""

    # The eight possible arrow directions a grid corner can point in.
    GRADIENTS = np.array(
        [[1, 1], [-1, 1], [1, -1], [-1, -1], [1, 0], [-1, 0], [0, 1], [0, -1]],
        dtype=float,
    )

    def __init__(self, seed: int | None = None) -> None:
        """Build the shuffled lookup table that makes this noise unique."""
        # A random number generator that will give the same results for the same seed.
        rng = np.random.default_rng(seed)
        # Shuffle the numbers 0..255 into a random order (the classic Perlin "permutation").
        permutation = rng.permutation(256)
        # Repeat the table twice so we can index it with values up to 511 without wrapping.
        self.perm = np.concatenate([permutation, permutation])

    @staticmethod
    def fade(t: np.ndarray) -> np.ndarray:
        """Ken Perlin's smoothing curve: 6t^5 - 15t^4 + 10t^3."""
        # This S-shaped curve eases in and out so the blend has no visible seams.
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    @staticmethod
    def lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Linear interpolation: slide from `a` (t=0) to `b` (t=1)."""
        # A straight-line blend between two values.
        return a + t * (b - a)

    def _corner_gradient(self, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
        """Look up the random arrow assigned to grid corner (ix, iy)."""
        # Hash the two integer coordinates through the shuffled table to get a repeatable "random" number.
        hashed = self.perm[(self.perm[ix % 256] + iy) % 256]
        # Reduce that number to 0..7 to pick one of the eight arrows.
        index = hashed % len(self.GRADIENTS)
        # Return the chosen arrow for every input point at once.
        return self.GRADIENTS[index]

    def noise(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Classic single-layer Perlin noise at the given (x, y) coordinates."""
        # Make sure we are working with float numpy arrays, even if given plain numbers.
        x = np.asarray(x, dtype=float)
        # Same for y.
        y = np.asarray(y, dtype=float)
        # The grid corner to the bottom-left of each point.
        x0 = np.floor(x).astype(int)
        # Same for the y coordinate.
        y0 = np.floor(y).astype(int)
        # The grid corner to the right is one step over.
        x1 = x0 + 1
        # The grid corner above is one step up.
        y1 = y0 + 1
        # How far the point is inside its grid square horizontally (0.0 to 1.0).
        dx = x - x0
        # How far the point is inside its grid square vertically (0.0 to 1.0).
        dy = y - y0
        # Fetch the arrow at the bottom-left corner.
        g00 = self._corner_gradient(x0, y0)
        # Dot product: how well the offset from that corner lines up with its arrow.
        n00 = g00[..., 0] * dx + g00[..., 1] * dy
        # Fetch the arrow at the bottom-right corner.
        g10 = self._corner_gradient(x1, y0)
        # Offset from the bottom-right corner is (dx - 1, dy).
        n10 = g10[..., 0] * (dx - 1.0) + g10[..., 1] * dy
        # Fetch the arrow at the top-left corner.
        g01 = self._corner_gradient(x0, y1)
        # Offset from the top-left corner is (dx, dy - 1).
        n01 = g01[..., 0] * dx + g01[..., 1] * (dy - 1.0)
        # Fetch the arrow at the top-right corner.
        g11 = self._corner_gradient(x1, y1)
        # Offset from the top-right corner is (dx - 1, dy - 1).
        n11 = g11[..., 0] * (dx - 1.0) + g11[..., 1] * (dy - 1.0)
        # Smooth the horizontal blend amount.
        u = self.fade(dx)
        # Smooth the vertical blend amount.
        v = self.fade(dy)
        # Blend the two bottom corners left-to-right.
        bottom = self.lerp(n00, n10, u)
        # Blend the two top corners left-to-right.
        top = self.lerp(n01, n11, u)
        # Blend bottom-to-top for the final value (roughly between -0.7 and 0.7).
        return self.lerp(bottom, top, v)

    def fractal(
        self,
        x: np.ndarray,
        y: np.ndarray,
        octaves: int = 4,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
    ) -> np.ndarray:
        """Stack several layers ("octaves") of noise for natural-looking detail.

        The first layer makes big hills, the next adds smaller bumps on top,
        the next adds pebbles, and so on. Minecraft does exactly this.
        """
        # Running total of all the layers added together.
        total = np.zeros(np.broadcast(x, y).shape, dtype=float)
        # How loud the current layer is (the first layer is full volume).
        amplitude = 1.0
        # How zoomed-in the current layer is (the first layer is zoomed out).
        frequency = 1.0
        # Sum of all amplitudes, used to scale the result back down at the end.
        max_amplitude = 0.0
        # Add one layer per octave.
        for _ in range(octaves):
            # Sample this layer at the current zoom level and add it at the current volume.
            total = total + self.noise(x * frequency, y * frequency) * amplitude
            # Remember how much volume we have added in total.
            max_amplitude += amplitude
            # The next layer is quieter...
            amplitude *= persistence
            # ...and more finely detailed.
            frequency *= lacunarity
        # Divide by the total volume so the result stays in the same range as one layer.
        return total / max_amplitude

    def grid(
        self,
        width: int,
        height: int,
        scale: float = 40.0,
        octaves: int = 4,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        equalize: bool = True,
    ) -> np.ndarray:
        """Produce a height map of shape (height, width) with values from 0.0 to 1.0.

        With `equalize` on, the heights are spread out evenly so that a
        threshold of 0.25 really does put a quarter of the cells below it.
        Raw fractal noise bunches up around the middle, which would make the
        chapter 4 thresholds give far less water and rock than intended.
        """
        # Column coordinates, divided by `scale` so one hill spans `scale` cells.
        xs = np.arange(width) / scale
        # Row coordinates, scaled the same way.
        ys = np.arange(height) / scale
        # Turn the two 1D coordinate lists into two 2D coordinate grids.
        grid_x, grid_y = np.meshgrid(xs, ys)
        # Sample layered noise at every grid cell in one vectorised call.
        values = self.fractal(grid_x, grid_y, octaves, persistence, lacunarity)
        # Optionally replace each height by its rank, which flattens the distribution.
        if equalize:
            # argsort twice gives each cell its rank from 0 (lowest) to N-1 (highest).
            ranks = values.ravel().argsort().argsort()
            # Scale the ranks to 0.0..1.0 and restore the grid shape.
            return (ranks / (ranks.size - 1)).reshape(values.shape)
        # The lowest value present in this particular map.
        low = values.min()
        # The highest value present in this particular map.
        high = values.max()
        # Stretch the values so the lowest becomes 0.0 and the highest becomes 1.0.
        return (values - low) / (high - low + 1e-12)
