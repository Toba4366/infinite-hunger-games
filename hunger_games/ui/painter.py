"""ui/painter.py - a paintable terrain grid.

The Map tab lets a game maker paint water, sand, grass and rock with a
brush, stamp circles and rectangles, cut the arena into a circle, and load
presets like the 75th games' island. `MapPainter` is that grid plus the
editing operations. It has no GUI code, so it is easy to test.
"""

# numpy for the grid.
import numpy as np

# The arena, used to generate a Perlin map and to derive painted heights.
from hunger_games.arena import Arena

# The settings.
from hunger_games.config import SimulationConfig

# The noise generator for texture on painted maps.
from hunger_games.noise import PerlinNoise

# Terrain kinds.
from hunger_games.terrain import TerrainType


class MapPainter:
    """A terrain grid with brushes, stamps and presets."""

    # Names of the presets offered in the dashboard.
    PRESETS = ("perlin", "flat_field", "flat_round", "quarter_quell", "lake_island")

    def __init__(self, width: int, height: int) -> None:
        """Start with a flat grass field."""
        # Width in cells.
        self.width = width
        # Height in cells.
        self.height = height
        # The terrain grid, all grass.
        self.terrain = np.full((height, width), int(TerrainType.GRASS), dtype=np.int8)
        # The height grid, flat.
        self.heights = np.full((height, width), 0.5, dtype=float)
        # Bumps to a new number whenever the map changes, so the canvas knows to redraw.
        self.version = 0
        # Noise used to give painted maps a little texture.
        self._noise = PerlinNoise(0).grid(width, height, scale=20.0, octaves=3)

    # ------------------------------------------------------------ basics

    def _changed(self) -> None:
        """Mark the map as changed."""
        # Bump the version.
        self.version += 1

    def load(self, terrain: np.ndarray, heights: np.ndarray | None = None) -> None:
        """Replace the whole map (for example from an `Arena` or a saved scenario)."""
        # Copy the terrain.
        self.terrain = np.array(terrain, dtype=np.int8)
        # Adopt the new size.
        self.height, self.width = self.terrain.shape
        # Regenerate the texture noise for the new size.
        self._noise = PerlinNoise(0).grid(self.width, self.height, scale=20.0, octaves=3)
        # Use the given heights or derive them.
        self.heights = np.array(heights, dtype=float) if heights is not None else self.derive_heights()
        # Changed.
        self._changed()

    def derive_heights(self) -> np.ndarray:
        """Heights for a painted map: a base per terrain type plus a little noise."""
        # The arena knows how (the same rule a game uses for painted maps).
        return Arena._heights_from_terrain(self.terrain, self._noise)

    def resize(self, width: int, height: int) -> None:
        """Start over at a new size (a flat grass field)."""
        # Rebuild.
        self.__init__(width, height)

    def in_bounds(self, x: int, y: int) -> bool:
        """Is the cell on the grid?"""
        # Check both axes.
        return 0 <= x < self.width and 0 <= y < self.height

    # ----------------------------------------------------------- painting

    def paint(self, x: int, y: int, terrain: TerrainType, radius: int = 1) -> None:
        """Paint a round brush of the given radius centred on (x, y)."""
        # A round stamp is exactly what a brush is.
        self.stamp_circle(x, y, radius, terrain)

    def stamp_circle(self, cx: int, cy: int, radius: float, terrain: TerrainType) -> None:
        """Fill every cell within `radius` of (cx, cy) with the terrain."""
        # Row and column index grids.
        ys, xs = np.indices((self.height, self.width))
        # Cells inside the circle.
        inside = np.hypot(xs - cx, ys - cy) <= radius
        # Paint them.
        self.terrain[inside] = int(terrain)
        # Changed.
        self._changed()

    def stamp_rectangle(self, x0: int, y0: int, x1: int, y1: int, terrain: TerrainType) -> None:
        """Fill the rectangle between two corners (inclusive, any order) with the terrain."""
        # Sort the corners and clip to the grid.
        left, right = max(0, min(x0, x1)), min(self.width - 1, max(x0, x1))
        # Same vertically.
        top, bottom = max(0, min(y0, y1)), min(self.height - 1, max(y0, y1))
        # Paint the slice.
        self.terrain[top : bottom + 1, left : right + 1] = int(terrain)
        # Changed.
        self._changed()

    def stamp_ring(self, cx: int, cy: int, inner: float, outer: float, terrain: TerrainType) -> None:
        """Fill the ring between two radii with the terrain."""
        # Row and column index grids.
        ys, xs = np.indices((self.height, self.width))
        # Distance of every cell from the centre.
        distance = np.hypot(xs - cx, ys - cy)
        # Cells in the ring.
        ring = (distance >= inner) & (distance <= outer)
        # Paint them.
        self.terrain[ring] = int(terrain)
        # Changed.
        self._changed()

    def fill(self, terrain: TerrainType) -> None:
        """Paint the whole grid one terrain type."""
        # Fill.
        self.terrain[:] = int(terrain)
        # Changed.
        self._changed()

    def carve_round(self) -> None:
        """Turn the map into a circle by voiding everything outside the inscribed circle."""
        # The centre and the largest radius that fits.
        cx, cy = self.width // 2, self.height // 2
        # Radius.
        radius = min(self.width, self.height) / 2.0 - 1.0
        # Row and column index grids.
        ys, xs = np.indices((self.height, self.width))
        # Outside the circle.
        outside = np.hypot(xs - cx, ys - cy) > radius
        # Void it.
        self.terrain[outside] = int(TerrainType.VOID)
        # Changed.
        self._changed()

    def finish(self) -> None:
        """Recompute heights after painting so relief shading and downhill match the new map."""
        # Derive.
        self.heights = self.derive_heights()
        # Changed.
        self._changed()

    # ------------------------------------------------------------ presets

    def apply_preset(self, name: str, config: SimulationConfig, seed: int | None = None) -> None:
        """Load one of the named presets."""
        # A fresh Perlin arena, exactly as a normal game would generate.
        if name == "perlin":
            # Build with the given seed (or the config's, or random).
            arena = Arena(config, np.random.default_rng(seed if seed is not None else config.seed))
            # Adopt its terrain and heights.
            self.load(arena.terrain, arena.heights)
            return
        # Every other preset starts from flat grass.
        self.fill(TerrainType.GRASS)
        # A plain square meadow.
        if name == "flat_field":
            pass
        # A plain round meadow.
        elif name == "flat_round":
            self.carve_round()
        # The 75th games: a central island in a sea, a ring of beach, jungle beyond.
        elif name == "quarter_quell":
            # Round arena.
            self.carve_round()
            # Centre and scale.
            cx, cy = self.width // 2, self.height // 2
            # The radius everything is measured against.
            radius = min(self.width, self.height) / 2.0 - 1.0
            # Sea from the centre out to just under half the radius.
            self.stamp_ring(cx, cy, 0.0, radius * 0.45, TerrainType.WATER)
            # The cornucopia island in the middle.
            self.stamp_circle(cx, cy, radius * 0.12, TerrainType.SAND)
            # Beach around the sea.
            self.stamp_ring(cx, cy, radius * 0.45, radius * 0.52, TerrainType.SAND)
            # Twelve thin rocky spokes, like the clock's wedges, so tributes can wade out.
            for spoke in range(12):
                # Angle of this spoke.
                angle = 2.0 * np.pi * spoke / 12.0
                # Single rocks along the spoke from the island to the beach.
                for step in np.linspace(radius * 0.14, radius * 0.43, 5):
                    # Position.
                    x, y = int(round(cx + np.cos(angle) * step)), int(round(cy + np.sin(angle) * step))
                    # One rock.
                    if self.in_bounds(x, y):
                        self.terrain[y, x] = int(TerrainType.ROCK)
        # A single big lake with an island in it, on a square field.
        elif name == "lake_island":
            # Centre and scale.
            cx, cy = self.width // 2, self.height // 2
            # Radius.
            radius = min(self.width, self.height) / 2.0
            # The lake.
            self.stamp_circle(cx, cy, radius * 0.45, TerrainType.WATER)
            # A sandy shore.
            self.stamp_ring(cx, cy, radius * 0.45, radius * 0.5, TerrainType.SAND)
            # The island.
            self.stamp_circle(cx, cy, radius * 0.12, TerrainType.GRASS)
            # Rocky hills in the corners.
            for corner_x, corner_y in ((0, 0), (self.width, 0), (0, self.height), (self.width, self.height)):
                # A quarter-circle of rock.
                self.stamp_circle(corner_x, corner_y, radius * 0.3, TerrainType.ROCK)
        # Unknown names are a mistake worth reporting.
        else:
            raise KeyError(f"Unknown preset '{name}'. Choose from: {', '.join(self.PRESETS)}")
        # Give the painted map its heights.
        self.finish()

    # ----------------------------------------------------------- summary

    def coverage(self) -> dict[str, float]:
        """The fraction of the grid each terrain type covers (for the dashboard)."""
        # Count cells per type, ignoring the void.
        inside = self.terrain != int(TerrainType.VOID)
        # Total cells inside the arena.
        total = max(1, int(inside.sum()))
        # One entry per type.
        return {
            kind.name.lower(): float((self.terrain[inside] == int(kind)).sum() / total)
            for kind in TerrainType
            if kind is not TerrainType.VOID
        }
