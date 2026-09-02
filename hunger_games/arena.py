"""arena.py - the world the tributes fight in.

An `Arena` owns the height map, the terrain grid, the supply grid, and some
pre-computed "where is the nearest water / grass" maps that players use to
navigate. It also knows its own shape, so the 74th-games open field and the
75th-games round arena are both just settings.
"""

# A double-ended queue for the breadth-first search that builds distance maps.
# Trigonometry for placing podiums around a circle.
import math
from collections import deque

# numpy for the grids.
import numpy as np

# The settings this module reads.
from hunger_games.config import ArenaShape, SimulationConfig

# The noise generator that shapes the land.
from hunger_games.noise import PerlinNoise

# The supply grids.
from hunger_games.resources import ResourceGrid

# The terrain types and the classifier.
from hunger_games.terrain import HUNT_DIFFICULTY, MOVE_SUCCESS, TerrainType, classify_heights

# The eight neighbouring cells around any cell, as (dx, dy) steps.
NEIGHBOUR_STEPS = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
# The four orthogonal neighbours, used by the distance-map search.
CROSS_STEPS = [(0, -1), (-1, 0), (1, 0), (0, 1)]


def sign(value: float) -> int:
    """Return -1, 0 or +1 depending on the sign of `value`."""
    # Positive numbers become +1, negatives -1, zero stays 0.
    return (value > 0) - (value < 0)


class Arena:
    """The terrain, supplies and geometry of one Hunger Games arena."""

    # A gentle height for each painted terrain type, so downhill still means something on hand-made maps.
    PAINTED_HEIGHTS = {
        TerrainType.VOID: 0.0,
        TerrainType.WATER: 0.1,
        TerrainType.SAND: 0.3,
        TerrainType.GRASS: 0.5,
        TerrainType.ROCK: 0.8,
    }

    def __init__(self, config: SimulationConfig, rng: np.random.Generator, terrain: np.ndarray | None = None) -> None:
        """Generate a fresh arena from the config, or build one around a painted terrain grid."""
        # Keep the config around for later lookups.
        self.config = config
        # Width in cells.
        self.width = config.width
        # Height in cells.
        self.height = config.height
        # The x coordinate of the middle cell.
        self.center_x = self.width // 2
        # The y coordinate of the middle cell.
        self.center_y = self.height // 2
        # Half the shorter side: the radius of the largest circle that fits the grid.
        self.radius = min(self.width, self.height) / 2.0 - 1.0
        # Give the noise its own seed drawn from the game's generator.
        noise_seed = int(rng.integers(0, 2**31 - 1))
        # Build the Perlin noise generator.
        self.noise = PerlinNoise(noise_seed)
        # More chaos means rougher terrain (extra detail layers stay louder).
        persistence = config.noise.persistence + 0.25 * config.chaos
        # Generate the 0-to-1 height map.
        self.heights = self.noise.grid(
            self.width,
            self.height,
            config.noise.scale,
            config.noise.octaves,
            persistence,
            config.noise.lacunarity,
        )
        # A painted map replaces the generated terrain entirely.
        if terrain is not None:
            # Use the painted grid (copied so later painting does not change a running game).
            self.terrain = np.array(terrain, dtype=np.int8)
            # Height comes from the terrain type plus a little noise for texture.
            self.heights = self._heights_from_terrain(self.terrain, self.heights)
        # Otherwise interpret the generated heights as terrain types.
        else:
            # Classify.
            self.terrain = classify_heights(self.heights, config.terrain)
            # If this is a round arena, cut away the corners of the square grid.
            if config.shape is ArenaShape.ROUND:
                # Mark everything outside the circle as VOID.
                self._carve_circle()
        # Start with an empty supply grid; a layout fills it in later.
        self.resources = ResourceGrid(self.width, self.height)
        # Pre-compute how far every cell is from water and which way to walk.
        self.water_distance, self.water_direction = self._distance_field(TerrainType.WATER)
        # Same for grass, the best hunting ground.
        self.grass_distance, self.grass_direction = self._distance_field(TerrainType.GRASS)

    # ------------------------------------------------------------------ shape

    def _carve_circle(self) -> None:
        """Turn the square grid into a round arena by voiding the corners."""
        # Row and column index grids for every cell.
        ys, xs = np.indices((self.height, self.width))
        # Straight-line distance of every cell from the centre.
        distance = np.hypot(xs - self.center_x, ys - self.center_y)
        # Cells beyond the radius are outside the arena.
        outside = distance > self.radius
        # Mark them as VOID so nobody can stand there.
        self.terrain[outside] = int(TerrainType.VOID)
        # Flatten their height so the renderer shades them evenly.
        self.heights[outside] = 0.0

    @classmethod
    def _heights_from_terrain(cls, terrain: np.ndarray, noise: np.ndarray) -> np.ndarray:
        """Give a painted map plausible heights: a base per terrain type plus 10% noise."""
        # Start flat.
        heights = np.zeros(terrain.shape, dtype=float)
        # Fill in the base height of each terrain type.
        for kind, base in cls.PAINTED_HEIGHTS.items():
            # Cells of this type.
            heights[terrain == int(kind)] = base
        # Add a little of the generated noise so slopes exist within a terrain type.
        heights = heights + 0.1 * (noise - 0.5)
        # Keep the void flat and everything in range.
        heights[terrain == int(TerrainType.VOID)] = 0.0
        # Clamp to 0..1.
        return np.clip(heights, 0.0, 1.0)

    # ---------------------------------------------------------------- queries

    def in_bounds(self, x: int, y: int) -> bool:
        """Is (x, y) inside the grid?"""
        # Check both axes against the grid size.
        return 0 <= x < self.width and 0 <= y < self.height

    def terrain_at(self, x: int, y: int) -> TerrainType:
        """The terrain type of a cell (VOID if off the grid)."""
        # Off-grid counts as void.
        if not self.in_bounds(x, y):
            return TerrainType.VOID
        # Read the grid (note: numpy grids are indexed [row, column] = [y, x]).
        return TerrainType(int(self.terrain[y, x]))

    def is_walkable(self, x: int, y: int) -> bool:
        """Can a player stand here? (Everything but VOID.)"""
        # Any real terrain, including water, can be walked on.
        return self.terrain_at(x, y) is not TerrainType.VOID

    def is_water(self, x: int, y: int) -> bool:
        """Is this cell water?"""
        # Simple comparison against the water type.
        return self.terrain_at(x, y) is TerrainType.WATER

    def is_land(self, x: int, y: int) -> bool:
        """Is this cell dry land inside the arena?"""
        # Walkable and not water.
        return self.is_walkable(x, y) and not self.is_water(x, y)

    def height_at(self, x: int, y: int) -> float:
        """The 0-to-1 height of a cell."""
        # Read the height map.
        return float(self.heights[y, x])

    def hunt_difficulty_at(self, x: int, y: int) -> float:
        """How hard it is to find food in this cell."""
        # Look the terrain's difficulty up in the table.
        return HUNT_DIFFICULTY[self.terrain_at(x, y)]

    def move_success_at(self, x: int, y: int) -> float:
        """The chance that a step into this cell succeeds."""
        # Look the terrain's speed up in the table.
        return MOVE_SUCCESS[self.terrain_at(x, y)]

    def distance_from_center(self, x: int, y: int) -> float:
        """Straight-line distance from the centre, in cells."""
        # Pythagoras.
        return math.hypot(x - self.center_x, y - self.center_y)

    def normalized_distance_from_center(self, x: int, y: int) -> float:
        """Distance from the centre as a fraction: 0.0 at the middle, 1.0 at the edge."""
        # Divide by the radius and clamp so square corners never exceed 1.0.
        return min(1.0, self.distance_from_center(x, y) / self.radius)

    def direction_to_center(self, x: int, y: int) -> tuple[int, int]:
        """A single (dx, dy) step toward the centre."""
        # Step in the direction of the centre along each axis.
        return sign(self.center_x - x), sign(self.center_y - y)

    def downhill_direction(self, x: int, y: int) -> tuple[int, int]:
        """The neighbouring step that drops the most height, or (0, 0) if none is lower."""
        # The best step found so far (none yet).
        best_step = (0, 0)
        # The height to beat: the height where we stand.
        lowest = self.height_at(x, y)
        # Try each of the eight neighbours.
        for dx, dy in NEIGHBOUR_STEPS:
            # The neighbour's coordinates.
            nx, ny = x + dx, y + dy
            # Skip neighbours we cannot stand on.
            if not self.is_walkable(nx, ny):
                continue
            # The neighbour's height.
            candidate = self.height_at(nx, ny)
            # If it is lower than anything seen so far, remember it.
            if candidate < lowest:
                # New lowest height.
                lowest = candidate
                # New best step.
                best_step = (dx, dy)
        # Return the steepest downhill step.
        return best_step

    def direction_to_water(self, x: int, y: int) -> tuple[int, int]:
        """A single (dx, dy) step along the shortest path to water."""
        # Read the pre-computed direction map.
        return int(self.water_direction[y, x, 0]), int(self.water_direction[y, x, 1])

    def distance_to_water(self, x: int, y: int) -> float:
        """Number of steps to the nearest water cell."""
        # Read the pre-computed distance map.
        return float(self.water_distance[y, x])

    def direction_to_grass(self, x: int, y: int) -> tuple[int, int]:
        """A single (dx, dy) step along the shortest path to grass."""
        # Read the pre-computed direction map.
        return int(self.grass_direction[y, x, 0]), int(self.grass_direction[y, x, 1])

    def distance_to_grass(self, x: int, y: int) -> float:
        """Number of steps to the nearest grass cell."""
        # Read the pre-computed distance map.
        return float(self.grass_distance[y, x])

    # -------------------------------------------------------------- placement

    def snap_to_land(self, x: int, y: int) -> tuple[int, int]:
        """Return the nearest dry-land cell to (x, y), searching in growing rings."""
        # If we are already on land, nothing to do.
        if self.is_land(x, y):
            return x, y
        # Search rings of increasing size around the point.
        for ring in range(1, max(self.width, self.height)):
            # Check every cell whose Chebyshev distance equals `ring`.
            for dx in range(-ring, ring + 1):
                # Only the two edge rows of the ring need the full sweep.
                for dy in range(-ring, ring + 1):
                    # Skip cells strictly inside the ring (already checked).
                    if max(abs(dx), abs(dy)) != ring:
                        continue
                    # The candidate cell.
                    nx, ny = x + dx, y + dy
                    # First dry cell found wins.
                    if self.is_land(nx, ny):
                        return nx, ny
        # An arena with no land at all falls back to the centre.
        return self.center_x, self.center_y

    def snap_to_walkable(self, x: int, y: int) -> tuple[int, int]:
        """Return the nearest cell inside the arena (water allowed), searching in growing rings."""
        # Already inside.
        if self.is_walkable(x, y):
            return x, y
        # Search rings of increasing size around the point.
        for ring in range(1, max(self.width, self.height)):
            # Every cell on the ring.
            for dx in range(-ring, ring + 1):
                # Both axes.
                for dy in range(-ring, ring + 1):
                    # Skip cells strictly inside the ring (already checked).
                    if max(abs(dx), abs(dy)) != ring:
                        continue
                    # The candidate cell.
                    nx, ny = x + dx, y + dy
                    # First walkable cell wins.
                    if self.is_walkable(nx, ny):
                        return nx, ny
        # Fall back to the centre.
        return self.center_x, self.center_y

    def snap_to_podium(self, x: int, y: int) -> tuple[int, int]:
        """Snap a podium onto a legal cell: any arena cell if water podiums are allowed, else dry land."""
        # The config decides whether tributes may start in the water.
        if self.config.allow_water_podiums:
            return self.snap_to_walkable(x, y)
        # Otherwise dry land only.
        return self.snap_to_land(x, y)

    def edge_positions(self, count: int) -> list[tuple[int, int]]:
        """Evenly spaced podiums along the outer edge of the arena."""
        # Collect the positions here.
        positions = []
        # Round arenas: walk around a circle just inside the boundary.
        if self.config.shape is ArenaShape.ROUND:
            # One podium per player.
            for index in range(count):
                # Angle around the circle for this podium.
                angle = 2.0 * math.pi * index / count
                # Stay a few cells inside the boundary.
                ring_radius = self.radius - 3.0
                # x position on the circle.
                x = int(round(self.center_x + math.cos(angle) * ring_radius))
                # y position on the circle.
                y = int(round(self.center_y + math.sin(angle) * ring_radius))
                # Nudge onto the arena if needed.
                positions.append(self.snap_to_podium(x, y))
            # Done with the round case.
            return positions
        # Open-field arenas: walk around the rectangle a few cells inside the edge.
        inset = 3
        # The rectangle's inner width and height.
        inner_w = self.width - 2 * inset
        # Inner height.
        inner_h = self.height - 2 * inset
        # Total length of the rectangle's perimeter.
        perimeter = 2 * (inner_w + inner_h)
        # One podium per player.
        for index in range(count):
            # How far along the perimeter this podium sits.
            t = perimeter * index / count
            # Top edge, left to right.
            if t < inner_w:
                x, y = inset + t, inset
            # Right edge, top to bottom.
            elif t < inner_w + inner_h:
                x, y = inset + inner_w, inset + (t - inner_w)
            # Bottom edge, right to left.
            elif t < 2 * inner_w + inner_h:
                x, y = inset + inner_w - (t - inner_w - inner_h), inset + inner_h
            # Left edge, bottom to top.
            else:
                x, y = inset, inset + inner_h - (t - 2 * inner_w - inner_h)
            # Nudge onto the arena if needed.
            positions.append(self.snap_to_podium(int(round(x)), int(round(y))))
        # Hand back all the podiums.
        return positions

    # ----------------------------------------------------------- navigation

    def _distance_field(self, target: TerrainType) -> tuple[np.ndarray, np.ndarray]:
        """Breadth-first search outward from every cell of `target` type.

        Returns two grids: the number of steps from each cell to the nearest
        target cell, and the (dx, dy) step that leads there.
        """
        # Start every cell at "infinitely far".
        distance = np.full((self.height, self.width), np.inf, dtype=float)
        # Start every cell with no direction.
        direction = np.zeros((self.height, self.width, 2), dtype=np.int8)
        # Find all the target cells (the "sources" of the search).
        source_ys, source_xs = np.nonzero(self.terrain == int(target))
        # A queue of cells still to expand.
        queue = deque()
        # Seed the queue with every source cell at distance zero.
        for sx, sy in zip(source_xs.tolist(), source_ys.tolist(), strict=False):
            # Sources are zero steps from themselves.
            distance[sy, sx] = 0.0
            # Put them in the queue.
            queue.append((sx, sy))
        # Keep expanding until nothing is left.
        while queue:
            # Take the oldest cell out of the queue.
            x, y = queue.popleft()
            # Look at its four neighbours.
            for dx, dy in CROSS_STEPS:
                # The neighbour's coordinates.
                nx, ny = x + dx, y + dy
                # Skip cells nobody can stand on.
                if not self.is_walkable(nx, ny):
                    continue
                # Skip cells we already reached (BFS reaches each cell first by the shortest path).
                if distance[ny, nx] != np.inf:
                    continue
                # The neighbour is one step further than the current cell.
                distance[ny, nx] = distance[y, x] + 1.0
                # To get from the neighbour back toward the source, reverse the step.
                direction[ny, nx] = (-dx, -dy)
                # Queue the neighbour so its own neighbours get expanded.
                queue.append((nx, ny))
        # Return both maps.
        return distance, direction
