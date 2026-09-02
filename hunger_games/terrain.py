"""terrain.py - turning raw heights into water, sand, grass and rock.

This is the "interpret the heights" step of chapter 4, including the trick
of defining each threshold relative to the one before it.
"""

# IntEnum members behave like integers, so we can store them in a numpy grid.
from enum import IntEnum

# numpy for grid-wide comparisons.
import numpy as np

# The threshold settings this module reads.
from hunger_games.config import TerrainConfig


class TerrainType(IntEnum):
    """The kinds of ground a cell can be."""

    # Outside the arena entirely (used to carve a round arena out of a square grid).
    VOID = 0
    # Lakes and rivers: drinkable, hard to hunt in, slow to cross.
    WATER = 1
    # Beaches: poor hunting, nothing to hide behind.
    SAND = 2
    # Meadows and forest: the easiest hunting in the arena.
    GRASS = 3
    # Mountains: very hard hunting, but the high ground.
    ROCK = 4


# How hard it is to catch food on each terrain (0.0 = trivial, 1.0 = impossible).
# Chapter 4 uses 0.2 for grass and 0.6 for water; sand and rock are our additions.
HUNT_DIFFICULTY = {
    TerrainType.VOID: 1.0,
    TerrainType.WATER: 0.6,
    TerrainType.SAND: 0.8,
    TerrainType.GRASS: 0.2,
    TerrainType.ROCK: 0.9,
}

# The chance a step *into* each terrain succeeds (1.0 = always, 0.5 = half the time).
# Wading through water or scrambling over rock is slow, which is how chases get resolved.
MOVE_SUCCESS = {
    TerrainType.VOID: 0.0,
    TerrainType.WATER: 0.5,
    TerrainType.SAND: 0.85,
    TerrainType.GRASS: 1.0,
    TerrainType.ROCK: 0.6,
}

# The RGB colour (each channel 0.0 to 1.0) used when drawing each terrain type.
TERRAIN_COLORS = {
    # Near-black with a hint of blue: the nothingness outside a round arena.
    TerrainType.VOID: (0.05, 0.05, 0.08),
    # Lake blue: low red, medium green, strong blue.
    TerrainType.WATER: (0.16, 0.42, 0.80),
    # Beach tan: high red and green, less blue makes it look yellow-brown.
    TerrainType.SAND: (0.86, 0.78, 0.52),
    # Meadow green: green channel dominates, red and blue held back.
    TerrainType.GRASS: (0.30, 0.62, 0.28),
    # Mid grey: all three channels equal, so no colour tint at all.
    TerrainType.ROCK: (0.50, 0.50, 0.50),
}


def classify_heights(heights: np.ndarray, config: TerrainConfig) -> np.ndarray:
    """Convert a 0-to-1 height map into a grid of TerrainType values."""
    # Water starts at the bottom and ends at the water threshold.
    water_threshold = config.water_threshold
    # Sand starts where water ends and lasts for `sand_size` worth of height.
    sand_threshold = water_threshold + config.sand_size
    # Grass starts where sand ends and lasts for `grass_size` worth of height.
    grass_threshold = sand_threshold + config.grass_size
    # Start by calling every cell rock; the checks below overwrite the lower cells.
    terrain = np.full(heights.shape, int(TerrainType.ROCK), dtype=np.int8)
    # Everything below the grass threshold is grass (for now).
    terrain[heights < grass_threshold] = int(TerrainType.GRASS)
    # Everything below the sand threshold is sand (overwriting some grass).
    terrain[heights < sand_threshold] = int(TerrainType.SAND)
    # Everything below the water threshold is water (overwriting some sand).
    terrain[heights < water_threshold] = int(TerrainType.WATER)
    # Hand back the finished terrain grid.
    return terrain
