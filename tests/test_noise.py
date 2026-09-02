"""Tests for the Perlin noise generator."""

import numpy as np

from hunger_games.noise import PerlinNoise


def test_grid_is_normalised_to_unit_range():
    """The height map must span exactly 0.0 to 1.0."""
    heights = PerlinNoise(seed=1).grid(64, 48, scale=16.0)
    assert heights.shape == (48, 64)
    assert heights.min() == 0.0
    assert np.isclose(heights.max(), 1.0)


def test_same_seed_same_terrain():
    """Perlin noise is repeatable: the same seed gives the same map."""
    first = PerlinNoise(seed=42).grid(32, 32)
    second = PerlinNoise(seed=42).grid(32, 32)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, PerlinNoise(seed=43).grid(32, 32))


def test_noise_is_smooth_not_static():
    """Neighbouring cells should differ only slightly (rolling hills, not TV static)."""
    heights = PerlinNoise(seed=7).grid(100, 100, scale=30.0, octaves=1)
    horizontal_steps = np.abs(np.diff(heights, axis=1)).mean()
    assert horizontal_steps < 0.05
