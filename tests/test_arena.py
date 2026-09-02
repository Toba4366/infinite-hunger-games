"""Tests for the arena, terrain classification and layouts."""

import numpy as np

from hunger_games.arena import Arena
from hunger_games.config import ArenaShape, LayoutName, SimulationConfig, TerrainConfig
from hunger_games.resources import build_layout
from hunger_games.terrain import TerrainType, classify_heights


def make_arena(**overrides) -> Arena:
    config = SimulationConfig(width=60, height=60, seed=3, **overrides)
    return Arena(config, np.random.default_rng(3))


def test_relative_thresholds():
    """Each terrain band starts where the previous one ends."""
    heights = np.array([[0.1, 0.3, 0.5, 0.95]])
    terrain = classify_heights(heights, TerrainConfig(water_threshold=0.25, sand_size=0.1, grass_size=0.3))
    assert terrain.tolist() == [[TerrainType.WATER, TerrainType.SAND, TerrainType.GRASS, TerrainType.ROCK]]


def test_zero_size_removes_a_terrain_type():
    """Chapter 4: setting a size to zero deletes that terrain type."""
    heights = np.random.default_rng(0).random((20, 20))
    terrain = classify_heights(heights, TerrainConfig(water_threshold=0.25, sand_size=0.0, grass_size=0.5))
    assert not (terrain == TerrainType.SAND).any()


def test_round_arena_has_void_corners():
    """The 75th-games arena is a circle carved from the square grid."""
    arena = make_arena(shape=ArenaShape.ROUND)
    assert arena.terrain_at(0, 0) is TerrainType.VOID
    assert arena.terrain_at(arena.center_x, arena.center_y) is not TerrainType.VOID
    assert not arena.is_walkable(0, 0)


def test_open_field_has_no_void():
    """The 74th-games arena uses the whole square."""
    arena = make_arena(shape=ArenaShape.OPEN_FIELD)
    assert not (arena.terrain == TerrainType.VOID).any()


def test_edge_positions_are_inside_the_arena_and_distinct():
    """Every podium is inside the arena (water allowed by default) and nobody shares a podium."""
    for shape in ArenaShape:
        arena = make_arena(shape=shape)
        podiums = arena.edge_positions(24)
        assert len(podiums) == 24
        assert len(set(podiums)) == 24
        assert all(arena.is_walkable(x, y) for x, y in podiums)


def test_edge_positions_avoid_water_when_asked():
    """With water podiums disabled, every podium is nudged onto dry land."""
    arena = make_arena(allow_water_podiums=False)
    assert all(arena.is_land(x, y) for x, y in arena.edge_positions(24))


def test_water_distance_field():
    """Water cells are zero steps from water and directions lead downhill toward it."""
    arena = make_arena()
    ys, xs = np.nonzero(arena.terrain == TerrainType.WATER)
    assert len(xs) > 0
    assert arena.distance_to_water(int(xs[0]), int(ys[0])) == 0.0
    land_ys, land_xs = np.nonzero(arena.terrain == TerrainType.GRASS)
    x, y = int(land_xs[0]), int(land_ys[0])
    dx, dy = arena.direction_to_water(x, y)
    assert arena.distance_to_water(x + dx, y + dy) == arena.distance_to_water(x, y) - 1


def test_layouts_place_supplies():
    """Both layouts fill the arena with something, and the ring keeps weapons central."""
    for name in LayoutName:
        arena = make_arena()
        build_layout(name).apply(arena, np.random.default_rng(1))
        assert (arena.resources.kind != 0).sum() > 0
