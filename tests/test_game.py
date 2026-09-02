"""Tests for a full game."""

from hunger_games.config import ArenaShape, LayoutName, SimulationConfig
from hunger_games.game import Game


def small_config(**overrides) -> SimulationConfig:
    settings = dict(width=60, height=60, seed=11, max_days=10)
    settings.update(overrides)
    return SimulationConfig(**settings)


def test_game_runs_to_completion_with_consistent_bookkeeping():
    """Placements are unique, eliminations match deaths, and the victor is placed first."""
    result = Game(small_config()).run()
    dead = [player for player in result.players if not player.alive_at_end]
    survivors = [player for player in result.players if player.alive_at_end]
    assert len(result.eliminations) == len(dead)
    assert sorted(player.placement for player in dead) == list(range(len(survivors) + 1, 25))
    assert all(player.placement == len(survivors) for player in survivors)
    if result.winner_id is not None:
        assert len(survivors) == 1 and survivors[0].placement == 1


def test_same_seed_reproduces_the_same_game():
    """Chaos is random but seeded: the same seed must replay exactly."""
    first = Game(small_config(chaos=1.0)).run()
    second = Game(small_config(chaos=1.0)).run()
    assert first.elimination_rows() == second.elimination_rows()


def test_all_shapes_and_layouts_run():
    """Every combination of arena shape and supply layout plays without error."""
    for shape in ArenaShape:
        for layout in LayoutName:
            result = Game(small_config(shape=shape, layout=layout, max_days=4)).run()
            assert result.ticks > 0


def test_strong_players_spread_apart():
    """Chapter 2: the top scorers should not start on neighbouring podiums."""
    game = Game(small_config())
    ordered = Game._spread_high_scorers(game.players)
    ranked = sorted(game.players, key=lambda player: player.training_score, reverse=True)
    top = set(ranked[: len(game.players) // 3])
    top_indices = [index for index, player in enumerate(ordered) if player in top]
    assert all(b - a >= 3 for a, b in zip(top_indices, top_indices[1:], strict=False))
