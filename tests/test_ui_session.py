"""Tests for the dashboard's GUI-free parts: the map painter and the session."""

import numpy as np

from hunger_games.config import SimulationConfig
from hunger_games.resources import ResourceKind
from hunger_games.terrain import TerrainType
from hunger_games.training import TrainingConfig
from hunger_games.ui.painter import MapPainter
from hunger_games.ui.session import Session


def test_painter_brush_stamps_and_presets():
    """Painting changes cells, the version bumps, and every preset loads."""
    painter = MapPainter(60, 60)
    before = painter.version
    painter.paint(10, 10, TerrainType.WATER, 2)
    assert painter.terrain[10, 10] == int(TerrainType.WATER)
    assert painter.terrain[10, 13] == int(TerrainType.GRASS)
    assert painter.version > before
    painter.stamp_rectangle(50, 50, 40, 40, TerrainType.ROCK)
    assert painter.terrain[45, 45] == int(TerrainType.ROCK)
    painter.carve_round()
    assert painter.terrain[0, 0] == int(TerrainType.VOID)
    config = SimulationConfig(width=60, height=60, seed=1)
    for name in MapPainter.PRESETS:
        painter.apply_preset(name, config)
        assert painter.terrain.shape == (60, 60)
        assert painter.heights.min() >= 0.0 and painter.heights.max() <= 1.0
    painter.apply_preset("quarter_quell", config)
    coverage = painter.coverage()
    assert coverage["water"] > 0.1 and coverage["sand"] > 0.02


def test_session_edit_play_and_files(tmp_path):
    """The session can edit a map and roster, run a game, scrub it, and save and load everything."""
    session = Session(SimulationConfig(seed=3, width=50, height=50, max_days=6, start_thirst_min=0.3))
    assert len(session.tributes) == 24
    session.place_loot(20, 20, ResourceKind.WEAPON, 1, 0.9)
    session.move_tribute(0, 20, 21)
    session.tribute(0).name = "Rue"
    assert session.tribute_at(20, 21) == 0
    assert session.tribute_at(0, 0) is None or session.tribute_at(0, 0) != 0
    session.new_game()
    session.playing = True
    session.ticks_per_second = 1000
    session.update(0.05)
    assert 0 < session.playhead <= 50
    session.run_to_end()
    assert session.recording.result is not None
    events = len(session.recording.result.eliminations) + len(session.recording.result.gifts)
    assert len(session.event_log(3)) == min(3, events)
    session.seek(3)
    assert session.current_frame.tick == 3
    assert len(session.event_log()) <= events
    session.save_scenario(tmp_path / "s.json")
    session.save_replay(tmp_path / "r.replay")
    session.save_config(tmp_path / "c.json")
    other = Session(SimulationConfig(seed=9, width=50, height=50))
    other.load_scenario(tmp_path / "s.json")
    other.load_replay(tmp_path / "r.replay")
    other.load_config(tmp_path / "c.json")
    assert other.tribute(0).name == "Rue"
    assert other.recording.length == session.recording.length
    assert other.config.max_days == 6


def test_session_training_and_champion():
    """Training runs in the background and the champion can be handed to tributes."""
    session = Session(SimulationConfig(seed=5, width=40, height=40, max_days=3))
    session.start_training(
        TrainingConfig(brain_name="voting", population_size=24, generations=1, rounds_per_generation=1, seed=0)
    )
    session._training_thread.join(timeout=60)
    assert not session.training_running
    assert len(session.training_history()) == 1
    assert session.give_champion([0, 1]) == 2
    assert session.tribute(0).genome is not None and session.tribute(2).genome is None
    session.new_game()
    assert session.game.players[0].brain.name == "voting"
    assert np.allclose(session.game.players[0].brain.genome(), session.trainer.champion)
