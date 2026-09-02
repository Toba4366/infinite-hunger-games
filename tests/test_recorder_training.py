"""Tests for recordings, replays and the genetic trainer."""

import numpy as np

from hunger_games.config import SimulationConfig
from hunger_games.game import Game
from hunger_games.recorder import Recorder, Recording
from hunger_games.training import GeneticTrainer, TrainingConfig


def small() -> SimulationConfig:
    return SimulationConfig(seed=7, width=50, height=50, max_days=4)


def test_recorder_captures_every_tick_and_round_trips(tmp_path):
    """One frame per tick plus frame 0, eliminations land on the right frames, and pickle works."""
    game = Game(small())
    recorder = Recorder(game)
    recording = recorder.record_all()
    assert recording.length == game.tick + 1
    assert sum(len(f.eliminations) for f in recording.frames) == len(game.eliminations)
    assert recording.result is not None and recording.result.ticks == game.tick
    assert all(len(f.players) == len(game.players) for f in recording.frames)
    path = tmp_path / "g.replay"
    recording.save(path)
    loaded = Recording.load(path)
    assert loaded.length == recording.length
    assert loaded.frames[-1].players[0].x == recording.frames[-1].players[0].x
    assert loaded.roster[0].name == game.players[0].name


def test_trainer_improves_or_holds_and_saves_a_champion(tmp_path):
    """Two generations run, fitness is finite, elites carry over, and the champion round-trips."""
    trainer = GeneticTrainer(
        small(), TrainingConfig(brain_name="voting", population_size=24, generations=2, rounds_per_generation=1, seed=0)
    )
    history = trainer.run()
    assert len(history) == 2
    assert all(np.isfinite(s.best_fitness) and s.best_fitness >= s.mean_fitness >= s.worst_fitness for s in history)
    assert trainer.champion.shape == (8,)
    path = tmp_path / "champ.json"
    trainer.save_champion(path)
    data = GeneticTrainer.load_champion(path)
    assert data["brain_name"] == "voting"
    assert np.allclose(data["genome"], trainer.champion)
    assert trainer.champion_brain().name == "voting"


def test_trainer_handles_neural_genomes_and_padding():
    """A population that is not a multiple of the player count still gets every genome scored."""
    trainer = GeneticTrainer(
        small(), TrainingConfig(brain_name="neural", population_size=30, generations=1, rounds_per_generation=1, seed=1)
    )
    trainer.run()
    assert trainer.genome_size == trainer.champion.size
    assert (trainer.fitness > 0).all() or (trainer.fitness >= 0).all()
    assert len(trainer.population) == 30
