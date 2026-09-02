"""Tests for the training feed: showcase recordings from both trainers and the session that plays them."""

import time

from hunger_games.config import SimulationConfig
from hunger_games.training import GeneticTrainer, ReinforceTrainer, RLConfig, TrainingConfig
from hunger_games.ui.session import Session


def small() -> SimulationConfig:
    return SimulationConfig(seed=3, width=40, height=40, max_days=3)


def test_genetic_trainer_records_one_showcase_game_per_generation():
    """With record_showcase on, every generation carries a full recording; off, none."""
    trainer = GeneticTrainer(
        small(),
        TrainingConfig(
            brain_name="voting", population_size=24, generations=2, rounds_per_generation=1, validation_games=0, seed=0
        ),
    )
    trainer.run()
    assert all(stats.showcase is not None and stats.showcase.length > 1 for stats in trainer.history)
    assert "showcase" not in trainer.history_rows()[0]
    quiet = GeneticTrainer(
        small(),
        TrainingConfig(
            brain_name="voting",
            population_size=24,
            generations=1,
            rounds_per_generation=1,
            validation_games=0,
            seed=0,
            record_showcase=False,
        ),
    )
    quiet.run()
    assert quiet.history[0].showcase is None


def test_reinforce_trainer_records_one_showcase_game_per_epoch():
    """The RL trainer records its first training game each epoch."""
    trainer = ReinforceTrainer(
        small(), RLConfig(epochs=1, episodes_per_epoch=2, learners_per_game=4, validation_games=0, seed=0)
    )
    trainer.run()
    assert trainer.history[0].showcase is not None
    assert trainer.history[0].showcase.result is not None


def test_session_feed_replays_and_then_plays_the_champion_live():
    """The feed loads the newest recording when the arena is free, and 'live' starts a champion game."""
    session = Session(small())
    session.feed_mode = "replay"
    session.start_training(
        TrainingConfig(
            brain_name="voting", population_size=24, generations=2, rounds_per_generation=1, validation_games=0, seed=0
        )
    )
    while session.training_running:
        time.sleep(0.05)
    session.update(0.01)
    assert session.game is None and session.recording is not None
    assert "replaying" in session.feed_label
    assert session.playing
    session.playhead = session.recording.length - 1
    session.feed_mode = "live"
    session._feed_steps_seen = 0
    session.update(0.01)
    assert session.game is not None and "live" in session.feed_label
    evolution = session.network_evolution()
    assert evolution is not None and len(evolution["steps"]) == 2 and evolution["genes"].shape[0] == 2
    assert evolution["change"][0] == 0.0
