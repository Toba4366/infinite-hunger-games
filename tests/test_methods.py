"""Tests for the shared training core, NEAT, PPO and the curriculum."""

import numpy as np

from hunger_games.brain.neat import InnovationTracker, NeatBrain, NeatConfig, NeatGenome
from hunger_games.brain.neural import MENU_SIZE
from hunger_games.config import SimulationConfig
from hunger_games.game import Game
from hunger_games.perception import VECTOR_SIZE
from hunger_games.training import (
    Curriculum,
    CurriculumConfig,
    GeneticTrainer,
    ImitationConfig,
    ImitationTrainer,
    NeatTrainer,
    NeatTrainerConfig,
    PPOConfig,
    PPOTrainer,
    ReinforceTrainer,
    RLConfig,
    SystemMonitor,
    TrainingConfig,
    save_run,
)


def small() -> SimulationConfig:
    return SimulationConfig(seed=9, width=40, height=40, max_days=3)


def test_neat_genome_grows_and_round_trips():
    """A minimal genome evaluates, mutates into a bigger feed-forward one, crosses over, and survives JSON."""
    rng = np.random.default_rng(0)
    tracker = InnovationTracker()
    config = NeatConfig(add_node_rate=1.0, add_connection_rate=1.0)
    genome = NeatGenome.minimal(VECTOR_SIZE, MENU_SIZE, rng, config, tracker)
    assert genome.forward(np.zeros(VECTOR_SIZE)).shape == (MENU_SIZE,)
    assert genome.enabled_count == (VECTOR_SIZE + 1) * MENU_SIZE
    for _ in range(5):
        genome.mutate(rng, config, tracker)
    assert genome.hidden_count >= 1
    depth = genome.depths()
    assert all(depth[c.src] < depth[c.dst] for c in genome.connections if c.enabled)
    other = NeatGenome.minimal(VECTOR_SIZE, MENU_SIZE, rng, config, tracker)
    child = genome.crossover(other, rng)
    assert child.forward(np.ones(VECTOR_SIZE)).shape == (MENU_SIZE,)
    assert genome.distance(other, config) > 0.0
    rebuilt = NeatGenome.from_dict(genome.to_dict())
    assert np.allclose(rebuilt.forward(np.ones(VECTOR_SIZE)), genome.forward(np.ones(VECTOR_SIZE)))
    brain = NeatBrain(genome, chaos=0.0)
    game = Game(small())
    perception = game.players[0].perceive(game.arena, game.players, False, 0.0, 1.0, 8)
    assert brain.decide(perception, rng) is not None
    assert "NEAT" in brain.describe()


def test_curriculum_promotes_on_score_or_timeout():
    """The learner moves up a stage when its recent mean score clears the threshold, or after enough tries."""
    curriculum = Curriculum(CurriculumConfig(opponents=(1, 3, 7), threshold=1.0, window=2, max_iterations_per_stage=4))
    assert curriculum.opponents == 1
    assert not curriculum.observe(0.0)
    assert curriculum.observe(2.0) is True and curriculum.opponents == 3
    assert not any(curriculum.observe(0.0) for _ in range(3))
    assert curriculum.observe(0.0) is True and curriculum.opponents == 7 and curriculum.finished
    assert Curriculum(CurriculumConfig(enabled=False)).opponents == 23


def test_every_method_produces_the_shared_iteration_stats(tmp_path):
    """All five trainers fill learning_history in the same shape and save a run folder with the shared curves."""
    config = small()
    trainers = {
        "imitation": ImitationTrainer(
            config, ImitationConfig(demonstration_games=1, epochs=1, validation_games=1, learners_per_game=4, seed=0)
        ),
        "genetic": GeneticTrainer(
            config,
            TrainingConfig(
                brain_name="neural",
                population_size=6,
                generations=1,
                rounds_per_generation=1,
                validation_games=1,
                seed=0,
            ),
        ),
        "neat": NeatTrainer(
            config, NeatTrainerConfig(population_size=6, generations=1, validation_games=1, learners_per_game=4, seed=0)
        ),
        "reinforce": ReinforceTrainer(
            config, RLConfig(epochs=1, episodes_per_epoch=1, learners_per_game=4, validation_games=1, seed=0)
        ),
        "ppo": PPOTrainer(
            config,
            PPOConfig(epochs=1, episodes_per_epoch=1, learners_per_game=4, validation_games=1, seed=0, update_epochs=2),
        ),
    }
    for method, trainer in trainers.items():
        stats = trainer.step()
        assert stats.iteration == 0 and len(trainer.learning_history) == 1, method
        assert np.isfinite(stats.mean_score) and stats.mean_length >= 0 and 0.0 <= stats.win_rate <= 1.0, method
        assert trainer.events.events, method
        assert trainer.learner_spec().kind in ("neural", "neat"), method
        folder = save_run(trainer, method, f"t_{method}", tmp_path)
        assert (folder / "learning.json").exists() and (folder / "events.txt").exists(), method
        assert (folder / "plots" / "score.png").exists(), method


def test_genetic_curriculum_and_winners_only_demonstrations():
    """The GA sizes its roster from the curriculum, and imitation can keep winners' decisions only."""
    curriculum = Curriculum(CurriculumConfig(opponents=(2, 5), threshold=99.0, window=1, max_iterations_per_stage=1))
    trainer = GeneticTrainer(
        small(),
        TrainingConfig(
            brain_name="neural",
            population_size=4,
            generations=2,
            rounds_per_generation=1,
            validation_games=0,
            learners_per_game=2,
            seed=0,
        ),
        curriculum=curriculum,
    )
    trainer.run()
    assert trainer.learning_history[0].opponents == 2 and trainer.learning_history[1].opponents == 5
    assert trainer.config.num_players == 2 + 5
    winners = ImitationTrainer(
        small(), ImitationConfig(demonstration_games=1, epochs=1, validation_games=0, winners_top=3, seed=0)
    )
    everyone = ImitationTrainer(small(), ImitationConfig(demonstration_games=1, epochs=1, validation_games=0, seed=0))
    assert winners.collect() < everyone.collect()


def test_system_monitor_reads():
    """The monitor returns the keys the dashboard shows."""
    reading = SystemMonitor().read()
    assert set(reading) == {"cpu_percent", "memory_mb", "memory_percent", "gpu"}
