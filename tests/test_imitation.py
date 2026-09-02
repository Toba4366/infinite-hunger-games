"""Tests for imitation pretraining and warm starts."""

import numpy as np

from hunger_games.actions import Action, ActionType
from hunger_games.brain.neural import ATTACK_INDEX, FIRST_MOVE_INDEX, FLEE_INDEX, NeuralBrain
from hunger_games.config import SimulationConfig
from hunger_games.training import (
    GeneticTrainer,
    ImitationConfig,
    ImitationTrainer,
    ReinforceTrainer,
    RLConfig,
    TrainingConfig,
    save_run,
)


def small() -> SimulationConfig:
    return SimulationConfig(seed=5, width=50, height=50, max_days=4)


def test_action_to_menu_index_is_the_inverse_of_menu_to_action():
    """Every menu index maps to an action and back."""
    assert NeuralBrain.action_to_menu_index(Action(ActionType.DRINK)) == 1
    assert NeuralBrain.action_to_menu_index(Action.attack(3)) == ATTACK_INDEX
    assert NeuralBrain.action_to_menu_index(Action.flee(1, 0)) == FLEE_INDEX
    assert NeuralBrain.action_to_menu_index(Action.move(1, 0)) == FIRST_MOVE_INDEX + 4
    assert NeuralBrain.action_to_menu_index(Action.move(0, 0)) == 0


def test_imitation_learns_the_teacher_and_saves(tmp_path):
    """Accuracy rises above chance, validation is tracked, and the run folder gets its charts."""
    trainer = ImitationTrainer(
        small(), ImitationConfig(demonstration_games=2, epochs=6, validation_games=1, learners_per_game=4, seed=0)
    )
    history = trainer.run()
    assert len(history) == 6
    assert history[-1].train_accuracy > history[0].train_accuracy or history[-1].train_loss < history[0].train_loss
    assert history[-1].val_accuracy > 1.0 / 16.0
    assert history[-1].showcase is not None
    assert trainer.champion.size == trainer.policy.parameter_count
    folder = save_run(trainer, "imitation", "test_im", tmp_path)
    assert (folder / "plots" / "accuracy.png").exists() and (folder / "champion.json").exists()


def test_warm_starts_begin_from_the_given_genome():
    """The GA population and the RL policy start from the supplied genome."""
    config = small()
    student = ImitationTrainer(config, ImitationConfig(demonstration_games=1, epochs=1, validation_games=0, seed=0))
    student.run()
    genome = student.champion
    ga = GeneticTrainer(
        config,
        TrainingConfig(
            brain_name="neural", population_size=8, generations=1, rounds_per_generation=1, validation_games=0, seed=0
        ),
        initial_genome=genome,
    )
    assert np.array_equal(ga.population[0], genome)
    assert not np.array_equal(ga.population[1], genome)
    rl = ReinforceTrainer(
        config, RLConfig(epochs=1, episodes_per_epoch=1, validation_games=0, seed=0), initial_genome=genome
    )
    assert np.array_equal(rl.policy.genome(), genome)
