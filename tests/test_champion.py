"""Tests for stage-aware champion selection: the highest curriculum stage wins, then validation wins, then score."""

import numpy as np

from hunger_games.config import SimulationConfig
from hunger_games.training import GenerationStats, GeneticTrainer, TrainingConfig
from hunger_games.training.common import champion_key


def test_champion_key_orders_by_stage_then_wins_then_score():
    """A win at a harder stage beats any score at an easier one; at equal stage, wins beat score."""
    easy_high_score = champion_key(0, 1.0, 5.0)
    hard_no_wins = champion_key(3, 0.0, -2.0)
    hard_some_wins = champion_key(3, 0.5, -3.0)
    hard_same_wins_better_score = champion_key(3, 0.5, 1.0)
    assert hard_no_wins > easy_high_score
    assert hard_some_wins > hard_no_wins
    assert hard_same_wins_better_score > hard_some_wins


def test_genetic_champion_prefers_the_highest_stage():
    """The GA's champion is the generation that ranks first by the key, not the highest training fitness."""
    trainer = GeneticTrainer(
        SimulationConfig(seed=0, width=40, height=40, max_days=2), TrainingConfig(population_size=4)
    )

    def generation(index: int, best_fitness: float, val_fitness: float, val_win_rate: float, stage: int):
        # A fake generation whose champion genome is filled with its index, so the choice can be read back.
        genome = np.full(trainer.population[0].shape, float(index))
        stats = GenerationStats(index, best_fitness, best_fitness, best_fitness, genome, 0.0, val_fitness)
        stats.val_win_rate = val_win_rate
        stats.stage = stage
        return stats

    trainer.history = [
        generation(0, best_fitness=9.0, val_fitness=4.0, val_win_rate=1.0, stage=0),  # easy rung, huge fitness
        generation(1, best_fitness=1.0, val_fitness=-1.0, val_win_rate=0.0, stage=2),  # harder rung, no wins
        generation(2, best_fitness=0.5, val_fitness=-2.0, val_win_rate=0.5, stage=2),  # harder rung, some wins
    ]
    assert float(trainer.champion[0]) == 2.0
