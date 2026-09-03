"""training - ways to make brains better.

`ImitationTrainer` pretrains the neural brain to copy the voting brain, so
it starts with working instincts. `GeneticTrainer` evolves a population of
genomes (any brain that exposes `genome()` / `set_genome()`) by playing them
against each other and breeding the winners. `ReinforceTrainer` trains the
neural brain by policy gradient with a value baseline, rewarding every
action. Both can warm-start from an imitation champion. `save_run` writes
either trainer's results to a timestamped folder with one PNG per chart.
The dashboard's Train tab drives them; the scripts in `experiments/` do too.
"""

# The genetic trainer and its settings.
# Shared pieces.
from hunger_games.training.common import (
    Curriculum,
    CurriculumConfig,
    EventLog,
    IterationStats,
    LearnerSpec,
    SystemMonitor,
)
from hunger_games.training.genetic import GenerationStats, GeneticTrainer, TrainingConfig

# The imitation (behaviour cloning) trainer and its settings.
from hunger_games.training.imitation import ImitationConfig, ImitationStats, ImitationTrainer

# NEAT.
from hunger_games.training.neat import NeatTrainer, NeatTrainerConfig

# PPO.
from hunger_games.training.ppo import PPOConfig, PPOTrainer

# The policy-gradient trainer and its settings.
from hunger_games.training.reinforce import EpochStats, ReinforceTrainer, RLConfig

# Run folders.
from hunger_games.training.runs import save_run

# What `from hunger_games.training import *` exposes.
__all__ = [
    "Curriculum",
    "CurriculumConfig",
    "EventLog",
    "IterationStats",
    "LearnerSpec",
    "SystemMonitor",
    "NeatTrainer",
    "NeatTrainerConfig",
    "PPOTrainer",
    "PPOConfig",
    "ImitationTrainer",
    "ImitationConfig",
    "ImitationStats",
    "GeneticTrainer",
    "TrainingConfig",
    "GenerationStats",
    "ReinforceTrainer",
    "RLConfig",
    "EpochStats",
    "save_run",
]
