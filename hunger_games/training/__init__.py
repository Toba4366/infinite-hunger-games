"""training - ways to make brains better.

`GeneticTrainer` evolves a population of genomes (any brain that exposes
`genome()` / `set_genome()`) by playing them against each other and
breeding the winners. `ReinforceTrainer` trains the neural brain by policy
gradient with a value baseline, rewarding every action. `save_run` writes
either trainer's results to a timestamped folder with one PNG per chart.
The dashboard's Train tab drives them; the scripts in `experiments/` do too.
"""

# The genetic trainer and its settings.
from hunger_games.training.genetic import GenerationStats, GeneticTrainer, TrainingConfig

# The policy-gradient trainer and its settings.
from hunger_games.training.reinforce import EpochStats, ReinforceTrainer, RLConfig

# Run folders.
from hunger_games.training.runs import save_run

# What `from hunger_games.training import *` exposes.
__all__ = [
    "GeneticTrainer",
    "TrainingConfig",
    "GenerationStats",
    "ReinforceTrainer",
    "RLConfig",
    "EpochStats",
    "save_run",
]
