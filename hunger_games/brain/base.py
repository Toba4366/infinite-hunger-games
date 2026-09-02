"""brain/base.py - the contract every brain must follow.

Chapter 4 splits a player into a *body* (what they can do) and a *brain*
(what they decide to do). This file defines the brain half as an abstract
class. Anything that implements `decide()` can drive a player: the voting
brain from the video, a random brain, a neural network, a genetic-algorithm
genome, or a reinforcement-learning policy.

The optional hooks (`genome`, `set_genome`, `observe`, `on_game_end`) exist so
future training code has somewhere to plug in without changing the game.
"""

# ABC / abstractmethod force subclasses to implement `decide`.
# Deep copies so a brain can be cloned for a new player.
import copy
from abc import ABC, abstractmethod

# numpy for genomes (parameter vectors).
import numpy as np

# The action the brain must return.
from hunger_games.actions import Action

# The perception the brain receives.
from hunger_games.perception import Perception


class Brain(ABC):
    """Base class for all decision-makers."""

    # A short label written into the results CSV so brains can be compared.
    name = "base"

    def __init__(self, chaos: float = 0.0) -> None:
        """Every brain gets the chaos dial so it can add decision noise."""
        # 0.0 = always pick the best action, 1.0 = very unpredictable.
        self.chaos = chaos

    @abstractmethod
    def decide(self, perception: Perception, rng: np.random.Generator) -> Action:
        """Look at what the player senses and choose one action for this tick."""

    # ------------------------------------------------------- learning hooks

    def genome(self) -> np.ndarray:
        """The brain's tunable numbers as a flat vector.

        A genetic algorithm mutates and recombines this vector; a brain with
        nothing to tune returns an empty array.
        """
        # By default there is nothing to tune.
        return np.zeros(0, dtype=float)

    def set_genome(self, genome: np.ndarray) -> None:
        """Load a flat vector of tunable numbers (the reverse of `genome`)."""
        # By default there is nothing to load into.
        return None

    def observe(self, perception: Perception, action: Action, reward: float) -> None:
        """Reinforcement-learning hook, called after each action with a reward.

        The base game does not compute rewards; a training script can wrap
        `Game` and call this itself.
        """
        # By default the brain ignores feedback.
        return None

    def on_game_end(self, placement: int, kills: int, days_survived: float) -> None:
        """Called once when the game ends, so a learner can score the whole episode."""
        # By default the brain ignores the outcome.
        return None

    def clone(self) -> "Brain":
        """A fresh, independent copy of this brain (for cloning winners)."""
        # deepcopy also copies any numpy arrays inside.
        return copy.deepcopy(self)
