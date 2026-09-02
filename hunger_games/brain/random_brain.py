"""brain/random_brain.py - a coin-flipping brain used as a baseline.

If a smarter brain cannot beat this one, the smarter brain is broken.
"""

# numpy for the random generator type hint.
import numpy as np

# The menu of actions.
from hunger_games.actions import DIRECTIONS, SIMPLE_ACTIONS, Action

# The base class.
from hunger_games.brain.base import Brain

# The perception type.
from hunger_games.perception import Perception


class RandomBrain(Brain):
    """Picks a completely random action every tick."""

    # Label for the results CSV.
    name = "random"

    def decide(self, perception: Perception, rng: np.random.Generator) -> Action:
        """Ignore the perception entirely and roll dice."""
        # All simple actions plus one MOVE per compass direction.
        menu = SIMPLE_ACTIONS + [Action.move(dx, dy) for dx, dy in DIRECTIONS]
        # Pick one at random.
        return menu[int(rng.integers(len(menu)))]
