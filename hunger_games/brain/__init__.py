"""brain - the decision-making half of a player.

Import `create_brain` to build a brain by name, or subclass `Brain` to
write your own and add it to `BRAIN_REGISTRY`.
"""

# numpy for the generator type hint.
import numpy as np

# The abstract base class.
from hunger_games.brain.base import Brain

# The three built-in brains.
from hunger_games.brain.neural import NeuralBrain
from hunger_games.brain.random_brain import RandomBrain
from hunger_games.brain.voting import VotingBrain

# The neural architecture settings.
from hunger_games.config import NeuralConfig

# Name -> class, so config files and command lines can pick a brain by string.
BRAIN_REGISTRY: dict[str, type[Brain]] = {
    VotingBrain.name: VotingBrain,
    RandomBrain.name: RandomBrain,
    NeuralBrain.name: NeuralBrain,
}


def create_brain(
    name: str, chaos: float, rng: np.random.Generator, neural: NeuralConfig | None = None, endgame: bool = False
) -> Brain:
    """Build a fresh brain of the given kind."""
    # A NEAT brain without a saved genome starts minimal: inputs wired straight to the outputs.
    if name == "neat":
        from hunger_games.brain.neat import InnovationTracker, NeatBrain, NeatConfig, NeatGenome  # noqa: PLC0415
        from hunger_games.brain.neural import MENU_SIZE  # noqa: PLC0415
        from hunger_games.perception import VECTOR_SIZE  # noqa: PLC0415

        return NeatBrain(
            NeatGenome.minimal(VECTOR_SIZE, MENU_SIZE, rng, NeatConfig(), InnovationTracker()), chaos=chaos
        )
    # Unknown names are a configuration mistake worth reporting clearly.
    if name not in BRAIN_REGISTRY:
        raise KeyError(f"Unknown brain '{name}'. Choose from: {', '.join(BRAIN_REGISTRY)} or neat")
    # The neural brain needs its architecture settings and a generator for its starting weights.
    if name == NeuralBrain.name:
        return NeuralBrain(chaos=chaos, config=neural, rng=rng)
    # The voting brain has an optional endgame instinct.
    if name == VotingBrain.name:
        return VotingBrain(chaos=chaos, endgame=endgame)
    # Every other brain only needs the chaos dial.
    return BRAIN_REGISTRY[name](chaos=chaos)


# What `from hunger_games.brain import *` exposes.
__all__ = ["Brain", "VotingBrain", "RandomBrain", "NeuralBrain", "BRAIN_REGISTRY", "create_brain"]
