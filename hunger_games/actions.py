"""actions.py - the vocabulary of things a player's body can do.

The brain *chooses* an `Action`; the body (player.py) and the referee
(game.py) *carry it out*. Keeping the list here means a future neural-network
brain and the voting brain speak exactly the same language.
"""

# `frozen=True` makes actions immutable, which lets them be dictionary keys.
from dataclasses import dataclass

# Enum for the fixed list of action kinds.
from enum import Enum


class ActionType(Enum):
    """Every kind of action a player can take in one tick."""

    # Do nothing except recover a sliver of health.
    REST = "rest"
    # Step one cell in a direction.
    MOVE = "move"
    # Drink from the water you are standing in.
    DRINK = "drink"
    # Eat one ration from your pack.
    EAT = "eat"
    # Try to catch food from the terrain you are standing on.
    HUNT = "hunt"
    # Take whatever supplies are in your cell.
    PICK_UP = "pick_up"
    # Use one medkit from your pack.
    HEAL = "heal"
    # Fight an adjacent player.
    ATTACK = "attack"
    # Move away from a threat (a MOVE with a different motive, useful for learning brains).
    FLEE = "flee"


@dataclass(frozen=True)
class Action:
    """One concrete decision: the kind of action plus any details it needs."""

    # Which kind of action this is.
    kind: ActionType
    # Horizontal step for MOVE / FLEE (-1, 0 or +1).
    dx: int = 0
    # Vertical step for MOVE / FLEE (-1, 0 or +1).
    dy: int = 0
    # The player id to fight for ATTACK.
    target_id: int | None = None

    @staticmethod
    def move(dx: int, dy: int) -> "Action":
        """Shortcut for a MOVE action."""
        # Build a MOVE with the given step.
        return Action(ActionType.MOVE, dx=dx, dy=dy)

    @staticmethod
    def flee(dx: int, dy: int) -> "Action":
        """Shortcut for a FLEE action."""
        # Build a FLEE with the given step.
        return Action(ActionType.FLEE, dx=dx, dy=dy)

    @staticmethod
    def attack(target_id: int) -> "Action":
        """Shortcut for an ATTACK action."""
        # Build an ATTACK aimed at the given player.
        return Action(ActionType.ATTACK, target_id=target_id)


# The eight compass directions, in a fixed order so learning brains can index them.
DIRECTIONS = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
# Simple actions that need no extra details.
SIMPLE_ACTIONS = [
    Action(ActionType.REST),
    Action(ActionType.DRINK),
    Action(ActionType.EAT),
    Action(ActionType.HUNT),
    Action(ActionType.PICK_UP),
    Action(ActionType.HEAL),
]
