"""resources.py - the supplies the game makers scatter around the arena.

Chapter 4 tracks resources as three grids (type, quantity, quality) instead
of individual items. `ResourceGrid` is those three grids. The two
`ResourceLayout` subclasses are the Cornucopia design and the ring redesign.
"""

# Abstract base classes let us declare "every layout must have these methods".
# Trigonometry for placing spawn podiums in a circle.
import math
from abc import ABC, abstractmethod

# Integer enums store cleanly inside numpy grids.
from enum import IntEnum

# Type hints only; avoids a circular import at runtime.
from typing import TYPE_CHECKING

# numpy for the three resource grids.
import numpy as np

# The layout names from the config so we can build a layout from a setting.
from hunger_games.config import LayoutName

# This import only happens for type checkers, never when the program runs.
if TYPE_CHECKING:
    from hunger_games.arena import Arena


class ResourceKind(IntEnum):
    """What sort of supply sits in a cell."""

    # Nothing here.
    NONE = 0
    # Rations: eating one restores some of the hunger bar.
    FOOD = 1
    # A weapon: its quality (0.0 to 1.0) decides how deadly it is.
    WEAPON = 2
    # A medkit: using one restores some of the health bar.
    MEDICINE = 3


# Weapon names by minimum quality, from worst to best.
WEAPON_TIERS = [
    (0.0, "fists"),
    (0.2, "rock"),
    (0.4, "knife"),
    (0.6, "spear"),
    (0.8, "sword"),
    (0.9, "bow"),
]


def weapon_name(quality: float) -> str:
    """Translate a 0-to-1 weapon quality into a human-readable name."""
    # Start with the worst weapon of all.
    name = "fists"
    # Walk up the tiers and keep the last one the quality is good enough for.
    for minimum_quality, label in WEAPON_TIERS:
        # If the quality reaches this tier, it earns this name.
        if quality >= minimum_quality:
            # Remember the best name so far.
            name = label
    # Return the best name reached.
    return name


def weapon_reach(quality: float) -> int:
    """How many cells away a weapon can strike: fists/knives 1, spears/swords 2, bows 3."""
    # Bows (quality 0.9 and up) shoot three cells.
    if quality >= 0.9:
        return 3
    # Spears and swords (0.6 and up) reach two cells.
    if quality >= 0.6:
        return 2
    # Everything else needs to be adjacent.
    return 1


class ResourceGrid:
    """Three parallel grids describing the supplies at every cell."""

    def __init__(self, width: int, height: int) -> None:
        """Create empty grids of the given size."""
        # What kind of supply is in each cell (0 = nothing).
        self.kind = np.zeros((height, width), dtype=np.int8)
        # How many of that supply are in each cell.
        self.quantity = np.zeros((height, width), dtype=np.int16)
        # How good that supply is, from 0.0 to 1.0.
        self.quality = np.zeros((height, width), dtype=float)

    def place(self, x: int, y: int, kind: ResourceKind, quantity: int, quality: float) -> None:
        """Put a stack of supplies in a cell (replacing whatever was there)."""
        # Record the type.
        self.kind[y, x] = int(kind)
        # Record how many.
        self.quantity[y, x] = quantity
        # Record how good, clamped to the legal range.
        self.quality[y, x] = float(np.clip(quality, 0.0, 1.0))

    def has(self, x: int, y: int) -> bool:
        """Is there anything in this cell?"""
        # Any non-zero kind means the cell holds something.
        return self.kind[y, x] != int(ResourceKind.NONE)

    def peek(self, x: int, y: int) -> tuple[ResourceKind, int, float]:
        """Look at a cell's supplies without taking them."""
        # Bundle the three values into a tuple.
        return ResourceKind(int(self.kind[y, x])), int(self.quantity[y, x]), float(self.quality[y, x])

    def take(self, x: int, y: int) -> tuple[ResourceKind, int, float]:
        """Remove and return everything in a cell."""
        # Read the contents first.
        contents = self.peek(x, y)
        # Then wipe the cell clean.
        self.kind[y, x] = int(ResourceKind.NONE)
        # Zero the quantity too.
        self.quantity[y, x] = 0
        # And the quality.
        self.quality[y, x] = 0.0
        # Hand back what was there.
        return contents

    def cells_of_kind(self, kind: ResourceKind) -> tuple[np.ndarray, np.ndarray]:
        """Return the (xs, ys) of every cell holding the given kind (used by the renderer)."""
        # Find the row and column indices where the kind matches.
        ys, xs = np.nonzero(self.kind == int(kind))
        # Return them in (x, y) order for plotting.
        return xs, ys


class ResourceLayout(ABC):
    """The contract every supply layout must follow."""

    @abstractmethod
    def apply(self, arena: "Arena", rng: np.random.Generator) -> None:
        """Fill `arena.resources` with supplies."""

    @abstractmethod
    def spawn_positions(self, arena: "Arena", count: int) -> list[tuple[int, int]]:
        """Return the starting podium for each of `count` players."""


class CornucopiaLayout(ResourceLayout):
    """The original 74th-games design: everything valuable in one central pile."""

    # How many cells out from the centre the golden horn's loot spreads.
    PILE_RADIUS = 4
    # How far from the centre the starting podiums stand.
    PODIUM_RADIUS = 10

    def apply(self, arena: "Arena", rng: np.random.Generator) -> None:
        """Pile great loot in the middle and almost nothing anywhere else."""
        # Visit every column...
        for x in range(arena.width):
            # ...and every row.
            for y in range(arena.height):
                # Skip water and cells outside the arena.
                if not arena.is_land(x, y):
                    continue
                # How far this cell is from the centre, in cells.
                distance = arena.distance_from_center(x, y)
                # Inside the pile: every cell gets something good.
                if distance <= self.PILE_RADIUS:
                    # Roll a number from 0 to 1 to pick the kind of item.
                    roll = rng.random()
                    # Half of the pile is weapons.
                    if roll < 0.5:
                        # A single high-quality weapon.
                        arena.resources.place(x, y, ResourceKind.WEAPON, 1, rng.uniform(0.6, 1.0))
                    # Most of the rest is food; medicine is rare (sponsors are the main source).
                    elif roll < 0.95:
                        # A generous stack of good rations.
                        arena.resources.place(x, y, ResourceKind.FOOD, int(rng.integers(4, 9)), rng.uniform(0.6, 1.0))
                    # The last 5% is medicine.
                    else:
                        # A single medkit.
                        arena.resources.place(x, y, ResourceKind.MEDICINE, 1, rng.uniform(0.6, 1.0))
                # Outside the pile: a 2% chance of a scrap of poor food.
                elif rng.random() < 0.02:
                    # One or two low-quality rations.
                    arena.resources.place(x, y, ResourceKind.FOOD, int(rng.integers(1, 3)), rng.uniform(0.1, 0.3))

    def spawn_positions(self, arena: "Arena", count: int) -> list[tuple[int, int]]:
        """Podiums in a tight circle around the pile, like the film."""
        # A list to collect the positions in.
        positions = []
        # One podium per player.
        for index in range(count):
            # Spread the podiums evenly around the full circle.
            angle = 2.0 * math.pi * index / count
            # Convert the angle to an x offset from the centre.
            x = int(round(arena.center_x + math.cos(angle) * self.PODIUM_RADIUS))
            # Convert the angle to a y offset from the centre.
            y = int(round(arena.center_y + math.sin(angle) * self.PODIUM_RADIUS))
            # Nudge onto the arena if the podium landed outside it (water is allowed if configured).
            positions.append(arena.snap_to_podium(x, y))
        # Hand back all the podiums.
        return positions


class RingLayout(ResourceLayout):
    """The video's redesign from chapter 2 and chapter 4.

    Lots of poor supplies around the edge, better and rarer supplies as you
    approach the centre, and a cache of top weapons right in the middle.
    """

    # Cells closer than this (as a fraction of the arena radius) count as "the centre".
    CENTER_FRACTION = 0.06

    def apply(self, arena: "Arena", rng: np.random.Generator) -> None:
        """Scatter supplies so that quantity falls and quality rises toward the centre."""
        # Visit every column...
        for x in range(arena.width):
            # ...and every row.
            for y in range(arena.height):
                # Skip water and cells outside the arena.
                if not arena.is_land(x, y):
                    continue
                # 0.0 at the exact centre, 1.0 at the edge of the arena.
                distance = arena.normalized_distance_from_center(x, y)
                # The central cache: a dense cluster of the best weapons.
                if distance < self.CENTER_FRACTION:
                    # Roughly half the central cells hold a weapon.
                    if rng.random() < 0.5:
                        # Quality 0.8 to 1.0 means swords and bows.
                        arena.resources.place(x, y, ResourceKind.WEAPON, 1, rng.uniform(0.8, 1.0))
                    # Move on to the next cell.
                    continue
                # Everywhere else: the chance of a cell holding something grows toward the edge.
                density = 0.015 + 0.09 * distance
                # Most cells stay empty.
                if rng.random() >= density:
                    continue
                # Quality rises toward the centre, with a little random wobble.
                quality = float(np.clip(1.0 - distance + rng.normal(0.0, 0.08), 0.05, 1.0))
                # Weapons become much more likely close to the centre.
                weapon_chance = (1.0 - distance) ** 2
                # Roll for a weapon first.
                if rng.random() < weapon_chance:
                    # A single weapon of the local quality.
                    arena.resources.place(x, y, ResourceKind.WEAPON, 1, quality)
                # Otherwise a rare chance of medicine (sponsors are the main source).
                elif rng.random() < 0.03:
                    # A single medkit.
                    arena.resources.place(x, y, ResourceKind.MEDICINE, 1, quality)
                # Otherwise food, with bigger stacks nearer the edge.
                else:
                    # One ration at the centre, up to five at the edge.
                    arena.resources.place(x, y, ResourceKind.FOOD, 1 + int(4 * distance), quality)

    def spawn_positions(self, arena: "Arena", count: int) -> list[tuple[int, int]]:
        """Podiums along the outer edge, so players must travel inward for loot."""
        # The arena knows its own outline, so let it place the podiums.
        return arena.edge_positions(count)


def build_layout(name: LayoutName) -> ResourceLayout:
    """Turn a config setting into a layout object."""
    # Map each enum value to the class that implements it.
    layouts = {
        LayoutName.CORNUCOPIA: CornucopiaLayout,
        LayoutName.RING: RingLayout,
    }
    # Look the class up and construct it.
    return layouts[name]()
