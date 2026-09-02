"""sponsors.py - gifts parachuted in from the Capitol.

In the films, a wounded or starving tribute with rich sponsors gets a
silver parachute: medicine, food, water. Sponsors back the tributes they
like, which in this simulation means a high training score, a career
district, and a record of kills. This replaces medkits lying around the
arena as the main way to survive a wound.
"""

# A record type for each gift.
from dataclasses import dataclass

# The player type, imported for type hints only.
from typing import TYPE_CHECKING

# numpy for the random generator type.
import numpy as np

# The settings.
from hunger_games.config import SimulationConfig

# Career district check.
from hunger_games.districts import is_career_district

# This import only happens for type checkers, never when the program runs.
if TYPE_CHECKING:
    from hunger_games.player import Player


@dataclass
class SponsorGift:
    """One parachute: who got it, when, and what was inside."""

    # Which game.
    game_id: int
    # The in-game day.
    day: int
    # The exact tick.
    tick: int
    # Who received it.
    player_id: int
    # Their name.
    player_name: str
    # What was inside: "medicine", "food" or "water".
    kind: str
    # The receiver's sponsor favour at the time, 0.0 to 1.0.
    favor: float


class SponsorPool:
    """Decides who gets gifts and delivers them."""

    # Rations in a food parcel.
    FOOD_PARCEL = 3
    # How much of the thirst bar a water parcel restores.
    WATER_PARCEL = 0.6
    # Health below this counts as "in need of medicine".
    NEEDS_MEDICINE = 0.6
    # Hunger below this counts as "in need of food".
    NEEDS_FOOD = 0.35
    # Thirst below this counts as "in need of water".
    NEEDS_WATER = 0.35

    def __init__(self, config: SimulationConfig) -> None:
        """Remember the settings that shape favour and gift chance."""
        # Whether gifts happen at all.
        self.enabled = config.sponsors_enabled
        # The daily chance for a fully favoured tribute in need.
        self.gift_chance = config.sponsor_gift_chance
        # Which districts are careers.
        self.career_districts = config.career_districts

    def favor(self, player: "Player") -> float:
        """How much the sponsors like this tribute, from 0.0 to 1.0."""
        # Half of the favour comes from the training score (12 = 0.5).
        score_part = 0.5 * player.training_score / 12.0
        # A quarter comes from being a career.
        career_part = 0.25 if is_career_district(player.district, self.career_districts) else 0.0
        # The rest is earned by kills (audiences love a killer), capped at a quarter.
        kill_part = min(0.25, 0.08 * player.kills)
        # Add any bonus a game maker granted in the dashboard, and clamp.
        return float(np.clip(score_part + career_part + kill_part + player.favor_bonus, 0.0, 1.0))

    @classmethod
    def need_of(cls, player: "Player") -> str | None:
        """What this tribute most needs right now, or None if they are fine."""
        # Wounds come first because untreated ones bleed.
        if player.health < cls.NEEDS_MEDICINE:
            return "medicine"
        # Then water, which kills fastest.
        if player.thirst < cls.NEEDS_WATER:
            return "water"
        # Then food.
        if player.hunger < cls.NEEDS_FOOD:
            return "food"
        # Nothing urgent.
        return None

    def deliver(self, player: "Player", kind: str) -> None:
        """Put the parcel's contents straight into the tribute's hands."""
        # Medicine goes into the pack (the brain decides when to use it).
        if kind == "medicine":
            player.medicine += 1
        # Food goes into the pack.
        elif kind == "food":
            player.food += self.FOOD_PARCEL
        # Water is drunk on the spot.
        elif kind == "water":
            player.thirst = min(1.0, player.thirst + self.WATER_PARCEL)

    def daily_gifts(
        self, players: list["Player"], rng: np.random.Generator, game_id: int, day: int, tick: int
    ) -> list[SponsorGift]:
        """Once a day: roll for every living tribute in need and deliver what they win."""
        # No sponsors, no gifts.
        if not self.enabled:
            return []
        # Gifts delivered today.
        gifts = []
        # Check every tribute.
        for player in players:
            # The dead get no parachutes.
            if not player.alive:
                continue
            # What they need, if anything.
            kind = self.need_of(player)
            # Comfortable tributes get nothing.
            if kind is None:
                continue
            # Their current favour.
            favor = self.favor(player)
            # Remember it for the dashboard.
            player.favor = favor
            # Roll against chance scaled by favour.
            if rng.random() < self.gift_chance * favor:
                # Deliver the parcel.
                self.deliver(player, kind)
                # Record it.
                gifts.append(SponsorGift(game_id, day, tick, player.player_id, player.name, kind, favor))
        # Hand back today's gifts.
        return gifts
