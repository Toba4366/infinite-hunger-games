"""records.py - the rows that get written to the spreadsheet.

Chapter 3 catalogs every elimination by day, method and weapon, and every
player by kills, placement and training score. These dataclasses are those
rows. `Game` fills them in; `Runner` turns them into CSV files.
"""

# Dataclasses and a helper that turns one into a dictionary.
from dataclasses import asdict, dataclass, field

# Enum for the three elimination categories.
from enum import Enum


class EliminationMethod(Enum):
    """Chapter 3's three broad categories."""

    # One tribute killed another.
    PLAYER = "player_vs_player"
    # The game makers' hazard did it.
    GAMEMAKER = "gamemaker"
    # Thirst or starvation.
    NATURAL = "natural_causes"


@dataclass
class Elimination:
    """One row in the eliminations spreadsheet."""

    # Which game this happened in.
    game_id: int
    # The in-game day (1 = the first day).
    day: int
    # The exact tick.
    tick: int
    # Who died.
    victim_id: int
    # Their name.
    victim_name: str
    # Their district.
    victim_district: int
    # Their training score.
    victim_training_score: int
    # One of the three categories (stored as text for the CSV).
    method: str
    # The weapon or cause ("knife", "dehydration", "arena hazard", ...).
    weapon: str
    # Who killed them, if a player did.
    killer_id: int | None
    # The killer's name, if any.
    killer_name: str | None
    # Where it happened.
    x: int
    # Row coordinate.
    y: int
    # The victim's final placing (24 = first out, 2 = runner-up).
    placement: int

    def to_row(self) -> dict:
        """Flatten into a plain dictionary for pandas."""
        # `asdict` walks the fields for us.
        return asdict(self)


@dataclass
class PlayerResult:
    """One row in the players spreadsheet: how one tribute's game went."""

    # Which game.
    game_id: int
    # Who.
    player_id: int
    # Their name.
    name: str
    # Their district.
    district: int
    # "F" or "M".
    sex: str
    # Their training score.
    training_score: int
    # Their survival aptitude.
    survival_score: float
    # Which brain drove them.
    brain: str
    # The sponsors' final opinion of them, 0.0 to 1.0.
    favor: float
    # How many parachutes they received.
    gifts_received: int
    # Final placing (1 = victor).
    placement: int
    # How many players they eliminated.
    kills: int
    # How many days they lasted.
    days_survived: float
    # How they died, or None if they won or survived a draw.
    cause_of_death: str | None
    # Whether they were still alive when the game stopped.
    alive_at_end: bool

    def to_row(self) -> dict:
        """Flatten into a plain dictionary for pandas."""
        # `asdict` walks the fields for us.
        return asdict(self)


@dataclass
class GameResult:
    """Everything worth keeping from one finished game."""

    # Which game.
    game_id: int
    # The seed that reproduces it.
    seed: int
    # How many days it lasted.
    days: int
    # How many ticks it lasted.
    ticks: int
    # The victor's id, or None for a draw.
    winner_id: int | None
    # The victor's name, or None.
    winner_name: str | None
    # How many times the game makers intervened.
    interventions: int
    # Every elimination, in order.
    eliminations: list[Elimination] = field(default_factory=list)
    # One result per player.
    players: list[PlayerResult] = field(default_factory=list)
    # Every sponsor gift, as dictionaries (kept plain so results pickle and CSV easily).
    gifts: list[dict] = field(default_factory=list)
    # Behaviour telemetry summary, when the runner asked for it.
    telemetry: dict | None = None

    def elimination_rows(self) -> list[dict]:
        """All eliminations as dictionaries."""
        # One dictionary per elimination.
        return [elimination.to_row() for elimination in self.eliminations]

    def player_rows(self) -> list[dict]:
        """All player results as dictionaries."""
        # One dictionary per player.
        return [player.to_row() for player in self.players]
