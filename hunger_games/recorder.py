"""recorder.py - record a game tick by tick so it can be replayed slowly later.

A game runs in a fraction of a second. To watch one, scrub back and forth,
click on a tribute mid-game, or export a GIF, we need every tick saved. The
`Recorder` copies the interesting state after each tick into a `Frame`; a
`Recording` is the list of frames plus everything needed to redraw them.
Recordings save to disk with pickle.
"""

# Dataclasses for the snapshots.
# Saving Python objects to disk.
import pickle
from dataclasses import dataclass, field

# Filesystem paths.
from pathlib import Path

# numpy for the grids.
import numpy as np

# The settings and custom setup types.
from hunger_games.config import SimulationConfig

# The game being recorded.
from hunger_games.game import Game

# Rows.
from hunger_games.records import Elimination, GameResult

# Custom setups.
from hunger_games.scenario import Scenario

# Parachutes.
from hunger_games.sponsors import SponsorGift


@dataclass
class PlayerSnapshot:
    """One tribute's state at one tick."""

    # Who.
    player_id: int
    # Column.
    x: int
    # Row.
    y: int
    # Still in the games?
    alive: bool
    # Thirst bar.
    thirst: float
    # Hunger bar.
    hunger: float
    # Health bar.
    health: float
    # Rations carried.
    food: int
    # Medkits carried.
    medicine: int
    # Best weapon quality.
    weapon_quality: float
    # Eliminations so far.
    kills: int
    # Sponsor favour.
    favor: float
    # The last action taken, as text.
    last_action: str


@dataclass
class Frame:
    """Everything needed to draw one tick."""

    # The tick.
    tick: int
    # The in-game day.
    day: int
    # Every tribute's state.
    players: list[PlayerSnapshot]
    # The supply kinds grid at this tick (small: one byte per cell).
    resource_kind: np.ndarray
    # The game makers' safe radius.
    safe_radius: float
    # Whether the circle should be drawn.
    circle_visible: bool
    # Eliminations that happened on this tick.
    eliminations: list[Elimination] = field(default_factory=list)
    # Parachutes that landed on this tick.
    gifts: list[SponsorGift] = field(default_factory=list)


@dataclass
class RosterEntry:
    """The fixed facts about one tribute, stored once per recording."""

    # Who.
    player_id: int
    # Display name.
    name: str
    # District.
    district: int
    # "F" or "M".
    sex: str
    # Training score.
    training_score: int
    # Survival aptitude.
    survival_score: float
    # Which brain.
    brain: str


@dataclass
class Recording:
    """A whole game, tick by tick, plus the world it happened in."""

    # The settings the game used.
    config: SimulationConfig
    # The custom setup, if any.
    scenario: Scenario | None
    # The terrain grid.
    terrain: np.ndarray
    # The height grid (for relief shading).
    heights: np.ndarray
    # The fixed facts about every tribute.
    roster: list[RosterEntry]
    # One frame per tick, starting with tick 0 before anyone moves.
    frames: list[Frame] = field(default_factory=list)
    # The final result, filled in when the game ends.
    result: GameResult | None = None

    @property
    def length(self) -> int:
        """How many frames were recorded."""
        # The frame count.
        return len(self.frames)

    def save(self, path: str | Path) -> None:
        """Write the recording to disk (a pickle file, conventionally `.replay`)."""
        # Serialise the whole object.
        Path(path).write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path: str | Path) -> "Recording":
        """Read a recording back. Only open replay files you made yourself: pickle runs code."""
        # Deserialise.
        return pickle.loads(Path(path).read_bytes())


class Recorder:
    """Watches a `Game` and snapshots it after every tick."""

    def __init__(self, game: Game) -> None:
        """Start a recording with the game's starting state as frame 0."""
        # The game being watched.
        self.game = game
        # How many eliminations and gifts were already captured (to find the new ones each tick).
        self._eliminations_seen = 0
        # Same for gifts.
        self._gifts_seen = 0
        # The recording being built.
        self.recording = Recording(
            config=game.config,
            scenario=game.scenario,
            terrain=game.arena.terrain.copy(),
            heights=game.arena.heights.copy(),
            roster=[
                RosterEntry(p.player_id, p.name, p.district, p.sex, p.training_score, p.survival_score, p.brain.name)
                for p in game.players
            ],
        )
        # Capture the starting positions.
        self.capture()

    def capture(self) -> Frame:
        """Snapshot the game right now and append the frame."""
        # The game.
        game = self.game
        # Eliminations that appeared since the last capture.
        new_eliminations = game.eliminations[self._eliminations_seen :]
        # Remember how many we have seen.
        self._eliminations_seen = len(game.eliminations)
        # Gifts that appeared since the last capture.
        new_gifts = game.gifts[self._gifts_seen :]
        # Remember how many we have seen.
        self._gifts_seen = len(game.gifts)
        # Build the frame.
        frame = Frame(
            tick=game.tick,
            day=game.day_number,
            players=[
                PlayerSnapshot(
                    p.player_id,
                    p.x,
                    p.y,
                    p.alive,
                    p.thirst,
                    p.hunger,
                    p.health,
                    p.food,
                    p.medicine,
                    p.weapon_quality,
                    p.kills,
                    p.favor,
                    p.last_action.kind.value if p.last_action else "",
                )
                for p in game.players
            ],
            resource_kind=game.arena.resources.kind.copy(),
            safe_radius=game.gamemaker.safe_radius,
            circle_visible=game.gamemaker.is_active,
            eliminations=list(new_eliminations),
            gifts=list(new_gifts),
        )
        # Append it.
        self.recording.frames.append(frame)
        # Hand it back.
        return frame

    def step(self) -> Frame:
        """Advance the game one tick and capture it."""
        # Tick.
        self.game.step()
        # Snapshot.
        frame = self.capture()
        # Fill in the result when the game ends.
        if self.game.is_over:
            self.recording.result = self.game.result()
        # Hand back the new frame.
        return frame

    def record_all(self) -> Recording:
        """Play the game to the end, capturing every tick."""
        # Step until done.
        while not self.game.is_over:
            self.step()
        # Make sure the result is filled in even for a game that was over at tick 0.
        if self.recording.result is None:
            self.recording.result = self.game.result()
        # The finished recording.
        return self.recording
