"""config.py - every adjustable knob of the simulation lives in this one place.

A beginner tip: when you want to change how the games behave, look here first.
Nothing in this file *does* anything; it only describes settings that the
other modules read. The dashboard (hunger_games/ui) edits these same values.
"""

# `dataclass` writes the boring __init__ method for us from a list of fields.
from dataclasses import dataclass, field

# `Enum` lets us give a fixed set of named choices (like a multiple-choice question).
from enum import Enum


class ArenaShape(Enum):
    """The overall outline of the arena."""

    # The 74th games (the first film) took place in an open forest / field.
    OPEN_FIELD = "open_field"
    # The 75th games (Catching Fire) took place in a circular "clock" arena.
    ROUND = "round"


class LayoutName(Enum):
    """How the game makers scatter supplies before the games begin."""

    # The classic design: one giant pile of loot in the centre of the arena.
    CORNUCOPIA = "cornucopia"
    # The video's redesign: cheap supplies at the edge, weapons in the centre.
    RING = "ring"


@dataclass
class NoiseConfig:
    """Settings for the Perlin noise that shapes the terrain (see noise.py)."""

    # How many grid cells one "hill" spans. Bigger numbers = smoother, wider hills.
    scale: float = 40.0
    # How many layers of detail get stacked on top of each other.
    octaves: int = 5
    # How much quieter each extra layer of detail is (0.5 = half as loud).
    persistence: float = 0.5
    # How much finer each extra layer of detail is (2.0 = twice as fine).
    lacunarity: float = 2.0


@dataclass
class TerrainConfig:
    """Where each terrain type begins, exactly as described in chapter 4.

    Instead of fixed cut-offs, each threshold is defined *relative* to the
    previous one, so shrinking `sand_size` automatically gives that height
    range to the next terrain type up.
    """

    # Heights below this number are water.
    water_threshold: float = 0.25
    # Sand occupies the next `sand_size` worth of height above the water.
    sand_size: float = 0.10
    # Grass occupies the next `grass_size` worth of height above the sand.
    grass_size: float = 0.50
    # Whatever is left over at the top becomes rock (so no setting is needed).


@dataclass
class NeuralConfig:
    """The shape and starting weights of the neural-network brain (see brain/neural.py)."""

    # Width of each hidden layer, in order. (16,) is one layer of 16 neurons; (32, 16) is two layers.
    hidden_layers: tuple[int, ...] = (16,)
    # The squashing function between layers: tanh, relu, leaky_relu, sigmoid or selu.
    activation: str = "tanh"
    # How the starting weights are drawn (see brain/initializers.py for the full list).
    initializer: str = "xavier_uniform"
    # The constant, or the spread, used by the constant / uniform / normal initializers.
    init_scale: float = 0.05
    # For the sparse initializer: the fraction of weights that are non-zero.
    sparsity: float = 0.1


@dataclass
class RewardConfig:
    """How a learning tribute scores points, tick by tick (see training/reinforce.py).

    Reinforcement learning needs a number that says "that was good" or "that
    was bad" after every action. These weights define it. The genetic
    algorithm ignores this and scores whole games by placement instead.
    """

    # Reward for every tick survived (a small constant pull toward staying alive).
    survive_tick: float = 0.01
    # Bonus for winning the games.
    win: float = 5.0
    # Penalty for dying.
    death: float = -3.0
    # Bonus per elimination.
    kill: float = 1.0
    # Penalty per point of health lost (a full bar lost costs this much).
    damage_taken: float = -2.0
    # Bonus per point of thirst or hunger restored while the bar was below half.
    need_gain: float = 0.5
    # End-of-game bonus scaled by placing: this much for first, nothing for last.
    placement: float = 2.0
    # Discount factor: how much a reward one tick later is worth compared to now.
    discount: float = 0.98


@dataclass
class SimulationConfig:
    """The master settings object handed to `Game`, `Runner`, `Renderer` and the dashboard."""

    # ---- the arena ----
    # Width of the arena grid in cells (120 fits 24 players and renders nicely).
    width: int = 120
    # Height of the arena grid in cells.
    height: int = 120
    # Outline of the arena: OPEN_FIELD (74th games) or ROUND (75th games).
    shape: ArenaShape = ArenaShape.OPEN_FIELD
    # Supply layout: CORNUCOPIA (original) or RING (the video's redesign).
    layout: LayoutName = LayoutName.RING
    # Whether starting podiums may stand in water (the 75th games had podiums in the sea).
    allow_water_podiums: bool = True
    # ---- the tributes ----
    # Number of tributes entering the arena (24 = two per district, one female and one male).
    num_players: int = 24
    # Which brain each player gets by default ("voting", "random" or "neural").
    brain_name: str = "voting"
    # Every tribute starts with a thirst bar drawn between this and 1.0 (1.0 = everyone starts full).
    start_thirst_min: float = 1.0
    # Same for the hunger bar.
    start_hunger_min: float = 1.0
    # Same for the health bar.
    start_health_min: float = 1.0
    # Districts whose tributes train for years and attract sponsors (the "careers").
    career_districts: tuple[int, ...] = (1, 2, 4)
    # ---- randomness ----
    # Master randomness dial from 0.0 (fully deterministic) to 1.0 (very chaotic).
    chaos: float = 0.5
    # Random seed; the same seed + settings reproduces the same game exactly.
    seed: int | None = None
    # ---- time ----
    # How many simulation steps make up one in-game day (24 = one per hour).
    ticks_per_day: int = 24
    # The games end in a draw if they last longer than this many days (a strict cutoff; the 74th ran about 18).
    max_days: int = 24
    # ---- senses and needs ----
    # How many cells away a player can see other players and supplies.
    vision_radius: int = 8
    # How many cells away a player can spot a lake or a meadow (big things show from afar).
    landmark_radius: int = 30
    # A player with a full thirst bar dies of dehydration after this many days.
    thirst_days: float = 3.0
    # A player with a full hunger bar dies of starvation after this many days.
    hunger_days: float = 7.0
    # ---- outside help and interference ----
    # Whether sponsors may parachute gifts to favoured tributes in need.
    sponsors_enabled: bool = True
    # The daily chance that a fully favoured tribute in need receives a gift.
    sponsor_gift_chance: float = 0.5
    # Whether the game makers may slowly shrink the arena when the games go quiet.
    # On by default only because, measured over 20 seeded games, a strict day cutoff alone
    # ends no game with a victor; the circle is slow (see `intervention_days`) and rarely kills.
    gamemaker_enabled: bool = True
    # How many days without an elimination before the game makers step in.
    quiet_days_before_intervention: float = 1.0
    # How many days of shrinking it takes the safe circle to close from the edge to the centre.
    intervention_days: float = 6.0
    # ---- what tributes know ----
    # The cannon and the nightly sky: tributes know how many remain and who (and how strong) they are.
    cannon_and_sky: bool = True
    # The endgame instinct: bold tributes head for the centre once fewer than half remain (off: rely on the cutoff or the circle).
    endgame_instinct: bool = False
    # ---- nested settings ----
    # The Perlin noise settings (a nested dataclass, so it needs a default factory).
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    # The terrain threshold settings (also a nested dataclass).
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    # The neural-network brain settings.
    neural: NeuralConfig = field(default_factory=NeuralConfig)
    # The reward function for reinforcement learning.
    reward: RewardConfig = field(default_factory=RewardConfig)

    @property
    def ticks_per_game(self) -> int:
        """The maximum number of steps a single game can run."""
        # Multiply days by steps-per-day to get the total step budget.
        return self.max_days * self.ticks_per_day

    @property
    def thirst_per_tick(self) -> float:
        """How much of the 0-to-1 thirst bar drains away every step."""
        # Spread a full bar evenly across every step of `thirst_days` days.
        return 1.0 / (self.thirst_days * self.ticks_per_day)

    @property
    def hunger_per_tick(self) -> float:
        """How much of the 0-to-1 hunger bar drains away every step."""
        # Spread a full bar evenly across every step of `hunger_days` days.
        return 1.0 / (self.hunger_days * self.ticks_per_day)

    def to_dict_raw(self) -> dict:
        """A shallow dictionary of the fields, keeping enums and nested objects as they are (for copying with changes)."""
        # Import here to avoid a circular import at module load time.
        from dataclasses import MISSING, fields

        # One entry per field, values untouched. A config pickled by an older version of this
        # file may lack a newly added field; fall back to that field's default rather than crash.
        values = {}
        for f in fields(self):
            if hasattr(self, f.name):
                values[f.name] = getattr(self, f.name)
            elif f.default is not MISSING:
                values[f.name] = f.default
            elif f.default_factory is not MISSING:  # type: ignore[misc]
                values[f.name] = f.default_factory()  # type: ignore[misc]
        return values

    def to_dict(self) -> dict:
        """Flatten the config (including nested parts) into JSON-friendly values."""
        # Import here to avoid a circular import at module load time.
        from dataclasses import asdict

        # asdict handles the nesting; enums are then turned into their string values.
        data = asdict(self)
        # Enums are not JSON-friendly, so store their names.
        data["shape"] = self.shape.value
        # Same for the layout.
        data["layout"] = self.layout.value
        # Hand back the plain dictionary.
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "SimulationConfig":
        """Rebuild a config from the dictionary made by `to_dict`."""
        # Copy so we do not modify the caller's dictionary.
        data = dict(data)
        # Turn the enum strings back into enums.
        data["shape"] = ArenaShape(data.get("shape", cls.shape.value))
        # Same for the layout.
        data["layout"] = LayoutName(data.get("layout", cls.layout.value))
        # Rebuild the nested dataclasses.
        data["noise"] = NoiseConfig(**data.get("noise", {}))
        # Terrain.
        data["terrain"] = TerrainConfig(**data.get("terrain", {}))
        # Neural, converting the JSON list of layer widths back to a tuple.
        neural = dict(data.get("neural", {}))
        # JSON has no tuples, so hidden_layers comes back as a list.
        neural["hidden_layers"] = tuple(neural.get("hidden_layers", NeuralConfig.hidden_layers))
        # Build the neural config.
        data["neural"] = NeuralConfig(**neural)
        # The reward config.
        data["reward"] = RewardConfig(**data.get("reward", {}))
        # Career districts also come back as a list.
        data["career_districts"] = tuple(data.get("career_districts", cls.career_districts))
        # Build the config.
        return cls(**data)
