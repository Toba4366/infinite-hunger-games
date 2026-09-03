"""scenario.py - everything a game maker can customise before the games begin.

A `Scenario` is the saved state of the dashboard: a painted map, hand-placed
loot, and a roster of tributes with names, scores, brains and podiums. It is
plain data, saved as JSON, and `Game` knows how to build a game from it.
Anything left as None falls back to the normal generated behaviour.
"""

# Dataclasses and helpers to turn them into dictionaries.
# JSON for saving.
import json
from dataclasses import asdict, dataclass, field

# Filesystem paths.
from pathlib import Path


@dataclass
class TributeSpec:
    """One tribute as edited in the dashboard."""

    # Unique id, 0 upward.
    player_id: int
    # Display name (rename your favourite here).
    name: str
    # District, 1 to 12.
    district: int
    # "F" or "M".
    sex: str
    # The 1 to 12 training score.
    training_score: int
    # The 0.05 to 0.95 survival aptitude.
    survival_score: float
    # Which brain drives them: "voting", "random" or "neural".
    brain_name: str = "voting"
    # A saved genome for that brain (a list of weights, or a NEAT genome dictionary), or None for a fresh one.
    genome: list[float] | dict | None = None
    # A weapon granted before the games (0.0 = none).
    weapon_quality: float = 0.0
    # Rations granted before the games.
    food: int = 0
    # Medkits granted before the games.
    medicine: int = 0
    # Extra sponsor favour granted by the game maker (0.0 to 1.0).
    favor_bonus: float = 0.0
    # Starting bars; None means "use the config's random range".
    start_thirst: float | None = None
    # Starting hunger.
    start_hunger: float | None = None
    # Starting health.
    start_health: float | None = None
    # Podium (x, y), or None to use the layout's podiums.
    podium: tuple[int, int] | None = None


@dataclass
class LootSpec:
    """One stack of supplies placed by hand."""

    # Column.
    x: int
    # Row.
    y: int
    # 1 = food, 2 = weapon, 3 = medicine (the ResourceKind values).
    kind: int
    # How many.
    quantity: int
    # How good, 0.0 to 1.0.
    quality: float


@dataclass
class Scenario:
    """A complete custom setup. Every field is optional."""

    # A painted terrain grid (rows of TerrainType integers), or None to generate one.
    terrain: list[list[int]] | None = None
    # Whether the chosen layout still scatters its own loot (False = only hand-placed loot).
    use_layout_loot: bool = True
    # Hand-placed loot.
    loot: list[LootSpec] = field(default_factory=list)
    # The roster, or None to generate one.
    tributes: list[TributeSpec] | None = None
    # A free-text label for the dashboard.
    title: str = "Untitled scenario"

    # ------------------------------------------------------------- saving

    def to_dict(self) -> dict:
        """Turn the scenario into JSON-friendly data."""
        # asdict walks every nested dataclass.
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Scenario":
        """Rebuild a scenario from the dictionary made by `to_dict`."""
        # Rebuild the loot list.
        loot = [LootSpec(**item) for item in data.get("loot", [])]
        # Rebuild the roster, if there is one.
        tributes = None
        # Only when saved.
        if data.get("tributes") is not None:
            # Convert each entry, restoring the podium tuple JSON turned into a list.
            tributes = []
            # One at a time.
            for item in data["tributes"]:
                # Copy so we can edit.
                item = dict(item)
                # JSON has no tuples.
                if item.get("podium") is not None:
                    item["podium"] = tuple(item["podium"])
                # Build the spec.
                tributes.append(TributeSpec(**item))
        # Assemble the scenario.
        return cls(
            terrain=data.get("terrain"),
            use_layout_loot=data.get("use_layout_loot", True),
            loot=loot,
            tributes=tributes,
            title=data.get("title", "Untitled scenario"),
        )

    def save(self, path: str | Path) -> None:
        """Write the scenario to a JSON file."""
        # Serialise and write.
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: str | Path) -> "Scenario":
        """Read a scenario from a JSON file."""
        # Read, parse and rebuild.
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ------------------------------------------------------------ helpers

    def tribute(self, player_id: int) -> TributeSpec | None:
        """Find a tribute spec by id."""
        # Linear search is fine for 24 entries.
        for spec in self.tributes or []:
            # Match on id.
            if spec.player_id == player_id:
                return spec
        # Not found.
        return None
