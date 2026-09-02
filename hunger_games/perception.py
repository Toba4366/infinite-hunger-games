"""perception.py - everything a player can sense in one tick.

The body gathers a `Perception`, hands it to the brain, and the brain
returns an `Action`. Because the brain only ever sees this object, you can
swap in a neural network, a genetic-algorithm brain or a hand-written one
without touching the rest of the simulator. `to_vector()` flattens the whole
thing into a fixed-length list of numbers ready for a neural network.
"""

# Dataclasses for plain "bag of values" objects.
from dataclasses import dataclass, field

# numpy for the flattened vector.
import numpy as np

# Supply kinds.
from hunger_games.resources import ResourceKind

# Terrain kinds.
from hunger_games.terrain import TerrainType


@dataclass
class NearbyPlayer:
    """What a player can tell about another player within sight."""

    # The other player's id, needed to target them with ATTACK.
    player_id: int
    # Horizontal offset from me (negative = to my left).
    dx: int
    # Vertical offset from me (negative = above me).
    dy: int
    # Chebyshev distance: how many king-moves away they are.
    distance: float
    # Their estimated deadliness from 0.0 to 1.0 (survival skill + weapon + health).
    threat: float
    # Their health bar, 0.0 to 1.0.
    health: float

    def direction_toward(self) -> tuple[int, int]:
        """A single step toward this player."""
        # Sign of each offset.
        return (self.dx > 0) - (self.dx < 0), (self.dy > 0) - (self.dy < 0)

    def direction_away(self) -> tuple[int, int]:
        """A single step away from this player."""
        # The opposite of the step toward them.
        toward_x, toward_y = self.direction_toward()
        # Flip both components.
        return -toward_x, -toward_y


@dataclass
class Perception:
    """A snapshot of a player's own state plus what they can see around them."""

    # --- my own body ---
    # Thirst bar, 1.0 = fully hydrated, 0.0 = dead.
    thirst: float
    # Hunger bar, 1.0 = full, 0.0 = dead.
    hunger: float
    # Health bar, 1.0 = unhurt, 0.0 = dead.
    health: float
    # Fixed survival aptitude from 0.0 to 1.0 (chapter 4's "survival score").
    survival_score: float
    # My own training score scaled to 0.0..1.0 (tributes know how they were rated).
    training_score: float
    # Quality of the best weapon I carry (0.0 = bare hands).
    weapon_quality: float
    # How many cells away my weapon can strike (1 for fists, up to 3 for a bow).
    reach: int
    # Rations in my pack.
    food_count: int
    # Medkits in my pack.
    medicine_count: int
    # --- where I am standing ---
    # The terrain under my feet.
    terrain_here: TerrainType
    # Convenience flag: am I standing in water?
    in_water: bool
    # How hard hunting is right here.
    hunt_difficulty: float
    # Steepest downhill step from here (the video's original water-seeking rule).
    downhill: tuple[int, int]
    # Step toward the nearest water, or (0, 0) if none is within sight.
    water_direction: tuple[int, int]
    # Steps to the nearest water (may be infinite).
    water_distance: float
    # Step toward the nearest grass, or (0, 0) if none is within sight.
    grass_direction: tuple[int, int]
    # Steps to the nearest grass (may be infinite).
    grass_distance: float
    # Step toward the centre of the arena.
    center_direction: tuple[int, int]
    # Distance from the centre as a fraction (0.0 = middle, 1.0 = edge).
    center_distance: float
    # --- supplies ---
    # What kind of supply is in my cell.
    resource_here_kind: ResourceKind
    # How many of it.
    resource_here_quantity: int
    # How good it is.
    resource_here_quality: float
    # Step toward the nearest visible supply (0, 0 if none or it is here).
    nearby_resource_direction: tuple[int, int]
    # Distance to that supply (infinite if none in sight).
    nearby_resource_distance: float
    # What kind of supply that is.
    nearby_resource_kind: ResourceKind
    # --- other people ---
    # Every other living player within my vision radius, nearest first.
    nearby_players: list[NearbyPlayer] = field(default_factory=list)
    # --- the game makers ---
    # Am I standing somewhere the game makers have made lethal?
    in_danger_zone: bool = False
    # How many cells of safe ground lie between me and the lethal edge (negative = already outside).
    hazard_distance: float = 999.0
    # Is the safe circle shrinking right now? (You can see the fog coming.)
    hazard_closing: bool = False
    # Step toward safety when the danger zone is closing in.
    safe_direction: tuple[int, int] = (0, 0)
    # --- the clock ---
    # How far through the maximum game length we are (0.0 to 1.0).
    day_fraction: float = 0.0
    # What fraction of the tributes are still alive (1.0 at the start; the cannon tells everyone).
    alive_fraction: float = 1.0
    # --- the field (from the nightly sky: tributes trained together and know who is left) ---
    # Whether the fields below are known (False when `cannon_and_sky` is off: they are then zero).
    field_known: bool = False
    # Mean training score of the other living tributes, scaled 0..1.
    field_strength: float = 0.0
    # The strongest other living tribute's training score, scaled 0..1.
    strongest_remaining: float = 0.0
    # The fraction of other living tributes whose training score is below mine (0.5 when unknown).
    my_rank: float = 0.5
    # How far I can see, so distances can be normalised.
    vision_radius: int = 8

    @property
    def nearest_threat(self) -> NearbyPlayer | None:
        """The closest other player, if anyone is in sight."""
        # The list is sorted nearest-first, so the first entry is the closest.
        return self.nearby_players[0] if self.nearby_players else None

    def to_vector(self) -> np.ndarray:
        """Flatten the perception into a fixed-length numpy vector.

        Every value is scaled to roughly the -1..1 range so a neural network
        can consume it directly. The layout is fixed; `VECTOR_SIZE` records
        its length so a network can size its input layer.
        """

        # A helper to scale a distance by the vision radius and cap it at 1.0.
        def scaled(distance: float) -> float:
            # Infinite (unknown) distances become 1.0, meaning "as far as I can see".
            return 1.0 if distance == np.inf else min(1.0, distance / self.vision_radius)

        # The nearest other player, or a stand-in with zeros if nobody is in sight.
        threat = self.nearest_threat
        # Values describing the nearest player (all zeros when alone).
        threat_values = [
            threat.dx / self.vision_radius if threat else 0.0,
            threat.dy / self.vision_radius if threat else 0.0,
            scaled(threat.distance) if threat else 1.0,
            threat.threat if threat else 0.0,
            threat.health if threat else 0.0,
        ]
        # One-hot encoding of the terrain underfoot (water, sand, grass, rock).
        terrain_one_hot = [
            float(self.terrain_here is kind)
            for kind in (TerrainType.WATER, TerrainType.SAND, TerrainType.GRASS, TerrainType.ROCK)
        ]
        # Assemble every feature in a fixed order.
        values = [
            self.thirst,
            self.hunger,
            self.health,
            self.survival_score,
            self.training_score,
            self.weapon_quality,
            self.reach / 3.0,
            min(1.0, self.food_count / 5.0),
            min(1.0, self.medicine_count / 3.0),
            float(self.in_water),
            self.hunt_difficulty,
            *self.downhill,
            *self.water_direction,
            scaled(self.water_distance),
            *self.grass_direction,
            scaled(self.grass_distance),
            *self.center_direction,
            self.center_distance,
            int(self.resource_here_kind) / 3.0,
            min(1.0, self.resource_here_quantity / 5.0),
            self.resource_here_quality,
            *self.nearby_resource_direction,
            scaled(self.nearby_resource_distance),
            int(self.nearby_resource_kind) / 3.0,
            *threat_values,
            min(1.0, len(self.nearby_players) / 5.0),
            float(self.in_danger_zone),
            max(-1.0, min(1.0, self.hazard_distance / self.vision_radius)),
            float(self.hazard_closing),
            *self.safe_direction,
            self.day_fraction,
            self.alive_fraction,
            float(self.field_known),
            self.field_strength,
            self.strongest_remaining,
            self.my_rank,
            *terrain_one_hot,
        ]
        # Return as a float array.
        return np.array(values, dtype=float)


# The length of `Perception.to_vector()`. Count: 11 body/terrain + 2 downhill
# + 3 water + 3 grass + 3 centre + 3 here + 4 nearby supply + 5 threat
# + 1 crowd + 3 hazard (inside, distance, closing) + 2 safe + 2 clock
# + 4 field (known, strength, strongest, rank) + 4 one-hot = 50.
VECTOR_SIZE = 50

# Human-readable names for every slot of the vector, in order (for the dashboard's network visualiser).
VECTOR_NAMES = [
    "thirst",
    "hunger",
    "health",
    "survival score",
    "training score",
    "weapon quality",
    "reach",
    "food carried",
    "medkits carried",
    "in water",
    "hunt difficulty",
    "downhill dx",
    "downhill dy",
    "water dx",
    "water dy",
    "water distance",
    "grass dx",
    "grass dy",
    "grass distance",
    "centre dx",
    "centre dy",
    "centre distance",
    "loot here kind",
    "loot here qty",
    "loot here quality",
    "nearby loot dx",
    "nearby loot dy",
    "nearby loot distance",
    "nearby loot kind",
    "threat dx",
    "threat dy",
    "threat distance",
    "threat level",
    "threat health",
    "players in sight",
    "in danger zone",
    "hazard distance",
    "hazard closing",
    "safe dx",
    "safe dy",
    "day fraction",
    "alive fraction",
    "field known",
    "field strength",
    "strongest remaining",
    "my rank",
    "on water",
    "on sand",
    "on grass",
    "on rock",
]
