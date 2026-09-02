"""player.py - the tribute's body.

Chapter 4 says the body is "their ability to perform actions: moving,
eating, fighting". Every method below defines one of those actions
mathematically. The body never decides anything; it gathers a `Perception`,
asks its `Brain`, and then does what it is told.
"""

# A small record type for fight results.
from dataclasses import dataclass

# numpy for the generator and vectorised supply scanning.
import numpy as np

# The action vocabulary.
from hunger_games.actions import Action

# The arena and a sign helper.
from hunger_games.arena import Arena, sign

# The brain interface.
from hunger_games.brain.base import Brain

# What the player senses.
from hunger_games.perception import NearbyPlayer, Perception

# Supply kinds.
from hunger_games.resources import ResourceKind, weapon_reach


@dataclass
class FightOutcome:
    """Who won a fight, who lost, and how badly the loser was hurt."""

    # The player who came out on top.
    winner: "Player"
    # The player who took the damage.
    loser: "Player"
    # How much health the loser lost.
    damage: float


class Player:
    """One tribute: their stats, position, inventory and brain."""

    # How much of the thirst bar one DRINK restores.
    DRINK_AMOUNT = 0.5
    # How much of the hunger bar one ration restores.
    EAT_AMOUNT = 0.35
    # How much of the health bar one medkit restores.
    HEAL_AMOUNT = 0.40
    # How much of the health bar one tick of REST restores.
    REST_AMOUNT = 0.02
    # Every 0.1 of survival score above the difficulty earns one more ration (chapter 4).
    HUNT_STEP = 0.1
    # Base damage a fight inflicts on the loser.
    BASE_DAMAGE = 0.35
    # Extra damage per point of the winner's weapon quality.
    WEAPON_DAMAGE = 0.45
    # Health below this is a serious wound: it bleeds and will not close by resting.
    WOUND_THRESHOLD = 0.5
    # How much health a serious wound drains per tick (about five days from 0.5 to death).
    BLEED_PER_TICK = 0.004

    def __init__(
        self,
        player_id: int,
        name: str,
        district: int,
        training_score: int,
        survival_score: float,
        brain: Brain,
        sex: str = "F",
    ) -> None:
        """Create a healthy, empty-handed tribute standing at (0, 0)."""
        # Unique number used to target this player in ATTACK actions.
        self.player_id = player_id
        # Display name.
        self.name = name
        # Home district, 1 to 12.
        self.district = district
        # The 1-to-12 score the game makers gave in training.
        self.training_score = training_score
        # Fixed 0-to-1 aptitude for finding food (chapter 4's survival score).
        self.survival_score = survival_score
        # The decision-maker.
        self.brain = brain
        # "F" or "M"; the renderer draws circles and squares to tell them apart.
        self.sex = sex
        # Extra sponsor favour granted by the game maker (0.0 to 1.0).
        self.favor_bonus = 0.0
        # The sponsors' current opinion of this tribute (updated by SponsorPool).
        self.favor = 0.0
        # Column position in the arena.
        self.x = 0
        # Row position in the arena.
        self.y = 0
        # Thirst bar: 1.0 full, 0.0 dead.
        self.thirst = 1.0
        # Hunger bar: 1.0 full, 0.0 dead.
        self.hunger = 1.0
        # Health bar: 1.0 unhurt, 0.0 dead.
        self.health = 1.0
        # Rations carried.
        self.food = 0
        # Medkits carried.
        self.medicine = 0
        # Quality of the best weapon carried (0.0 = fists).
        self.weapon_quality = 0.0
        # Number of players this one has eliminated.
        self.kills = 0
        # Whether the player is still in the games.
        self.alive = True
        # Final placing (1 = victor); filled in when eliminated or at the end.
        self.placement: int | None = None
        # Text description of how the player died, if they did.
        self.cause_of_death: str | None = None
        # Who killed them, if a player did.
        self.killer_id: int | None = None
        # The last action taken (handy for the renderer and for debugging).
        self.last_action: Action | None = None
        # The last perception (the dashboard's network visualiser feeds it through the brain again).
        self.last_perception: Perception | None = None

    # ----------------------------------------------------------- derived

    @property
    def threat_level(self) -> float:
        """How dangerous this player looks: skill, weapon and health combined."""
        # A weighted blend, matching what the voting brain assumes.
        return 0.4 * self.survival_score + 0.4 * self.weapon_quality + 0.2 * self.health

    @property
    def reach(self) -> int:
        """How many cells away this player can strike with their current weapon."""
        # Delegate to the weapon table.
        return weapon_reach(self.weapon_quality)

    @property
    def position(self) -> tuple[int, int]:
        """(x, y) as a tuple."""
        # Bundle the coordinates.
        return self.x, self.y

    def distance_to(self, other: "Player") -> int:
        """Chebyshev (king-move) distance to another player."""
        # The larger of the two axis gaps.
        return max(abs(self.x - other.x), abs(self.y - other.y))

    # ------------------------------------------------------------ sensing

    def perceive(
        self,
        arena: Arena,
        others: list["Player"],
        lethal_here: bool,
        day_fraction: float,
        alive_fraction: float,
        vision_radius: int,
        landmark_radius: int | None = None,
        hazard_distance: float = 999.0,
        hazard_closing: bool = False,
        field: tuple[bool, float, float, float] | None = None,
    ) -> Perception:
        """Gather everything this player can sense into a `Perception`."""
        # Shorthand for my coordinates.
        x, y = self.x, self.y
        # Lakes and meadows can be spotted from further away than people (default: three times as far).
        landmark_radius = landmark_radius if landmark_radius is not None else vision_radius * 3
        # Steps to the nearest water.
        water_distance = arena.distance_to_water(x, y)
        # Only tell the brain which way water lies if it is close enough to spot.
        water_direction = arena.direction_to_water(x, y) if water_distance <= landmark_radius else (0, 0)
        # Steps to the nearest grass.
        grass_distance = arena.distance_to_grass(x, y)
        # Same rule for grass.
        grass_direction = arena.direction_to_grass(x, y) if grass_distance <= landmark_radius else (0, 0)
        # What is in my cell.
        here_kind, here_quantity, here_quality = arena.resources.peek(x, y)
        # The nearest supply within sight.
        resource_direction, resource_distance, resource_kind = self._scan_resources(arena, vision_radius)
        # Other living players within sight.
        nearby = []
        # Check every other player.
        for other in others:
            # Skip myself and the dead.
            if other is self or not other.alive:
                continue
            # Offsets from me.
            dx, dy = other.x - x, other.y - y
            # King-move distance.
            distance = max(abs(dx), abs(dy))
            # Only players within sight are perceived.
            if distance <= vision_radius:
                # Record what I can tell about them.
                nearby.append(NearbyPlayer(other.player_id, dx, dy, float(distance), other.threat_level, other.health))
        # Closest first, so `nearest_threat` is a simple index.
        nearby.sort(key=lambda entry: entry.distance)
        # What the sky and the cannon have told me about the field (unknown if not given).
        field_known, field_strength, strongest, my_rank = field if field is not None else (False, 0.0, 0.0, 0.5)
        # Bundle it all up.
        return Perception(
            thirst=self.thirst,
            hunger=self.hunger,
            health=self.health,
            survival_score=self.survival_score,
            training_score=self.training_score / 12.0,
            weapon_quality=self.weapon_quality,
            reach=self.reach,
            food_count=self.food,
            medicine_count=self.medicine,
            terrain_here=arena.terrain_at(x, y),
            in_water=arena.is_water(x, y),
            hunt_difficulty=arena.hunt_difficulty_at(x, y),
            downhill=arena.downhill_direction(x, y),
            water_direction=water_direction,
            water_distance=water_distance,
            grass_direction=grass_direction,
            grass_distance=grass_distance,
            center_direction=arena.direction_to_center(x, y),
            center_distance=arena.normalized_distance_from_center(x, y),
            resource_here_kind=here_kind,
            resource_here_quantity=here_quantity,
            resource_here_quality=here_quality,
            nearby_resource_direction=resource_direction,
            nearby_resource_distance=resource_distance,
            nearby_resource_kind=resource_kind,
            nearby_players=nearby,
            in_danger_zone=lethal_here,
            hazard_distance=hazard_distance,
            hazard_closing=hazard_closing,
            safe_direction=arena.direction_to_center(x, y),
            day_fraction=day_fraction,
            alive_fraction=alive_fraction,
            field_known=field_known,
            field_strength=field_strength,
            strongest_remaining=strongest,
            my_rank=my_rank,
            vision_radius=vision_radius,
        )

    def _scan_resources(self, arena: Arena, vision_radius: int) -> tuple[tuple[int, int], float, ResourceKind]:
        """Find the nearest supply within sight using a numpy window (fast)."""
        # Left edge of the window, clipped to the grid.
        x0 = max(0, self.x - vision_radius)
        # Right edge (exclusive), clipped to the grid.
        x1 = min(arena.width, self.x + vision_radius + 1)
        # Top edge, clipped.
        y0 = max(0, self.y - vision_radius)
        # Bottom edge (exclusive), clipped.
        y1 = min(arena.height, self.y + vision_radius + 1)
        # The slice of the kind grid I can see.
        window = arena.resources.kind[y0:y1, x0:x1]
        # Row and column indices (within the window) of every non-empty cell.
        ys, xs = np.nonzero(window)
        # Nothing in sight.
        if len(xs) == 0:
            return (0, 0), float("inf"), ResourceKind.NONE
        # Offsets of each supply from me.
        dxs = xs + x0 - self.x
        # Same for rows.
        dys = ys + y0 - self.y
        # King-move distance to each.
        distances = np.maximum(np.abs(dxs), np.abs(dys))
        # The closest one.
        index = int(np.argmin(distances))
        # A single step toward it.
        direction = (sign(int(dxs[index])), sign(int(dys[index])))
        # Report the direction, distance and kind.
        return direction, float(distances[index]), ResourceKind(int(window[ys[index], xs[index]]))

    def decide(self, perception: Perception, rng: np.random.Generator) -> Action:
        """Ask the brain what to do."""
        # Delegate entirely to the brain.
        action = self.brain.decide(perception, rng)
        # Remember both for the dashboard.
        self.last_action = action
        # The perception.
        self.last_perception = perception
        # Hand it back to the game.
        return action

    # ------------------------------------------------------------- body

    def move(self, dx: int, dy: int, arena: Arena, rng: np.random.Generator) -> bool:
        """Step one cell; returns False if blocked or slowed by the terrain."""
        # The cell we are trying to reach.
        nx, ny = self.x + dx, self.y + dy
        # Refuse to leave the arena.
        if not arena.is_walkable(nx, ny):
            return False
        # Water and rock are slow: the step sometimes fails (this is how chases end).
        if rng.random() > arena.move_success_at(nx, ny):
            return False
        # Update the column.
        self.x = nx
        # Update the row.
        self.y = ny
        # Report success.
        return True

    def drink(self, arena: Arena) -> bool:
        """Drink if standing in water; returns whether it worked."""
        # Drinking dry land does nothing.
        if not arena.is_water(self.x, self.y):
            return False
        # Raise the thirst bar, capped at full.
        self.thirst = min(1.0, self.thirst + self.DRINK_AMOUNT)
        # Report success.
        return True

    def eat(self) -> bool:
        """Eat one ration if carrying any; returns whether it worked."""
        # Nothing to eat.
        if self.food <= 0:
            return False
        # Use up one ration.
        self.food -= 1
        # Raise the hunger bar, capped at full.
        self.hunger = min(1.0, self.hunger + self.EAT_AMOUNT)
        # Report success.
        return True

    def hunt(self, arena: Arena, rng: np.random.Generator, chaos: float) -> int:
        """Try to catch food here, exactly as chapter 4 describes.

        Compare survival score (plus chaos-scaled luck) to the terrain's
        difficulty: below it, nothing; equal, one ration; plus one more for
        every 0.1 above.
        """
        # How hard hunting is on this terrain.
        difficulty = arena.hunt_difficulty_at(self.x, self.y)
        # Luck: zero at chaos 0, a wide wobble at chaos 1.
        luck = chaos * rng.normal(0.0, 0.15)
        # The effective score for this attempt.
        roll = self.survival_score + luck
        # Below the difficulty: fail.
        if roll < difficulty:
            return 0
        # One ration for matching, plus one per 0.1 above.
        caught = 1 + int((roll - difficulty) / self.HUNT_STEP)
        # Add to the pack.
        self.food += caught
        # Report the haul.
        return caught

    def pick_up(self, arena: Arena) -> ResourceKind:
        """Take whatever is in this cell; returns the kind taken."""
        # Remove the supplies from the arena.
        kind, quantity, quality = arena.resources.take(self.x, self.y)
        # Food goes in the pack.
        if kind is ResourceKind.FOOD:
            self.food += quantity
        # A weapon is kept only if it beats the one we have.
        elif kind is ResourceKind.WEAPON:
            self.weapon_quality = max(self.weapon_quality, quality)
        # Medicine goes in the pack.
        elif kind is ResourceKind.MEDICINE:
            self.medicine += quantity
        # Report what was found.
        return kind

    def heal(self) -> bool:
        """Use one medkit if carrying any; returns whether it worked."""
        # Nothing to use.
        if self.medicine <= 0:
            return False
        # Use up one medkit.
        self.medicine -= 1
        # Raise the health bar, capped at full.
        self.health = min(1.0, self.health + self.HEAL_AMOUNT)
        # Report success.
        return True

    def rest(self) -> None:
        """Recover a sliver of health, but only from minor wounds: a deep wound needs medicine."""
        # A serious wound does not close by resting.
        if self.health < self.WOUND_THRESHOLD:
            return
        # Raise the health bar a little, capped at full.
        self.health = min(1.0, self.health + self.REST_AMOUNT)

    def attack(self, other: "Player", rng: np.random.Generator, chaos: float) -> FightOutcome:
        """Fight another player.

        Each side's strength is their threat level plus chaos-scaled luck.
        The stronger side wins and the loser takes damage that scales with
        the winner's weapon. Even the winner gets a little hurt.
        """
        # My strength for this fight.
        my_roll = self.threat_level + chaos * rng.normal(0.0, 0.2)
        # Their strength for this fight.
        their_roll = other.threat_level + chaos * rng.normal(0.0, 0.2)
        # Decide the winner (ties go to the attacker).
        winner, loser = (self, other) if my_roll >= their_roll else (other, self)
        # Damage to the loser: worse with a better weapon.
        damage = self.BASE_DAMAGE + self.WEAPON_DAMAGE * winner.weapon_quality
        # Apply it.
        loser.health -= damage
        # The winner takes a scratch that scales with the loser's weapon.
        winner.health -= 0.05 + 0.10 * loser.weapon_quality
        # Report the result.
        return FightOutcome(winner, loser, damage)

    def tick_needs(self, thirst_per_tick: float, hunger_per_tick: float) -> None:
        """Time passes: thirst and hunger bars drain."""
        # Drain thirst, not below zero.
        self.thirst = max(0.0, self.thirst - thirst_per_tick)
        # Drain hunger, not below zero.
        self.hunger = max(0.0, self.hunger - hunger_per_tick)
        # A serious untreated wound bleeds (even a cut is deadly in the arena).
        if 0.0 < self.health < self.WOUND_THRESHOLD:
            self.health = max(0.0, self.health - self.BLEED_PER_TICK)

    def natural_cause_of_death(self) -> str | None:
        """If a need has hit zero, say which one; otherwise None."""
        # Thirst kills first because it drains fastest.
        if self.thirst <= 0.0:
            return "dehydration"
        # Then starvation.
        if self.hunger <= 0.0:
            return "starvation"
        # Then bleeding out from an untreated wound.
        if self.health <= 0.0:
            return "untreated wound"
        # Otherwise still alive from natural causes.
        return None
