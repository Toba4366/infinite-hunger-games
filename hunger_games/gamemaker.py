"""gamemaker.py - the head game maker's interventions.

Chapter 1 complains that Seneca Crane constantly herds players together.
Chapter 3 counts those interventions as their own elimination category. To
reproduce that, this class watches for quiet stretches with no eliminations
and then shrinks a "safe circle" toward the centre. Anyone caught outside
takes damage every tick. Turn it off in the config to test whether the ring
layout can keep the games moving without help.
"""

# The arena, for distances.
from hunger_games.arena import Arena

# The settings.
from hunger_games.config import SimulationConfig


class Gamemaker:
    """Shrinks the safe area of the arena when the games go quiet."""

    # How much health a player outside the safe circle loses per tick (about 12 ticks to die).
    DAMAGE_PER_TICK = 0.08
    # The circle never shrinks below this many cells, leaving room for a final fight.
    MIN_SAFE_RADIUS = 8.0
    # The label used for eliminations in the results CSV.
    WEAPON_LABEL = "arena hazard"

    def __init__(self, config: SimulationConfig, arena: Arena) -> None:
        """Start with the whole arena safe."""
        # Whether interventions are allowed at all.
        self.enabled = config.gamemaker_enabled
        # How many ticks of silence trigger an intervention.
        self.quiet_ticks = int(config.quiet_days_before_intervention * config.ticks_per_day)
        # Keep the arena for distance checks.
        self.arena = arena
        # Start the safe circle exactly at the arena's edge, so the first shrink bites at once.
        self.safe_radius = arena.radius * 1.5
        # Shrink so that the circle closes from the edge to the centre over `intervention_days` days of shrinking.
        self.shrink_per_tick = arena.radius / (config.intervention_days * config.ticks_per_day)
        # Once triggered, keep shrinking for at least this many ticks even if someone dies.
        self.minimum_shrink_ticks = config.ticks_per_day
        # The tick until which the current intervention is committed to run.
        self.shrink_until_tick = -1
        # Count of separate interventions (for the results).
        self.interventions = 0
        # Whether the circle is shrinking right now.
        self.shrinking = False

    def update(self, tick: int, last_elimination_tick: int, alive_count: int) -> None:
        """Called once per tick: decide whether to shrink the circle."""
        # Disabled, or the games are over: never intervene.
        if not self.enabled or alive_count <= 1:
            self.shrinking = False
            return
        # How long since anyone was eliminated.
        quiet_for = tick - last_elimination_tick
        # Not quiet enough yet, and no intervention already committed: leave the circle alone.
        if quiet_for < self.quiet_ticks and tick >= self.shrink_until_tick:
            self.shrinking = False
            return
        # A new intervention starts when shrinking switches on.
        if not self.shrinking:
            self.interventions += 1
            self.shrinking = True
            # Commit to a full day of shrinking so one lucky kill does not cancel it.
            self.shrink_until_tick = tick + self.minimum_shrink_ticks
        # Pull the circle inward, but never below the minimum.
        self.safe_radius = max(self.MIN_SAFE_RADIUS, self.safe_radius - self.shrink_per_tick)

    @property
    def is_active(self) -> bool:
        """Has the circle ever shrunk into the arena? (Used by the renderer.)"""
        # The starting radius is 1.5 times the arena radius by construction.
        return self.safe_radius < self.arena.radius * 1.5

    def hazard_distance(self, x: int, y: int) -> float:
        """Cells of safe ground between this cell and the lethal edge (negative if outside)."""
        # Positive inside the circle, negative outside it.
        return self.safe_radius - self.arena.distance_from_center(x, y)

    def is_lethal(self, x: int, y: int) -> bool:
        """Is the given cell outside the safe circle?"""
        # Compare the cell's distance from the centre to the safe radius.
        return self.arena.distance_from_center(x, y) > self.safe_radius
