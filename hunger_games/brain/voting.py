"""brain/voting.py - the "instincts vote on an action" brain from chapter 4.

Each instinct (thirst, hunger, survival, danger, greed) looks at the
perception and casts votes for the action it prefers. Lower scores cast
*more* votes, so a player about to die of thirst will insist on drinking.
The action with the most votes wins, unless the chaos dial adds noise.

Every weight that shapes the votes is stored in a genome vector, so a
genetic algorithm can evolve better voters without touching this code.
"""

# numpy for the genome vector and the weighted random choice.
import numpy as np

# Actions the brain can vote for.
from hunger_games.actions import DIRECTIONS, Action, ActionType

# The base class.
from hunger_games.brain.base import Brain

# The perception type.
from hunger_games.perception import Perception

# Supply kinds.
from hunger_games.resources import ResourceKind

# Terrain kinds.
from hunger_games.terrain import TerrainType

# The names of the tunable genes, in the order they appear in the genome.
GENE_NAMES = [
    "thirst_weight",  # how loudly the thirst instinct votes
    "hunger_weight",  # how loudly the hunger instinct votes
    "survival_weight",  # how loudly long-term planning votes
    "danger_weight",  # how loudly the fight-or-flight instinct votes
    "greed_weight",  # how loudly the loot instinct votes
    "aggression",  # 0.0 = never picks fights, 1.0 = always attacks
    "caution",  # 0.0 = never runs, 1.0 = always runs
    "urgency_power",  # how sharply low scores turn into extra votes
]
# The default gene values, tuned by hand to give sensible behaviour.
DEFAULT_GENES = np.array([1.0, 1.0, 1.0, 1.5, 0.6, 0.5, 0.5, 2.0], dtype=float)


class Ballot:
    """Collects votes for actions and picks the winner."""

    def __init__(self) -> None:
        """Start with an empty ballot box."""
        # Maps each candidate action to its running vote total.
        self.votes: dict[Action, float] = {}

    def cast(self, action: Action, votes: float) -> None:
        """Add `votes` to an action's total."""
        # Ignore zero or negative votes.
        if votes <= 0.0:
            return
        # Add to whatever the action already has (starting from zero).
        self.votes[action] = self.votes.get(action, 0.0) + votes

    def winner(self, rng: np.random.Generator, chaos: float) -> Action:
        """Pick the winning action.

        With chaos 0.0 the top-voted action always wins. With chaos 1.0 each
        action's chance of winning is proportional to its votes. In between,
        the two rules are blended.
        """
        # An empty ballot means the player has nothing better to do than rest.
        if not self.votes:
            return Action(ActionType.REST)
        # The candidate actions in a fixed order.
        actions = list(self.votes.keys())
        # Their vote totals as an array.
        counts = np.array([self.votes[action] for action in actions], dtype=float)
        # The index of the top-voted action.
        best = int(np.argmax(counts))
        # No chaos: the favourite wins outright.
        if chaos <= 0.0:
            return actions[best]
        # Chance of each action if we drew in proportion to votes.
        proportional = counts / counts.sum()
        # Chance of each action if we always picked the favourite.
        certain = np.zeros_like(proportional)
        # Only the favourite gets any chance in the certain case.
        certain[best] = 1.0
        # Blend the two according to the chaos dial.
        probabilities = (1.0 - chaos) * certain + chaos * proportional
        # Draw one action index using those probabilities.
        choice = int(rng.choice(len(actions), p=probabilities))
        # Return the drawn action.
        return actions[choice]


class VotingBrain(Brain):
    """The chapter 4 brain: instincts cast votes, the most-voted action wins."""

    # Label for the results CSV.
    name = "voting"
    # A stronger player closer than this triggers all-out flight.
    PANIC_DISTANCE = 4
    # A need bar below this level is life-threatening.
    CRITICAL_LEVEL = 0.2
    # Extra votes a life-threatening need casts, enough to beat fear (about 15) and greed (about 10).
    CRITICAL_BONUS = 20.0

    def __init__(self, chaos: float = 0.0, genome: np.ndarray | None = None, endgame: bool = False) -> None:
        """Create a voting brain with default genes unless a genome is supplied."""
        # Store the chaos dial via the base class.
        super().__init__(chaos)
        # Whether the endgame instinct (head for the centre when few remain) is switched on.
        self.endgame = endgame
        # Start from the hand-tuned defaults (copy so each brain owns its genes).
        self.genes = DEFAULT_GENES.copy()
        # If a genome was supplied (e.g. by a genetic algorithm), load it.
        if genome is not None:
            self.set_genome(genome)

    # ---------------------------------------------------------- genome API

    def genome(self) -> np.ndarray:
        """Return a copy of the genes so callers cannot change them by accident."""
        # A copy, not the original array.
        return self.genes.copy()

    def set_genome(self, genome: np.ndarray) -> None:
        """Replace the genes with a new vector of the same length."""
        # Convert to a float array and check the size matches.
        genes = np.asarray(genome, dtype=float)
        # A wrong-sized genome is a programming error worth catching early.
        if genes.shape != DEFAULT_GENES.shape:
            raise ValueError(f"VotingBrain genome must have {len(DEFAULT_GENES)} values")
        # Store our own copy.
        self.genes = genes.copy()

    def gene(self, gene_name: str) -> float:
        """Read one gene by name."""
        # Find the gene's position in the vector and return its value.
        return float(self.genes[GENE_NAMES.index(gene_name)])

    # ----------------------------------------------------------- helpers

    def urgency(self, score: float) -> float:
        """Turn a 0-to-1 need score into a number of votes (0 to 10).

        A full bar (1.0) gives zero votes; an empty bar (0.0) gives ten.
        `urgency_power` above 1 keeps votes low until the need becomes serious.
        """
        # Clamp the score into range in case of rounding errors.
        score = float(np.clip(score, 0.0, 1.0))
        # Invert (low score = high urgency), sharpen, and scale to ten votes.
        votes = ((1.0 - score) ** self.gene("urgency_power")) * 10.0
        # A nearly empty bar is an emergency: it must out-vote fear and greed (chapter 4's "insist on drinking").
        if score < self.CRITICAL_LEVEL:
            votes += self.CRITICAL_BONUS
        # Done.
        return votes

    @staticmethod
    def random_step(rng: np.random.Generator) -> tuple[int, int]:
        """A random one of the eight compass directions (for wandering)."""
        # Pick a random index into the direction list.
        return DIRECTIONS[int(rng.integers(len(DIRECTIONS)))]

    # ---------------------------------------------------------- deciding

    def decide(self, perception: Perception, rng: np.random.Generator) -> Action:
        """Let every instinct vote, then pick the winner."""
        # A fresh ballot for this tick.
        ballot = Ballot()
        # Escaping the game makers overrides everything else.
        self._vote_hazard(perception, ballot)
        # Thirst instinct.
        self._vote_thirst(perception, ballot, rng)
        # Hunger instinct.
        self._vote_hunger(perception, ballot, rng)
        # Long-term survival instinct.
        self._vote_survival(perception, ballot)
        # Fight-or-flight instinct.
        self._vote_danger(perception, ballot)
        # Loot instinct.
        self._vote_greed(perception, ballot)
        # Endgame instinct: as the field thins, go looking for the last opponents.
        self._vote_endgame(perception, ballot)
        # A trickle of votes for resting and wandering so nobody freezes.
        self._vote_idle(perception, ballot, rng)
        # Count the votes.
        return ballot.winner(rng, self.chaos)

    def _vote_hazard(self, p: Perception, ballot: Ballot) -> None:
        """If the game makers have made this cell lethal, vote to leave it."""
        # Nothing to do if there is no direction to safety.
        if p.safe_direction == (0, 0):
            return
        # Already inside the lethal zone: fifty votes beats any other instinct's ten.
        if p.in_danger_zone:
            ballot.cast(Action.move(*p.safe_direction), 50.0)
        # The fog is closing and it is within sight: head inward, harder the closer it is.
        elif p.hazard_closing and p.hazard_distance <= p.vision_radius:
            ballot.cast(
                Action.move(*p.safe_direction), 20.0 * (1.0 - p.hazard_distance / (p.vision_radius + 1.0)) + 5.0
            )

    def _vote_thirst(self, p: Perception, ballot: Ballot, rng: np.random.Generator) -> None:
        """Vote to drink, or to walk toward water."""
        # A mostly-full thirst bar has no opinion.
        if p.thirst > 0.85:
            return
        # Votes grow as the bar empties.
        votes = self.gene("thirst_weight") * self.urgency(p.thirst)
        # Standing in water: drink.
        if p.in_water:
            ballot.cast(Action(ActionType.DRINK), votes)
        # Water has been spotted: walk toward it.
        elif p.water_direction != (0, 0):
            ballot.cast(Action.move(*p.water_direction), votes)
        # No water in sight but there is a slope: follow it down (the video's rule).
        elif p.downhill != (0, 0):
            ballot.cast(Action.move(*p.downhill), votes)
        # Flat and dry: wander and hope.
        else:
            ballot.cast(Action.move(*self.random_step(rng)), votes * 0.5)

    def _vote_hunger(self, p: Perception, ballot: Ballot, rng: np.random.Generator) -> None:
        """Vote to eat, to hunt here, or to move somewhere hunting is easier."""
        # A mostly-full hunger bar has no opinion.
        if p.hunger > 0.8:
            return
        # Votes grow as the bar empties.
        votes = self.gene("hunger_weight") * self.urgency(p.hunger)
        # Food in the pack: eat it.
        if p.food_count > 0:
            ballot.cast(Action(ActionType.EAT), votes)
            return
        # How much better than the local difficulty my survival score is.
        expected = p.survival_score - p.hunt_difficulty
        # A fair chance of success here: vote to hunt, more confidently the bigger the margin.
        if expected >= -0.05:
            ballot.cast(Action(ActionType.HUNT), votes * max(0.2, 0.5 + expected))
        # Not on grass but grass has been spotted: consider walking to easier hunting.
        if p.terrain_here is not TerrainType.GRASS and p.grass_direction != (0, 0):
            # Walk with full force if hunting here is hopeless, less if it is merely worse.
            ballot.cast(Action.move(*p.grass_direction), votes * (1.0 if expected < 0 else 0.6))
        # Not on grass and none in sight: explore.
        elif p.terrain_here is not TerrainType.GRASS:
            ballot.cast(Action.move(*self.random_step(rng)), votes * 0.4)

    def _vote_survival(self, p: Perception, ballot: Ballot) -> None:
        """Long-term thinking: stockpile food, patch wounds, top off water."""
        # Skilled survivors plan more; the weight scales with their score.
        votes = self.gene("survival_weight") * (0.5 + p.survival_score) * 2.0
        # Low on rations while standing on grass: hunt now, before hunger bites.
        if p.food_count < 2 and p.terrain_here is TerrainType.GRASS and p.hunger < 0.9:
            ballot.cast(Action(ActionType.HUNT), votes)
        # Wounded with a medkit: use it, more urgently the worse the wound.
        if p.health < 0.6 and p.medicine_count > 0:
            ballot.cast(Action(ActionType.HEAL), votes + self.urgency(p.health))
        # Slightly wounded and alone: rest.
        elif p.health < 0.8 and not p.nearby_players:
            ballot.cast(Action(ActionType.REST), votes * 0.5)
        # Already in water and not quite full: keep drinking a little.
        if p.in_water and p.thirst < 0.95:
            ballot.cast(Action(ActionType.DRINK), votes * 0.3)

    def _vote_danger(self, p: Perception, ballot: Ballot) -> None:
        """Fight or flight, based on who looks stronger."""
        # The nearest other player, if any.
        threat = p.nearest_threat
        # Nobody in sight: nothing to vote on.
        if threat is None:
            return
        # My own deadliness, computed the same way the body reports it.
        my_strength = 0.4 * p.survival_score + 0.4 * p.weapon_quality + 0.2 * p.health
        # Positive means I look stronger than them.
        advantage = my_strength - threat.threat
        # Closer threats matter more (1.0 when adjacent, ~0 at the edge of sight).
        closeness = 1.0 - threat.distance / (p.vision_radius + 1.0)
        # Base votes for this instinct.
        votes = self.gene("danger_weight") * closeness * 10.0
        # Aggressive players attack even at a slight disadvantage.
        confidence = advantage + (self.gene("aggression") - 0.5)
        # Feeling confident: fight.
        if confidence > 0.0:
            # Within my weapon's reach: attack directly.
            if threat.distance <= p.reach:
                ballot.cast(Action.attack(threat.player_id), votes * (0.5 + self.gene("aggression")))
            # Not in reach yet: close in.
            else:
                ballot.cast(Action.move(*threat.direction_toward()), votes * (0.5 + self.gene("aggression")))
        # Not confident and they are close: run.
        elif threat.distance <= self.PANIC_DISTANCE:
            ballot.cast(Action.flee(*threat.direction_away()), votes * (0.5 + self.gene("caution")))
        # Not confident but they are still far: quietly drift away and carry on.
        else:
            ballot.cast(Action.flee(*threat.direction_away()), votes * 0.3 * (0.5 + self.gene("caution")))

    def _vote_greed(self, p: Perception, ballot: Ballot) -> None:
        """Grab loot, walk to visible loot, and drift toward the centre for better weapons."""
        # Base votes for this instinct.
        votes = self.gene("greed_weight") * 3.0
        # Standing on supplies: picking them up is nearly free, so vote hard.
        if p.resource_here_kind is not ResourceKind.NONE:
            ballot.cast(Action(ActionType.PICK_UP), votes * 3.0 + 5.0)
        # Supplies in sight (but not underfoot): walk toward them, closer = keener.
        if 0 < p.nearby_resource_distance <= p.vision_radius:
            ballot.cast(
                Action.move(*p.nearby_resource_direction),
                votes * (1.0 - p.nearby_resource_distance / (p.vision_radius + 1.0)),
            )
        # Poorly armed players are tempted toward the centre where the weapons are.
        if p.weapon_quality < 0.6 and p.center_direction != (0, 0):
            ballot.cast(
                Action.move(*p.center_direction),
                votes * (0.6 - p.weapon_quality) * (0.5 + self.gene("aggression")),
            )

    def _vote_endgame(self, p: Perception, ballot: Ballot) -> None:
        """The nightly cannon tells every tribute how many remain. When few are
        left, the bold ones stop waiting and head for the centre, where the
        best loot and the other survivors are. Without this, the last two
        tributes can wander opposite corners for days (the stalled endgame
        chapter 2 warns about).
        """
        # Off unless the config switched it on, and nothing to do while the arena is still crowded.
        if not self.endgame or p.alive_fraction > 0.5 or p.center_direction == (0, 0):
            return
        # Grows from 0 at half the field alive to 1 with two left.
        thinning = (0.5 - p.alive_fraction) / 0.5
        # Aggressive, well-armed tributes hunt; timid ones drift in more slowly.
        boldness = 0.3 + self.gene("aggression") * 0.7 + 0.3 * p.weapon_quality
        # Knowing you outrank most of the field (from the nightly sky) makes you bolder; the reverse makes you cautious.
        if p.field_known:
            boldness *= 0.5 + p.my_rank
        # Head for the centre.
        ballot.cast(Action.move(*p.center_direction), self.gene("danger_weight") * thinning**2 * boldness * 12.0)

    def _vote_idle(self, p: Perception, ballot: Ballot, rng: np.random.Generator) -> None:
        """Tiny votes so that a player with no strong opinion still does something."""
        # Resting is the fallback.
        ballot.cast(Action(ActionType.REST), 0.5)
        # Wandering keeps the arena moving.
        ballot.cast(Action.move(*self.random_step(rng)), 0.3)
