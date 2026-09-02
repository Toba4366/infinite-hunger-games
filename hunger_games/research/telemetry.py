"""research/telemetry.py - behavioural measurements taken while a game runs.

To prove a brain is learning the right habits, we need more than a win
count. This collector plugs into `Game.decision_hooks` and `Game.tick_hooks`
and tallies, for every decision, what the tribute chose against how thirsty,
hungry, hurt and threatened it was, where it stood, and how many tributes
remained. `summary()` turns the tallies into plain lists that can be saved
as JSON, merged across games, and drawn by research/plots.py.
"""

# numpy for the tallies.
import numpy as np

# Action kinds.
from hunger_games.actions import Action, ActionType

# The game type (for the tick hook) and the player type.
from hunger_games.game import Game

# What a tribute senses.
from hunger_games.perception import Perception

# The body.
from hunger_games.player import Player

# The action categories tracked, in a fixed order.
ACTION_NAMES = [kind.value for kind in ActionType]
# Position of each action kind in that order.
ACTION_INDEX = {kind: index for index, kind in enumerate(ActionType)}
# Need bars (thirst, hunger, health) are grouped into these five bins.
NEED_BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
# Labels for those bins.
NEED_BIN_LABELS = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
# The alive fraction is grouped into these five bins (from nearly everyone to the final pair).
ALIVE_BIN_EDGES = [0.0, 0.15, 0.3, 0.5, 0.75, 1.0001]
# Labels for those bins.
ALIVE_BIN_LABELS = ["final few", "<30%", "<50%", "<75%", "most alive"]
# The arena is divided into this many cells per side for heatmaps.
HEATMAP_CELLS = 30
# A weapon at least this good counts as "armed" for the armed/unarmed heatmaps.
ARMED_THRESHOLD = 0.4


def bin_index(value: float, edges: list[float]) -> int:
    """Which bin a value falls into (the last bin if it is off the top)."""
    # Walk the edges.
    for index in range(len(edges) - 1):
        # Inside this bin.
        if edges[index] <= value < edges[index + 1]:
            return index
    # Off the end.
    return len(edges) - 2


class BehaviorTelemetry:
    """Tallies decisions against internal state, danger, position and the size of the field."""

    def __init__(self, width: int, height: int, tracked_ids: set[int] | None = None) -> None:
        """Create empty tallies for an arena of the given size, optionally for a subset of tributes."""
        # Arena size, for the heatmaps.
        self.width = width
        # Height.
        self.height = height
        # Which tributes to measure (None = everyone).
        self.tracked_ids = tracked_ids
        # How many actions there are.
        n = len(ACTION_NAMES)
        # Total count of each action.
        self.action_counts = np.zeros(n)
        # Action counts split by the thirst bin at the moment of decision.
        self.action_by_thirst = np.zeros((5, n))
        # Split by hunger bin.
        self.action_by_hunger = np.zeros((5, n))
        # Split by health bin.
        self.action_by_health = np.zeros((5, n))
        # Split by alive-fraction bin.
        self.action_by_alive = np.zeros((5, n))
        # Attack (column 0) versus flee (column 1) counts by health bin, only when someone is in sight.
        self.combat_by_health = np.zeros((5, 2))
        # Where tributes spend their time.
        self.position_heat = np.zeros((HEATMAP_CELLS, HEATMAP_CELLS))
        # Where armed tributes spend their time.
        self.armed_heat = np.zeros((HEATMAP_CELLS, HEATMAP_CELLS))
        # Where unarmed tributes spend their time.
        self.unarmed_heat = np.zeros((HEATMAP_CELLS, HEATMAP_CELLS))
        # Sum of distance to the nearest visible tribute, by alive bin (and how many samples).
        self.proximity_sum = np.zeros(5)
        # Sample counts for the proximity sums.
        self.proximity_count = np.zeros(5)
        # Thirst level at every drink (histogram over ten bins).
        self.thirst_at_drink = np.zeros(10)
        # Hunger level at every meal.
        self.hunger_at_eat = np.zeros(10)
        # Health at every medkit use.
        self.health_at_heal = np.zeros(10)
        # Sums of thirst, hunger and health at the moment of death, and the count of deaths.
        self.death_needs = np.zeros(3)
        # How many deaths contributed.
        self.death_count = 0
        # Deaths by cause name.
        self.deaths_by_cause: dict[str, int] = {}
        # Ticks survived per tracked tribute per game.
        self.survival_ticks: list[int] = []
        # Kills per tracked tribute per game.
        self.kills: list[int] = []
        # Wins per tracked tribute per game (1 or 0).
        self.wins: list[int] = []
        # Placements per tracked tribute per game.
        self.placements: list[int] = []
        # Ticks survived after first dropping below half health (injury adaptation).
        self.post_injury_ticks: list[int] = []
        # Games observed.
        self.games = 0
        # Bookkeeping for the current game: who has been counted dead, and when injuries began.
        self._dead_seen: set[int] = set()
        # Tick of first serious injury per tribute.
        self._injured_at: dict[int, int] = {}

    # ---------------------------------------------------------- attaching

    def attach(self, game: Game) -> "BehaviorTelemetry":
        """Register the hooks on a game and return self."""
        # Decisions.
        game.decision_hooks.append(self.on_decision)
        # Ticks.
        game.tick_hooks.append(self.on_tick)
        # Chainable.
        return self

    def tracks(self, player: Player) -> bool:
        """Is this tribute one we measure?"""
        # Everyone, or the chosen subset.
        return self.tracked_ids is None or player.player_id in self.tracked_ids

    # -------------------------------------------------------------- hooks

    def on_decision(self, player: Player, perception: Perception, action: Action) -> None:
        """Tally one decision against the state it was made in."""
        # Only tracked tributes.
        if not self.tracks(player):
            return
        # Which action.
        a = ACTION_INDEX[action.kind]
        # Total.
        self.action_counts[a] += 1
        # By need bins.
        self.action_by_thirst[bin_index(perception.thirst, NEED_BIN_EDGES), a] += 1
        # Hunger.
        self.action_by_hunger[bin_index(perception.hunger, NEED_BIN_EDGES), a] += 1
        # Health.
        self.action_by_health[bin_index(perception.health, NEED_BIN_EDGES), a] += 1
        # Field size.
        alive_bin = bin_index(perception.alive_fraction, ALIVE_BIN_EDGES)
        # By alive bin.
        self.action_by_alive[alive_bin, a] += 1
        # Combat choices when someone is in sight.
        threat = perception.nearest_threat
        # Someone is there.
        if threat is not None:
            # Distance kept from them, by field size.
            self.proximity_sum[alive_bin] += threat.distance
            # Count it.
            self.proximity_count[alive_bin] += 1
            # Attack or flee.
            if action.kind is ActionType.ATTACK:
                self.combat_by_health[bin_index(perception.health, NEED_BIN_EDGES), 0] += 1
            elif action.kind is ActionType.FLEE:
                self.combat_by_health[bin_index(perception.health, NEED_BIN_EDGES), 1] += 1
        # Consumption timing.
        if action.kind is ActionType.DRINK:
            self.thirst_at_drink[min(9, int(perception.thirst * 10))] += 1
        elif action.kind is ActionType.EAT:
            self.hunger_at_eat[min(9, int(perception.hunger * 10))] += 1
        elif action.kind is ActionType.HEAL:
            self.health_at_heal[min(9, int(perception.health * 10))] += 1

    def on_tick(self, game: Game) -> None:
        """Tally positions, injuries and deaths at the end of a tick."""
        # Every tribute.
        for player in game.players:
            # Only tracked ones.
            if not self.tracks(player):
                continue
            # The living: positions and injuries.
            if player.alive:
                # Heatmap cell.
                hx = min(HEATMAP_CELLS - 1, int(player.x * HEATMAP_CELLS / self.width))
                # Row.
                hy = min(HEATMAP_CELLS - 1, int(player.y * HEATMAP_CELLS / self.height))
                # Overall.
                self.position_heat[hy, hx] += 1
                # Armed or not.
                if player.weapon_quality >= ARMED_THRESHOLD:
                    self.armed_heat[hy, hx] += 1
                else:
                    self.unarmed_heat[hy, hx] += 1
                # First serious injury.
                if player.health < Player.WOUND_THRESHOLD and player.player_id not in self._injured_at:
                    self._injured_at[player.player_id] = game.tick
            # The newly dead: bars at death.
            elif player.player_id not in self._dead_seen:
                # Count once.
                self._dead_seen.add(player.player_id)
                # Bars at death.
                self.death_needs += (player.thirst, player.hunger, max(0.0, player.health))
                # Count.
                self.death_count += 1
                # Cause.
                cause = player.cause_of_death or "unknown"
                self.deaths_by_cause[cause] = self.deaths_by_cause.get(cause, 0) + 1
                # Survival after injury.
                if player.player_id in self._injured_at:
                    self.post_injury_ticks.append(game.tick - self._injured_at[player.player_id])
        # At the end of the game, record outcomes.
        if game.is_over:
            self.on_game_end(game)

    def on_game_end(self, game: Game) -> None:
        """Record per-tribute outcomes once the game is over."""
        # Count the game.
        self.games += 1
        # Each tracked tribute.
        for player in game.players:
            # Only tracked ones.
            if not self.tracks(player):
                continue
            # Ticks survived.
            self.survival_ticks.append(game.death_ticks.get(player.player_id, game.tick))
            # Kills.
            self.kills.append(player.kills)
            # Wins.
            self.wins.append(1 if player.alive and len(game.alive_players) == 1 else 0)
            # Placement: survivors are placed by Game._finish() after this hook fires, so
            # compute their shared placing (the number of survivors) here when it is not set yet.
            self.placements.append(player.placement if player.placement is not None else len(game.alive_players))
            # Survivors who were injured survived the rest of the game after it.
            if player.alive and player.player_id in self._injured_at:
                self.post_injury_ticks.append(game.tick - self._injured_at[player.player_id])
        # Reset the per-game bookkeeping.
        self._dead_seen = set()
        # Injuries.
        self._injured_at = {}

    # ------------------------------------------------------------ summary

    def entropy(self) -> float:
        """Shannon entropy (in nats) of the overall action distribution: high = varied, 0 = one action only."""
        # Probabilities.
        total = self.action_counts.sum()
        # No decisions yet.
        if total == 0:
            return 0.0
        # Normalise.
        p = self.action_counts / total
        # Only non-zero entries contribute.
        p = p[p > 0]
        # Shannon.
        return float(-(p * np.log(p)).sum())

    def summary(self) -> dict:
        """Everything as plain lists and numbers, ready for JSON and for plots.py."""
        # Assemble.
        return {
            "games": self.games,
            "action_names": ACTION_NAMES,
            "action_counts": self.action_counts.tolist(),
            "action_by_thirst": self.action_by_thirst.tolist(),
            "action_by_hunger": self.action_by_hunger.tolist(),
            "action_by_health": self.action_by_health.tolist(),
            "action_by_alive": self.action_by_alive.tolist(),
            "combat_by_health": self.combat_by_health.tolist(),
            "position_heat": self.position_heat.tolist(),
            "armed_heat": self.armed_heat.tolist(),
            "unarmed_heat": self.unarmed_heat.tolist(),
            "proximity_sum": self.proximity_sum.tolist(),
            "proximity_count": self.proximity_count.tolist(),
            "thirst_at_drink": self.thirst_at_drink.tolist(),
            "hunger_at_eat": self.hunger_at_eat.tolist(),
            "health_at_heal": self.health_at_heal.tolist(),
            "death_needs": self.death_needs.tolist(),
            "death_count": self.death_count,
            "deaths_by_cause": dict(self.deaths_by_cause),
            "survival_ticks": list(self.survival_ticks),
            "kills": list(self.kills),
            "wins": list(self.wins),
            "placements": list(self.placements),
            "post_injury_ticks": list(self.post_injury_ticks),
            "entropy": self.entropy(),
            "mean_survival_ticks": float(np.mean(self.survival_ticks)) if self.survival_ticks else 0.0,
            "win_rate": float(np.mean(self.wins)) if self.wins else 0.0,
            "kill_rate": float(np.mean(self.kills)) if self.kills else 0.0,
            "mean_death_needs": (self.death_needs / max(1, self.death_count)).tolist(),
        }

    @staticmethod
    def merge(summaries: list[dict]) -> dict:
        """Add up several summaries (from several games or workers) into one."""
        # Nothing to merge.
        if not summaries:
            return BehaviorTelemetry(1, 1).summary()
        # Start from a copy of the first.
        merged = {key: (list(value) if isinstance(value, list) else value) for key, value in summaries[0].items()}
        # Keys that are numeric arrays to be summed.
        array_keys = [
            "action_counts",
            "action_by_thirst",
            "action_by_hunger",
            "action_by_health",
            "action_by_alive",
            "combat_by_health",
            "position_heat",
            "armed_heat",
            "unarmed_heat",
            "proximity_sum",
            "proximity_count",
            "thirst_at_drink",
            "hunger_at_eat",
            "health_at_heal",
            "death_needs",
        ]
        # Keys that are lists to be concatenated.
        list_keys = ["survival_ticks", "kills", "wins", "placements", "post_injury_ticks"]
        # Fold in the rest.
        for other in summaries[1:]:
            # Sum the arrays.
            for key in array_keys:
                merged[key] = (np.asarray(merged[key]) + np.asarray(other[key])).tolist()
            # Concatenate the lists.
            for key in list_keys:
                merged[key] = list(merged[key]) + list(other[key])
            # Counters.
            merged["games"] += other["games"]
            merged["death_count"] += other["death_count"]
            # Causes.
            for cause, count in other["deaths_by_cause"].items():
                merged["deaths_by_cause"][cause] = merged["deaths_by_cause"].get(cause, 0) + count
        # Recompute the derived numbers.
        counts = np.asarray(merged["action_counts"])
        # Entropy.
        p = counts / counts.sum() if counts.sum() else counts
        p = p[p > 0]
        merged["entropy"] = float(-(p * np.log(p)).sum()) if p.size else 0.0
        # Means.
        merged["mean_survival_ticks"] = float(np.mean(merged["survival_ticks"])) if merged["survival_ticks"] else 0.0
        merged["win_rate"] = float(np.mean(merged["wins"])) if merged["wins"] else 0.0
        merged["kill_rate"] = float(np.mean(merged["kills"])) if merged["kills"] else 0.0
        merged["mean_death_needs"] = (np.asarray(merged["death_needs"]) / max(1, merged["death_count"])).tolist()
        # Done.
        return merged
