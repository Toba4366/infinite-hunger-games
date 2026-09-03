"""game.py - one complete Hunger Games, from the podiums to the victor.

`Game` is the referee. Each tick it lets every living player sense, decide
and act, then lets time pass (thirst, hunger, wounds, the game makers), then
checks who died. Once a day the sponsors send parachutes. It never decides
anything for a player and never draws anything; it only enforces the rules
and writes down what happened.

A game can be built two ways: purely from a `SimulationConfig` (everything
generated), or from a config plus a `Scenario` (a painted map, hand-placed
loot and an edited roster from the dashboard).
"""

# Type hint for the optional brain factory.
from collections.abc import Callable

# numpy for the random generator and painted terrain arrays.
import numpy as np

# The action vocabulary.
from hunger_games.actions import Action, ActionType

# The world.
from hunger_games.arena import Arena

# Brain construction by name.
from hunger_games.brain import Brain, create_brain

# The settings.
from hunger_games.config import SimulationConfig

# District names and sexes.
from hunger_games.districts import SEXES, default_tribute_name

# The interventions.
from hunger_games.gamemaker import Gamemaker

# What a tribute senses.
from hunger_games.perception import Perception

# The tributes.
from hunger_games.player import Player

# The spreadsheet rows.
from hunger_games.records import Elimination, EliminationMethod, GameResult, PlayerResult

# Supply layouts and weapon names.
from hunger_games.resources import ResourceKind, build_layout, weapon_name

# Custom setups.
from hunger_games.scenario import Scenario, TributeSpec

# Parachutes.
from hunger_games.sponsors import SponsorGift, SponsorPool

# A brain factory takes (player index, rng) and returns a Brain.
BrainFactory = Callable[[int, np.random.Generator], Brain]


class Game:
    """Runs a single Hunger Games under one `SimulationConfig` (and optional `Scenario`)."""

    def __init__(
        self,
        config: SimulationConfig,
        game_id: int = 0,
        brain_factory: BrainFactory | None = None,
        scenario: Scenario | None = None,
    ) -> None:
        """Build the arena, scatter supplies, create and place the players."""
        # Keep the settings.
        self.config = config
        # Which game this is in a batch (also used to vary the seed).
        self.game_id = game_id
        # The custom setup, if any.
        self.scenario = scenario
        # The base seed: the configured one, or a fresh random one.
        base_seed = config.seed if config.seed is not None else int(np.random.SeedSequence().entropy % (2**31))
        # Offset by the game id so every game in a batch differs but stays reproducible.
        self.seed = (base_seed + game_id) % (2**31)
        # The single random generator every part of this game draws from.
        self.rng = np.random.default_rng(self.seed)
        # A painted map, if the scenario has one.
        painted = (
            np.array(scenario.terrain, dtype=np.int8) if scenario is not None and scenario.terrain is not None else None
        )
        # Generate (or adopt) the world.
        self.arena = Arena(config, self.rng, terrain=painted)
        # Choose the supply layout.
        self.layout = build_layout(config.layout)
        # Scatter the layout's supplies unless the scenario says hand-placed only.
        if scenario is None or scenario.use_layout_loot:
            self.layout.apply(self.arena, self.rng)
        # Add any hand-placed loot on top.
        for loot in scenario.loot if scenario is not None else []:
            # Only inside the arena.
            if self.arena.is_walkable(loot.x, loot.y):
                self.arena.resources.place(loot.x, loot.y, ResourceKind(loot.kind), loot.quantity, loot.quality)
        # How to build each player's brain (default: by name from the config or the roster).
        self.brain_factory = brain_factory
        # Create the tributes.
        self.players = self._create_players()
        # A lookup so ATTACK actions can find their target quickly.
        self.player_by_id = {player.player_id: player for player in self.players}
        # Stand them on their podiums.
        self._place_players()
        # The head game maker.
        self.gamemaker = Gamemaker(config, self.arena)
        # The sponsors.
        self.sponsors = SponsorPool(config)
        # Give every tribute their starting favour so the dashboard can show it.
        for player in self.players:
            player.favor = self.sponsors.favor(player)
        # The clock.
        self.tick = 0
        # Every elimination so far, in order.
        self.eliminations: list[Elimination] = []
        # Every parachute so far, in order.
        self.gifts: list[SponsorGift] = []
        # When the last elimination happened (the game makers watch this).
        self.last_elimination_tick = 0
        # The tick each dead player died on, for "days survived".
        self.death_ticks: dict[int, int] = {}
        # Whether end-of-game bookkeeping has already run.
        self.finished = False
        # Research hooks: called as hook(player, perception, action) after every decision.
        self.decision_hooks: list[Callable[[Player, Perception, Action], None]] = []
        # Research hooks: called as hook(game) at the end of every tick.
        self.tick_hooks: list[Callable[[Game], None]] = []

    # ------------------------------------------------------------- setup

    def _make_brain(self, index: int, name: str, genome: list[float] | dict | None) -> Brain:
        """Build one brain: the factory wins, otherwise the named kind with an optional genome."""
        # A factory (used by trainers) overrides everything.
        if self.brain_factory is not None:
            return self.brain_factory(index, self.rng)
        # A NEAT genome is a dictionary of nodes and connections, not a flat vector.
        if name == "neat" and isinstance(genome, dict):
            from hunger_games.brain.neat import NeatBrain, NeatGenome  # noqa: PLC0415 - avoid an import cycle

            return NeatBrain(NeatGenome.from_dict(genome), chaos=self.config.chaos)
        # Build the named kind with the config's neural architecture and instinct toggles.
        brain = create_brain(name, self.config.chaos, self.rng, self.config.neural, self.config.endgame_instinct)
        # Load a saved genome if the roster has one.
        if genome is not None:
            brain.set_genome(np.asarray(genome, dtype=float))
        # Done.
        return brain

    def _start_value(self, minimum: float, override: float | None) -> float:
        """A starting bar: the roster's exact value if given, else random between the minimum and full."""
        # The roster wins.
        if override is not None:
            return float(np.clip(override, 0.01, 1.0))
        # Otherwise draw between the configured minimum and 1.0.
        return float(self.rng.uniform(min(minimum, 1.0), 1.0))

    def _create_players(self) -> list[Player]:
        """Create the tributes from the roster if there is one, else generate them."""
        # Collect them here.
        players = []
        # Use the roster or make a generated spec for each slot.
        specs = (
            self.scenario.tributes
            if self.scenario is not None and self.scenario.tributes
            else [self._generated_spec(index) for index in range(self.config.num_players)]
        )
        # Build a player from each spec.
        for index, spec in enumerate(specs):
            # The brain.
            brain = self._make_brain(index, spec.brain_name, spec.genome)
            # The body.
            player = Player(
                spec.player_id, spec.name, spec.district, spec.training_score, spec.survival_score, brain, spec.sex
            )
            # Starting bars.
            player.thirst = self._start_value(self.config.start_thirst_min, spec.start_thirst)
            # Hunger.
            player.hunger = self._start_value(self.config.start_hunger_min, spec.start_hunger)
            # Health.
            player.health = self._start_value(self.config.start_health_min, spec.start_health)
            # Gifts from the game maker.
            player.weapon_quality = spec.weapon_quality
            # Rations.
            player.food = spec.food
            # Medkits.
            player.medicine = spec.medicine
            # Sponsor favour bonus.
            player.favor_bonus = spec.favor_bonus
            # Add to the roster.
            players.append(player)
        # The full roster.
        return players

    def _generated_spec(self, index: int) -> TributeSpec:
        """Roll a default tribute: district pairs, alternating sexes, film-like scores."""
        # Two tributes per district, cycling through 1..12.
        district = (index // 2) % 12 + 1
        # First of each pair is female, second is male.
        sex = SEXES[index % 2]
        # Training scores cluster around 6-7 like the films, clamped to 1..12.
        training_score = int(np.clip(round(self.rng.normal(6.5, 2.5)), 1, 12))
        # Survival aptitude: mostly from the score, partly luck, never quite 0 or 1.
        survival_score = float(np.clip(0.6 * training_score / 12.0 + 0.4 * self.rng.random(), 0.05, 0.95))
        # Package it up.
        return TributeSpec(
            player_id=index,
            name=default_tribute_name(district, sex),
            district=district,
            sex=sex,
            training_score=training_score,
            survival_score=survival_score,
            brain_name=self.config.brain_name,
        )

    def _place_players(self) -> None:
        """Put each player on a podium: the layout's podiums, overridden by any roster podiums."""
        # The layout knows where its podiums are.
        podiums = self.layout.spawn_positions(self.arena, len(self.players))
        # Decide who stands on which podium (chapter 2's spreading plan).
        ordered = self._spread_high_scorers(self.players)
        # Assign positions in that order.
        for player, (x, y) in zip(ordered, podiums, strict=False):
            # Column.
            player.x = x
            # Row.
            player.y = y
        # Roster podiums win over the layout's.
        if self.scenario is not None and self.scenario.tributes:
            # Check each spec.
            for spec in self.scenario.tributes:
                # Only specs with a podium.
                if spec.podium is not None and spec.player_id in self.player_by_id:
                    # Snap it onto a legal cell in case the map changed.
                    x, y = self.arena.snap_to_podium(int(spec.podium[0]), int(spec.podium[1]))
                    # Place.
                    self.player_by_id[spec.player_id].x = x
                    # Row.
                    self.player_by_id[spec.player_id].y = y

    @staticmethod
    def _spread_high_scorers(players: list[Player]) -> list[Player]:
        """Chapter 2's podium plan: the top third of scorers get evenly spaced
        podiums so they avoid each other early; everyone else fills the gaps.
        """
        # Strongest first.
        ranked = sorted(players, key=lambda player: player.training_score, reverse=True)
        # How many podiums there are.
        count = len(ranked)
        # Empty podium list.
        slots: list[Player | None] = [None] * count
        # The top third (at least one player).
        top = ranked[: max(1, count // 3)]
        # Everyone else.
        rest = ranked[len(top) :]
        # Distance between the top scorers' podiums.
        stride = count / len(top)
        # Space the top scorers out evenly.
        for index, player in enumerate(top):
            # Their podium number.
            slots[int(index * stride)] = player
        # An iterator over the remaining players.
        remaining = iter(rest)
        # Fill every empty podium in order.
        for index in range(count):
            # Only empty podiums get filled.
            if slots[index] is None:
                slots[index] = next(remaining)
        # The final podium order (no Nones remain).
        return [player for player in slots if player is not None]

    # ------------------------------------------------------------ state

    @property
    def alive_players(self) -> list[Player]:
        """Everyone still in the games."""
        # Filter on the alive flag.
        return [player for player in self.players if player.alive]

    @property
    def day_number(self) -> int:
        """The current in-game day, starting from 1."""
        # Integer-divide ticks by ticks-per-day and add one.
        return self.tick // self.config.ticks_per_day + 1

    @property
    def day_fraction(self) -> float:
        """How far through the maximum game length we are (0.0 to 1.0)."""
        # Ticks so far over the tick budget.
        return self.tick / self.config.ticks_per_game

    @property
    def is_over(self) -> bool:
        """One (or zero) players left, or the time limit reached."""
        # Either condition ends the games.
        return len(self.alive_players) <= 1 or self.tick >= self.config.ticks_per_game

    def field_knowledge(self, player: Player, alive: list[Player]) -> tuple[bool, float, float, float]:
        """What the cannon and the nightly sky tell this tribute about the rest
        of the field: whether it is known, the mean and maximum training
        score of the other living tributes (scaled 0..1), and the fraction of
        them this tribute outranks.
        """
        # With the cannon and sky switched off, the field is a mystery.
        if not self.config.cannon_and_sky:
            return False, 0.0, 0.0, 0.5
        # The other living tributes' scores.
        others = [p.training_score for p in alive if p is not player]
        # Alone: nothing to compare against.
        if not others:
            return True, 0.0, 0.0, 1.0
        # Mean, max, and my rank among them.
        mean_score = sum(others) / len(others) / 12.0
        # The strongest.
        strongest = max(others) / 12.0
        # Fraction weaker than me.
        rank = sum(1 for score in others if score < player.training_score) / len(others)
        # Done.
        return True, mean_score, strongest, rank

    def days_survived(self, player: Player) -> float:
        """How many days a player lasted (so far, if still alive)."""
        # Dead players stopped at their death tick; the living are still going.
        last_tick = self.death_ticks.get(player.player_id, self.tick)
        # Convert ticks to days.
        return last_tick / self.config.ticks_per_day

    # ------------------------------------------------------------ ticking

    def step(self) -> None:
        """Advance the games by one tick."""
        # Nothing happens after the end.
        if self.is_over:
            self._finish()
            return
        # At the start of each new day (not the first), the sponsors send parachutes.
        if self.tick > 0 and self.tick % self.config.ticks_per_day == 0:
            self.gifts.extend(
                self.sponsors.daily_gifts(self.players, self.rng, self.game_id, self.day_number, self.tick)
            )
        # Who is playing this tick.
        alive = self.alive_players
        # Act in a random order so nobody always goes first.
        order = list(alive)
        # Shuffle in place.
        self.rng.shuffle(order)
        # Fraction of tributes still alive, for the perception.
        alive_fraction = len(alive) / max(1, len(self.players))
        # Each living player senses, decides and acts.
        for player in order:
            # Someone earlier in the order may have killed this player already.
            if not player.alive:
                continue
            # Gather the senses.
            perception = player.perceive(
                self.arena,
                self.players,
                self.gamemaker.is_lethal(player.x, player.y),
                self.day_fraction,
                alive_fraction,
                self.config.vision_radius,
                self.config.landmark_radius,
                self.gamemaker.hazard_distance(player.x, player.y),
                self.gamemaker.shrinking,
                self.field_knowledge(player, alive),
            )
            # Ask the brain.
            action = player.decide(perception, self.rng)
            # Let any research hooks see the decision before it is carried out.
            for hook in self.decision_hooks:
                hook(player, perception, action)
            # Carry out the action.
            self._resolve_action(player, action)
        # Time passes for everyone.
        self._environment_tick()
        # Advance the clock.
        self.tick += 1
        # Let any research hooks see the end of the tick.
        for hook in self.tick_hooks:
            hook(self)
        # If that ended the games, do the bookkeeping.
        if self.is_over:
            self._finish()

    def _resolve_action(self, player: Player, action: Action) -> None:
        """Apply one player's chosen action to the world."""
        # Movement and fleeing are both just a step.
        if action.kind in (ActionType.MOVE, ActionType.FLEE):
            player.move(action.dx, action.dy, self.arena, self.rng)
        # Drinking.
        elif action.kind is ActionType.DRINK:
            player.drink(self.arena)
        # Eating.
        elif action.kind is ActionType.EAT:
            player.eat()
        # Hunting.
        elif action.kind is ActionType.HUNT:
            player.hunt(self.arena, self.rng, self.config.chaos)
        # Looting.
        elif action.kind is ActionType.PICK_UP:
            player.pick_up(self.arena)
        # Using a medkit.
        elif action.kind is ActionType.HEAL:
            player.heal()
        # Resting.
        elif action.kind is ActionType.REST:
            player.rest()
        # Fighting.
        elif action.kind is ActionType.ATTACK:
            self._resolve_attack(player, action.target_id)

    def _resolve_attack(self, attacker: Player, target_id: int | None) -> None:
        """Run a fight if the target is real, alive and within reach."""
        # Look the target up.
        target = self.player_by_id.get(target_id) if target_id is not None else None
        # No such player, or they are dead, or out of reach: the attack fizzles.
        if target is None or not target.alive or attacker.distance_to(target) > attacker.reach:
            return
        # Fight.
        outcome = attacker.attack(target, self.rng, self.config.chaos)
        # A dead loser is eliminated by the winner's weapon.
        if outcome.loser.health <= 0.0:
            self._eliminate(
                outcome.loser, EliminationMethod.PLAYER, weapon_name(outcome.winner.weapon_quality), outcome.winner
            )
        # A winner who was already nearly dead can also fall.
        if outcome.winner.health <= 0.0:
            self._eliminate(
                outcome.winner, EliminationMethod.PLAYER, weapon_name(outcome.loser.weapon_quality), outcome.loser
            )

    def _environment_tick(self) -> None:
        """Needs drain, wounds bleed, the game makers act, and natural deaths are checked."""
        # Who is alive going into this phase.
        alive = self.alive_players
        # Let the game makers decide whether to shrink the safe circle.
        self.gamemaker.update(self.tick, self.last_elimination_tick, len(alive))
        # Apply time to each living player.
        for player in alive:
            # Needs drain and wounds bleed.
            player.tick_needs(self.config.thirst_per_tick, self.config.hunger_per_tick)
            # Outside the safe circle: take hazard damage.
            if self.gamemaker.is_lethal(player.x, player.y):
                # Lose health.
                player.health -= Gamemaker.DAMAGE_PER_TICK
                # Dead from the hazard.
                if player.health <= 0.0:
                    self._eliminate(player, EliminationMethod.GAMEMAKER, Gamemaker.WEAPON_LABEL)
                    continue
            # Check thirst, hunger and bleeding.
            cause = player.natural_cause_of_death()
            # Dead from natural causes.
            if cause is not None:
                self._eliminate(player, EliminationMethod.NATURAL, cause)

    def _eliminate(self, player: Player, method: EliminationMethod, weapon: str, killer: Player | None = None) -> None:
        """Remove a player from the games and record the elimination."""
        # Never eliminate anyone twice.
        if not player.alive:
            return
        # Placement is how many were alive at the moment of death (24 = first out).
        placement = len(self.alive_players)
        # Mark them dead.
        player.alive = False
        # Record their placing.
        player.placement = placement
        # Record the cause.
        player.cause_of_death = weapon
        # Record the killer, if any.
        player.killer_id = killer.player_id if killer is not None else None
        # Credit the killer.
        if killer is not None:
            killer.kills += 1
        # Remember when they died.
        self.death_ticks[player.player_id] = self.tick
        # Write the spreadsheet row.
        self.eliminations.append(
            Elimination(
                game_id=self.game_id,
                day=self.day_number,
                tick=self.tick,
                victim_id=player.player_id,
                victim_name=player.name,
                victim_district=player.district,
                victim_training_score=player.training_score,
                method=method.value,
                weapon=weapon,
                killer_id=killer.player_id if killer is not None else None,
                killer_name=killer.name if killer is not None else None,
                x=player.x,
                y=player.y,
                placement=placement,
            )
        )
        # Reset the game makers' quiet clock.
        self.last_elimination_tick = self.tick

    # ------------------------------------------------------------ ending

    def _finish(self) -> None:
        """Assign final placings to survivors and notify the brains (once)."""
        # Only run once.
        if self.finished:
            return
        # Mark as done.
        self.finished = True
        # Survivors share the remaining placing (1 for a sole victor).
        survivors = self.alive_players
        # Assign it.
        for player in survivors:
            player.placement = len(survivors)
        # Tell every brain how its player did (a learning hook).
        for player in self.players:
            player.brain.on_game_end(player.placement or 0, player.kills, self.days_survived(player))

    def run(self) -> GameResult:
        """Play the whole game to the end and return the results."""
        # Step until done.
        while not self.is_over:
            self.step()
        # Make sure the bookkeeping ran even if the game was over from the start.
        self._finish()
        # Package the results.
        return self.result()

    def result(self) -> GameResult:
        """Package everything worth keeping into a `GameResult`."""
        # Ensure placings are assigned.
        self._finish()
        # The survivors.
        survivors = self.alive_players
        # A sole survivor is the victor; otherwise it was a draw or a wipe-out.
        winner = survivors[0] if len(survivors) == 1 else None
        # How many parachutes each tribute received.
        gift_counts = {player.player_id: 0 for player in self.players}
        # Count them.
        for gift in self.gifts:
            gift_counts[gift.player_id] = gift_counts.get(gift.player_id, 0) + 1
        # One row per player.
        player_rows = [
            PlayerResult(
                game_id=self.game_id,
                player_id=player.player_id,
                name=player.name,
                district=player.district,
                sex=player.sex,
                training_score=player.training_score,
                survival_score=player.survival_score,
                brain=player.brain.name,
                favor=player.favor,
                gifts_received=gift_counts[player.player_id],
                placement=player.placement or 0,
                kills=player.kills,
                days_survived=self.days_survived(player),
                cause_of_death=player.cause_of_death,
                alive_at_end=player.alive,
            )
            for player in self.players
        ]
        # Bundle it all up.
        return GameResult(
            game_id=self.game_id,
            seed=self.seed,
            days=self.day_number,
            ticks=self.tick,
            winner_id=winner.player_id if winner else None,
            winner_name=winner.name if winner else None,
            interventions=self.gamemaker.interventions,
            eliminations=list(self.eliminations),
            players=player_rows,
            gifts=[vars(gift).copy() for gift in self.gifts],
        )
