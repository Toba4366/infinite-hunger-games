# `game.py`

**Source:** [hunger_games/game.py](../hunger_games/game.py)
**Depends on:** [actions.py](actions.md) (`Action`, `ActionType`), [arena.py](arena.md) (`Arena`), [brain/init.md](brain/init.md) (`Brain`, `create_brain`), `brain/neat.py` (`NeatBrain`, `NeatGenome`, imported inside `_make_brain` only when a NEAT roster genome is met), [config.py](config.md) (`SimulationConfig`), [districts.py](districts.md) (`SEXES`, `default_tribute_name`), [gamemaker.py](gamemaker.md) (`Gamemaker`), [perception.py](perception.md) (`Perception`), [player.py](player.md) (`Player`), [records.py](records.md) (`Elimination`, `EliminationMethod`, `GameResult`, `PlayerResult`), [resources.py](resources.md) (`ResourceKind`, `build_layout`, `weapon_name`), [scenario.py](scenario.md) (`Scenario`, `TributeSpec`), [sponsors.py](sponsors.md) (`SponsorGift`, `SponsorPool`), `numpy`, and `collections.abc.Callable`.
**Used by:** [runner.py](runner.md), [init.md](init.md), [main.md](main.md), [renderer.py](renderer.md), [recorder.py](recorder.md), `research/telemetry.py` (attaches hooks), [training/genetic.md](training/genetic.md), `training/reinforce.py` (attaches hooks), `training/imitation.py`, `ui/session.py`, and [tests/test_game.md](tests/test_game.md), [tests/test_scenario.md](tests/test_scenario.md), [tests/test_brains.md](tests/test_brains.md), [tests/test_recorder_training.md](tests/test_recorder_training.md), [tests/test_research.md](tests/test_research.md).

## Purpose

`Game` is the referee for one Hunger Games. It builds the arena, scatters supplies, creates and places the tributes, and then advances the clock one tick at a time. Each tick every living player senses, decides and acts, then time passes for everyone (bars drain, wounds bleed, the circle shrinks) and deaths are recorded. Once a day the sponsors send parachutes.

It never decides for a player and never draws anything. It enforces the rules and writes down what happened as the rows in [records.md](records.md).

Two things make it useful for research. A `brain_factory` lets a trainer supply its own brains. `decision_hooks` and `tick_hooks` let a collector watch every decision and every tick without changing the rules.

## Concepts you need

**Seeded generators.** `np.random.default_rng(seed)` gives a generator whose sequence depends only on the seed. One generator is shared by the arena, the layout, the brains and every dice roll, so the same seed replays the same game.

**Callables as data.** `BrainFactory = Callable[[int, np.random.Generator], Brain]` is a type alias meaning "a function that takes a player index and a generator and returns a Brain". Hooks are also plain functions stored in lists.

**Properties for derived state.** `alive_players`, `day_number`, `day_fraction` and `is_over` are computed each time they are read.

**Leading underscores.** `_make_brain`, `_resolve_action`, `_eliminate` and so on are internal. `step`, `run`, `result`, `field_knowledge` and `days_survived` are the public surface.

**Idempotent finish.** `_finish` uses a `finished` flag so it can be called from `step`, `run` and `result` without assigning placings twice.

**Two shapes of genome.** The voting and neural brains store their genome as a flat list of numbers. A NEAT brain stores its genome as a dictionary of node and connection genes, because its network shape is part of what evolves. `_make_brain` tells the two apart with `isinstance(genome, dict)`.

**Lazy imports.** `from hunger_games.brain.neat import NeatBrain, NeatGenome` sits inside `_make_brain` rather than at the top of the file. `brain/neat.py` imports things that import `game.py`, so a top-level import would be a cycle. The `# noqa: PLC0415` comment tells ruff this is on purpose.

## Walkthrough

### `BrainFactory`

```python
BrainFactory = Callable[[int, np.random.Generator], Brain]
```

The signature a custom brain factory must have.

### `Game.__init__`

```python
def __init__(
    self,
    config: SimulationConfig,
    game_id: int = 0,
    brain_factory: BrainFactory | None = None,
    scenario: Scenario | None = None,
) -> None
```

Setup, in order:

1. Seed: `config.seed` if set, else fresh entropy. `self.seed = (base_seed + game_id) % 2**31` so games in a batch differ but replay.
2. `self.rng = np.random.default_rng(self.seed)`.
3. If the scenario has a painted `terrain`, it becomes an `int8` array and `Arena` adopts it; otherwise the arena is generated.
4. `self.layout = build_layout(config.layout)`. Its supplies are applied unless the scenario says `use_layout_loot` is false. Hand-placed `scenario.loot` is added on walkable cells.
5. `self.players = self._create_players()` and `self.player_by_id` for `ATTACK` lookups.
6. `_place_players()` puts everyone on a podium.
7. `self.gamemaker = Gamemaker(config, self.arena)` and `self.sponsors = SponsorPool(config)`. Each player's starting `favor` is filled in.
8. Bookkeeping: `tick = 0`, `eliminations = []`, `gifts = []`, `last_elimination_tick = 0`, `death_ticks = {}`, `finished = False`.
9. `decision_hooks: list[Callable[[Player, Perception, Action], None]] = []` and `tick_hooks: list[Callable[[Game], None]] = []`.

### `Game._make_brain`

```python
def _make_brain(self, index: int, name: str, genome: list[float] | dict | None) -> Brain
```

Builds one tribute's brain. Three cases, checked in this order:

1. **A factory wins outright.** If a `brain_factory` was given, `self.brain_factory(index, self.rng)` is returned and `name` and `genome` are ignored. Trainers use this.
2. **A NEAT genome.** If `name == "neat"` and `genome` is a dictionary, `NeatBrain(NeatGenome.from_dict(genome), chaos=self.config.chaos)` is returned. `NeatGenome.from_dict` rebuilds the node and connection genes from the plain lists a champion file or a saved scenario holds. This branch never calls `create_brain`, so `config.neural` and `config.endgame_instinct` play no part.
3. **A named kind.** Otherwise `create_brain(name, config.chaos, self.rng, config.neural, config.endgame_instinct)` builds the named kind. The fifth argument is the endgame flag: `create_brain` passes it to `VotingBrain(endgame=...)` and ignores it for other brains. A roster genome, if present, is loaded with `brain.set_genome(np.asarray(genome, dtype=float))`.

The roster's `genome` field comes from `TributeSpec.genome` (see [scenario.md](scenario.md)). The dashboard fills it in with a flat list for voting and neural champions and with a dictionary for NEAT champions.

### `Game._start_value`

```python
def _start_value(self, minimum: float, override: float | None) -> float
```

A roster value wins (clipped to 0.01..1.0). Otherwise a uniform draw between `min(minimum, 1.0)` and 1.0.

### `Game._create_players`

```python
def _create_players(self) -> list[Player]
```

Uses `scenario.tributes` if there are any, else `_generated_spec(index)` for each of `config.num_players`. For each spec it calls `_make_brain(index, spec.brain_name, spec.genome)`, builds the `Player`, sets the three starting bars, and copies `weapon_quality`, `food`, `medicine` and `favor_bonus` from the spec.

### `Game._generated_spec`

```python
def _generated_spec(self, index: int) -> TributeSpec
```

District `(index // 2) % 12 + 1`, sex alternating F then M, training score `normal(6.5, 2.5)` rounded and clipped to 1..12, survival score `0.6 * score / 12 + 0.4 * random` clipped to 0.05..0.95, name from `default_tribute_name`, brain from `config.brain_name`. The `genome` field is left at `None`, so generated tributes always get a fresh brain.

### `Game._place_players`

```python
def _place_players(self) -> None
```

Asks the layout for podiums, orders players with `_spread_high_scorers`, and assigns positions in that order. Roster podiums then override, snapped onto a legal cell with `arena.snap_to_podium`.

### `Game._spread_high_scorers`

```python
@staticmethod
def _spread_high_scorers(players: list[Player]) -> list[Player]
```

Chapter 2's plan. Sort by training score. The top third (at least one) get podiums spaced `count / len(top)` apart. Everyone else fills the gaps in order. Tested by `test_strong_players_spread_apart`.

### `Game.alive_players`

```python
@property
def alive_players(self) -> list[Player]
```

Players whose `alive` flag is set.

### `Game.day_number`

```python
@property
def day_number(self) -> int
```

`tick // ticks_per_day + 1`, so the first day is day 1.

### `Game.day_fraction`

```python
@property
def day_fraction(self) -> float
```

`tick / ticks_per_game`, 0.0 to 1.0.

### `Game.is_over`

```python
@property
def is_over(self) -> bool
```

`len(alive_players) <= 1 or tick >= ticks_per_game`.

### `Game.field_knowledge`

```python
def field_knowledge(self, player: Player, alive: list[Player]) -> tuple[bool, float, float, float]
```

What the cannon and the nightly sky tell this tribute. Returns `(known, mean_score, strongest, rank)`:

- `cannon_and_sky` off: `(False, 0.0, 0.0, 0.5)`.
- Nobody else alive: `(True, 0.0, 0.0, 1.0)`.
- Otherwise `known` is `True`, `mean_score` is the mean training score of the other living tributes divided by 12, `strongest` is their maximum divided by 12, and `rank` is the fraction of them with a strictly lower score than this player.

`step` passes the result straight into `Player.perceive` as `field`.

### `Game.days_survived`

```python
def days_survived(self, player: Player) -> float
```

The death tick (or the current tick for the living) divided by `ticks_per_day`.

### `Game.step`

```python
def step(self) -> None
```

One tick. The exact order of events:

1. If `is_over`, call `_finish()` and return. Nothing else happens.
2. If `tick > 0` and `tick % ticks_per_day == 0` (the start of every day after the first), `sponsors.daily_gifts(players, rng, game_id, day_number, tick)` runs and its gifts are appended to `self.gifts`.
3. `alive = self.alive_players` is captured once. A shuffled copy is the acting order. `alive_fraction = len(alive) / len(players)` is fixed for the tick.
4. For each player in the shuffled order:
   1. Skip if already dead (someone earlier this tick may have killed them).
   2. `perception = player.perceive(arena, players, gamemaker.is_lethal(x, y), day_fraction, alive_fraction, vision_radius, landmark_radius, gamemaker.hazard_distance(x, y), gamemaker.shrinking, field_knowledge(player, alive))`.
   3. `action = player.decide(perception, rng)`.
   4. Every hook in `decision_hooks` is called as `hook(player, perception, action)`, before the action is carried out.
   5. `_resolve_action(player, action)`.
5. `_environment_tick()`: the game makers update, bars drain, hazard and natural deaths are recorded (details below).
6. `tick += 1`.
7. Every hook in `tick_hooks` is called as `hook(self)`. At this point the clock already shows the new tick.
8. If `is_over`, `_finish()` runs (placings, brain notifications).

So decision hooks see each decision at the moment it is made, and tick hooks see the world after everyone has acted and time has passed. `research/telemetry.py` and `training/reinforce.py` both rely on this order.

### `Game._resolve_action`

```python
def _resolve_action(self, player: Player, action: Action) -> None
```

Dispatches on `action.kind`: `MOVE` and `FLEE` both call `move`; `DRINK`, `EAT`, `HUNT`, `PICK_UP`, `HEAL`, `REST` call the matching body method; `ATTACK` goes to `_resolve_attack`.

### `Game._resolve_attack`

```python
def _resolve_attack(self, attacker: Player, target_id: int | None) -> None
```

The attack fizzles if the target is missing, dead or beyond `attacker.reach`. Otherwise `attacker.attack(target, rng, chaos)` runs. A loser at or below zero health is eliminated by the winner with `weapon_name(winner.weapon_quality)`. A winner at or below zero is also eliminated, credited to the loser.

### `Game._environment_tick`

```python
def _environment_tick(self) -> None
```

In order: capture `alive`; `gamemaker.update(tick, last_elimination_tick, len(alive))`; then for each living player `tick_needs`, then hazard damage of `Gamemaker.DAMAGE_PER_TICK` if outside the circle (an elimination as `GAMEMAKER` with `Gamemaker.WEAPON_LABEL` if that kills, skipping the natural check), then `natural_cause_of_death` (an elimination as `NATURAL`).

### `Game._eliminate`

```python
def _eliminate(self, player: Player, method: EliminationMethod, weapon: str, killer: Player | None = None) -> None
```

Ignores the already dead. Placement is the number alive at the moment of death (24 for the first out). Sets `alive`, `placement`, `cause_of_death`, `killer_id`; credits the killer; records `death_ticks`; appends an `Elimination` row; resets `last_elimination_tick`.

### `Game._finish`

```python
def _finish(self) -> None
```

Runs once. Every survivor gets placing `len(survivors)` (1 for a sole victor, 2 for a two-way draw). Then every brain hears `on_game_end(placement, kills, days_survived)`.

### `Game.run`

```python
def run(self) -> GameResult
```

Steps until `is_over`, calls `_finish`, returns `result()`.

### `Game.result`

```python
def result(self) -> GameResult
```

Calls `_finish` (safe to repeat), picks the victor if exactly one survivor, counts gifts per player, builds one `PlayerResult` per tribute, and returns a `GameResult` with `interventions` from the game maker, the eliminations list, the player rows and gifts as plain dictionaries (`vars(gift).copy()`).

## How to use it / experiment

Run one game and attach a hook:

```python
from hunger_games.config import SimulationConfig
from hunger_games.game import Game

game = Game(SimulationConfig(width=60, height=60, seed=11, max_days=10))
attacks = []
game.decision_hooks.append(lambda player, perception, action: attacks.append(action.kind.value == "attack"))
game.tick_hooks.append(lambda g: print(f"tick {g.tick}: {len(g.alive_players)} alive") if g.tick % 24 == 0 else None)
result = game.run()
print(result.winner_name, sum(attacks), "attack decisions")
```

Supply your own brains with a factory:

```python
from hunger_games.brain import create_brain

game = Game(config, brain_factory=lambda index, rng: create_brain("neural", 0.5, rng, config.neural))
```

Put a saved NEAT champion into one tribute through the roster. The champion file's `genome` is already the dictionary `_make_brain` expects:

```python
import json
from hunger_games.scenario import Scenario, TributeSpec

data = json.load(open("results/neat_run/champion.json"))
roster = [
    TributeSpec(0, "Learner", 1, "F", 8, 0.6, brain_name="neat", genome=data["genome"]),
    TributeSpec(1, "Rival", 1, "M", 8, 0.6, brain_name="voting"),
]
game = Game(SimulationConfig(width=60, height=60, seed=1), scenario=Scenario(tributes=roster))
print(game.players[0].brain.name, game.players[0].brain.describe())
```

To step manually (the renderer and dashboard do this), call `game.step()` in a loop and read `game.players`, `game.gamemaker.safe_radius` and `game.eliminations` between calls.

## Gotchas

- `alive` and `alive_fraction` are captured at the start of the tick. A tribute killed early in the tick still counts in the field statistics and the alive fraction until the next tick.
- Decision hooks fire before the action resolves. To see the effect of an action, use a tick hook and compare with what you saved in the decision hook.
- Tick hooks run after `tick += 1`, so `game.tick` inside a tick hook is one more than the tick the decisions were made on.
- `tick_hooks` receive the game after `_environment_tick` but before `_finish`, so brains have not yet been told the outcome when the last tick hook runs. Check `game.is_over` inside the hook if you need end-of-game handling, as telemetry does.
- `brain_factory` completely bypasses `config.brain_name`, roster brain names, roster genomes (flat or NEAT) and `endgame_instinct`.
- A roster entry with `brain_name="neat"` and a dictionary genome gets that genome; with `genome=None` it falls through to `create_brain("neat", ...)`, which builds a minimal random NEAT genome (inputs wired straight to the outputs), so the tribute plays but has no training behind it. A flat list is not a NEAT genome and would be rejected by `NeatBrain.set_genome` unless its length matches the connection count.
- The NEAT branch uses `config.chaos` for the brain's temperature and ignores `config.neural`. A NEAT genome carries its own shape, so there is no architecture to mismatch, but it still expects the 50-number perception vector it was trained on.
- A game that is over from the start (for example one player) still gets bookkeeping via `run` calling `_finish` explicitly.
- `field_knowledge` compares training scores strictly (`<`), so tributes with equal scores do not count as weaker.
