# `player.py`

**Source:** [hunger_games/player.py](../hunger_games/player.py)
**Depends on:** [actions.py](actions.md) (`Action`), [arena.py](arena.md) (`Arena`, `sign`), [brain/base.md](brain/base.md) (`Brain`), [perception.py](perception.md) (`NearbyPlayer`, `Perception`), [resources.py](resources.md) (`ResourceKind`, `weapon_reach`), `numpy`, and `dataclasses`.
**Used by:** [game.py](game.md) (creates, places, and drives every player), [sponsors.py](sponsors.md) (type hints only), `research/telemetry.py` (reads bars, position, weapon and `Player.WOUND_THRESHOLD`), `training/reinforce.py` (reads bars and kills for rewards), [tests/test_scenario.md](tests/test_scenario.md) (`rest`, `tick_needs`) and [tests/test_brains.md](tests/test_brains.md) (`perceive`).

## Purpose

A `Player` is a tribute's body. Chapter 4 splits a tribute into a body ("their ability to perform actions: moving, eating, fighting") and a brain that decides. This file is the body. Every method is one physical thing the body can do, written as a small piece of arithmetic. The body never chooses: `perceive` gathers the senses, `decide` hands them to the brain, and `Game` calls the matching body method for whatever action came back.

## Concepts you need

**Bars from 0.0 to 1.0.** Thirst, hunger and health are floats. 1.0 is full, 0.0 is dead. `min(1.0, ...)` caps a gain and `max(0.0, ...)` floors a loss.

**Class constants.** `DRINK_AMOUNT` and friends live on the class, not on each instance. Read them as `Player.DRINK_AMOUNT` or `self.DRINK_AMOUNT`. Other modules borrow them (telemetry uses `WOUND_THRESHOLD`).

**Forward references.** `winner: "Player"` is a string because `Player` is not defined yet when `FightOutcome` is created. Python treats the string as the type name.

**Seeded randomness.** Every random draw uses the `np.random.Generator` the game passes in. `chaos * rng.normal(0.0, 0.15)` is zero at chaos 0 and a wide wobble at chaos 1.

**Chebyshev distance.** `max(abs(dx), abs(dy))`: king moves. Reach, vision and fights all use it.

**numpy windows.** `_scan_resources` slices the supply grid instead of looping over cells. `np.nonzero` gives the coordinates of every non-empty cell in the slice.

## Walkthrough

### `FightOutcome`

```python
@dataclass
class FightOutcome:
    winner: "Player"
    loser: "Player"
    damage: float
```

Returned by `attack`. `damage` is how much health the loser lost. `Game._resolve_attack` checks both players' health afterwards, since the winner also gets scratched.

### `Player` class constants

| Constant | Value | Meaning |
| --- | --- | --- |
| `DRINK_AMOUNT` | `0.5` | Thirst restored by one DRINK. |
| `EAT_AMOUNT` | `0.35` | Hunger restored by one ration. |
| `HEAL_AMOUNT` | `0.40` | Health restored by one medkit. |
| `REST_AMOUNT` | `0.02` | Health restored by one tick of REST. |
| `HUNT_STEP` | `0.1` | Each 0.1 of survival score above the difficulty earns one more ration. |
| `BASE_DAMAGE` | `0.35` | Damage a fight always inflicts on the loser. |
| `WEAPON_DAMAGE` | `0.45` | Extra damage per point of the winner's weapon quality. |
| `WOUND_THRESHOLD` | `0.5` | Below this health a wound bleeds and will not close by resting. |
| `BLEED_PER_TICK` | `0.004` | Health lost per tick from a serious wound (about five days from 0.5 to death). |

### `Player.__init__`

```python
def __init__(
    self,
    player_id: int,
    name: str,
    district: int,
    training_score: int,
    survival_score: float,
    brain: Brain,
    sex: str = "F",
) -> None
```

Creates a healthy, empty-handed tribute standing at `(0, 0)`. Attributes set:

| Attribute | Start | Meaning |
| --- | --- | --- |
| `player_id`, `name`, `district`, `sex` | from arguments | Identity. `sex` is `"F"` or `"M"`. |
| `training_score` | argument | The 1..12 score from training. |
| `survival_score` | argument | Fixed 0..1 aptitude for finding food. |
| `brain` | argument | The decision-maker. |
| `favor_bonus` | `0.0` | Extra sponsor favour from the game maker. |
| `favor` | `0.0` | Sponsors' current opinion, updated by `SponsorPool`. |
| `x`, `y` | `0` | Position. `Game._place_players` moves them to a podium. |
| `thirst`, `hunger`, `health` | `1.0` | The bars. |
| `food`, `medicine` | `0` | Pack contents. |
| `weapon_quality` | `0.0` | Fists. |
| `kills` | `0` | Eliminations credited. |
| `alive` | `True` | Still in the games. |
| `placement` | `None` | Final placing, 1 for the victor. |
| `cause_of_death` | `None` | Text, filled by `Game._eliminate`. |
| `killer_id` | `None` | Who killed them, if a player did. |
| `last_action` | `None` | The most recent action returned by the brain. |
| `last_perception` | `None` | The most recent perception the brain saw. |

`last_perception` exists so the dashboard's network visualiser can feed the same inputs through the brain again and draw the activations.

### `Player.threat_level`

```python
@property
def threat_level(self) -> float
```

`0.4 * survival_score + 0.4 * weapon_quality + 0.2 * health`. This is what other tributes see as `NearbyPlayer.threat`, and what `attack` rolls against.

### `Player.reach`

```python
@property
def reach(self) -> int
```

`weapon_reach(self.weapon_quality)`: 1 for fists, up to 3 for a bow. `Game._resolve_attack` refuses attacks beyond it.

### `Player.position`

```python
@property
def position(self) -> tuple[int, int]
```

`(x, y)` as one tuple.

### `Player.distance_to`

```python
def distance_to(self, other: "Player") -> int
```

Chebyshev distance to another player.

### `Player.perceive`

```python
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
) -> Perception
```

Gathers everything into a `Perception`. Step by step:

1. `landmark_radius` defaults to `vision_radius * 3`. Lakes and meadows can be spotted further away than people.
2. Water and grass distances come from the arena's precomputed fields. The direction is only given if the distance is within the landmark radius; otherwise `(0, 0)`.
3. `arena.resources.peek(x, y)` reports what is in my cell.
4. `_scan_resources` finds the nearest visible supply.
5. Every other living player within `vision_radius` becomes a `NearbyPlayer`. The list is sorted nearest first.
6. `field` is unpacked into `field_known`, `field_strength`, `strongest_remaining`, `my_rank`. If `None`, the defaults `(False, 0.0, 0.0, 0.5)` are used. `Game.step` passes `Game.field_knowledge(player, alive)` here.
7. The `Perception` is built. `training_score` is divided by 12 to land in 0..1. `safe_direction` is always the direction to the centre.

The test suite calls it with only the six required arguments, which is fine for a perception that ignores the game makers and the field.

### `Player._scan_resources`

```python
def _scan_resources(self, arena: Arena, vision_radius: int) -> tuple[tuple[int, int], float, ResourceKind]
```

Slices `arena.resources.kind[y0:y1, x0:x1]` around me, clipped to the grid. `np.nonzero` lists every non-empty cell. If none, returns `((0, 0), inf, ResourceKind.NONE)`. Otherwise computes the king-move distance to each, takes the `argmin`, and returns a one-step direction (via `sign`), the distance and the kind.

### `Player.decide`

```python
def decide(self, perception: Perception, rng: np.random.Generator) -> Action
```

Calls `self.brain.decide(perception, rng)`, stores the result in `last_action` and the input in `last_perception`, and returns the action.

### `Player.move`

```python
def move(self, dx: int, dy: int, arena: Arena, rng: np.random.Generator) -> bool
```

Refuses to step onto a non-walkable cell. Then rolls against `arena.move_success_at(nx, ny)`: water and rock are slow, so the step sometimes fails. This is how chases end. Returns whether the position changed.

### `Player.drink`

```python
def drink(self, arena: Arena) -> bool
```

Only works standing in water. Adds `DRINK_AMOUNT`, capped at 1.0.

### `Player.eat`

```python
def eat(self) -> bool
```

Uses one ration if any, adds `EAT_AMOUNT` to hunger.

### `Player.hunt`

```python
def hunt(self, arena: Arena, rng: np.random.Generator, chaos: float) -> int
```

Chapter 4's rule. `roll = survival_score + chaos * normal(0, 0.15)`. Below the terrain's difficulty: nothing. Otherwise `1 + int((roll - difficulty) / HUNT_STEP)` rations are added to the pack and returned.

```python
import numpy as np
from hunger_games.config import SimulationConfig
from hunger_games.game import Game

game = Game(SimulationConfig(width=60, height=60, seed=1))
player = game.players[0]
print(player.hunt(game.arena, np.random.default_rng(0), chaos=0.0), player.food)
```

### `Player.pick_up`

```python
def pick_up(self, arena: Arena) -> ResourceKind
```

Takes whatever is in my cell out of the arena. Food and medicine go into the pack. A weapon is kept only if it beats the current one (`max`). Returns the kind found, possibly `NONE`.

### `Player.heal`

```python
def heal(self) -> bool
```

Uses one medkit, adds `HEAL_AMOUNT`.

### `Player.rest`

```python
def rest(self) -> None
```

Adds `REST_AMOUNT` to health, but only if health is at or above `WOUND_THRESHOLD`. A deep wound needs medicine.

### `Player.attack`

```python
def attack(self, other: "Player", rng: np.random.Generator, chaos: float) -> FightOutcome
```

Each side rolls `threat_level + chaos * normal(0, 0.2)`. Ties go to the attacker. The loser loses `BASE_DAMAGE + WEAPON_DAMAGE * winner.weapon_quality`. The winner loses `0.05 + 0.10 * loser.weapon_quality`. Neither health is clamped here; `Game._resolve_attack` checks for `<= 0.0`.

### `Player.tick_needs`

```python
def tick_needs(self, thirst_per_tick: float, hunger_per_tick: float) -> None
```

Drains both bars (floored at 0.0). If `0.0 < health < WOUND_THRESHOLD`, the wound bleeds `BLEED_PER_TICK`.

### `Player.natural_cause_of_death`

```python
def natural_cause_of_death(self) -> str | None
```

Checks thirst, then hunger, then health, returning `"dehydration"`, `"starvation"`, `"untreated wound"` or `None`. Order matters: a tribute with two bars at zero is recorded under the first.

## How to use it / experiment

You rarely build a `Player` by hand; `Game._create_players` does it from a `TributeSpec`. To study one body in isolation:

```python
import numpy as np
from hunger_games.brain import create_brain
from hunger_games.player import Player

rng = np.random.default_rng(0)
a = Player(0, "A", 1, 10, 0.8, create_brain("random", 0.0, rng))
b = Player(1, "B", 2, 3, 0.2, create_brain("random", 0.0, rng))
b.x = 1
outcome = a.attack(b, rng, chaos=0.0)
print(outcome.winner.name, round(outcome.damage, 2), round(b.health, 2))
```

Ideas: change `BLEED_PER_TICK` to make wounds more or less deadly, or `WEAPON_DAMAGE` to make the cornucopia weapons matter more. Since `Game` reads `Gamemaker.DAMAGE_PER_TICK` and `Player` constants from the classes, you can also monkeypatch them in an experiment script without editing the package.

## Gotchas

- `attack` can push health below zero. Only `Game` turns that into an elimination. If you call it yourself, check `health <= 0.0` on both sides.
- `move` fails silently on water and rock some of the time. A brain that expects every step to succeed will be surprised.
- `perceive` does not compute the field itself. Without the `field` argument, `field_known` is `False` even when `cannon_and_sky` is on. Only `Game.step` supplies it.
- `last_perception` and `last_action` are `None` until the first `decide` call, so guard against that when inspecting a fresh game.
- `pick_up` discards a weaker weapon rather than carrying two; there is no inventory of weapons.
- `rest` does nothing below half health, and `tick_needs` bleeds there, so a badly wounded tribute with no medkit is slowly dying whatever it does.
