# `actions.py`

**Source:** [hunger_games/actions.py](../hunger_games/actions.py)
**Depends on:** Python standard library only (`dataclasses`, `enum`). No project modules, no third-party libraries.
**Used by:** [game.md](game.md) (`game.py`), [player.md](player.md) (`player.py`), [brain/base.md](brain/base.md), [brain/voting.md](brain/voting.md), [brain/neural.md](brain/neural.md), [brain/random_brain.md](brain/random_brain.md), and `tests/test_brains.py`.

## Purpose

This file is the shared vocabulary between a player's brain and a player's body. The video's chapter 4 splits every tribute into a *brain* (what they decide) and a *body* (what they can physically do). A brain looks at a `Perception` and hands back one `Action`. The body in [player.md](player.md) and the referee in [game.md](game.md) then carry that action out. Nothing in this file *does* anything; it only names the choices.

Keeping the list in one tiny module matters for a reason the file's docstring spells out: every brain must speak the same language. The hand-written voting brain, the dice-rolling random brain and the numpy neural network all return `Action` objects built from the same nine `ActionType` values. If you later train a neural network or evolve a genome, its output still has to be turned into one of these actions, and nothing else in the simulator needs to change.

In the pipeline this file sits after `resources.py` and `arena.py` and before `perception.py`. It has zero project imports, so it can never cause a circular import, and it is safe to import from anywhere.

## Concepts you need

**Enum.** `class ActionType(Enum)` creates a fixed set of named constants. `ActionType.MOVE` is a member, `ActionType.MOVE.value` is the string `"move"`, and `ActionType("move")` looks a member up by its value. Members compare by identity, so `kind is ActionType.MOVE` is the idiomatic test.

**Frozen dataclass.** `@dataclass(frozen=True)` generates `__init__`, `__eq__` and `__repr__` for you and then makes every instance read-only. Trying `action.dx = 5` raises `FrozenInstanceError`. Because frozen dataclasses also get a `__hash__`, two actions with the same field values are equal *and* hash the same. That is what lets the voting brain use actions as dictionary keys: `votes[Action.move(1, 0)] += 3` adds to the same bucket every time.

**Static methods.** `@staticmethod` marks a function that lives inside a class for tidiness but does not receive `self`. `Action.move(1, 0)` is just a nicer way to spell `Action(ActionType.MOVE, dx=1, dy=0)`.

**Optional types.** `target_id: int | None = None` means the field holds an integer or nothing. Only `ATTACK` actions fill it in.

## Walkthrough

### `class ActionType(Enum)`

The nine kinds of thing a body can do in one tick.

| Member | Value | Meaning |
| --- | --- | --- |
| `REST` | `"rest"` | Do nothing except recover a sliver of health. |
| `MOVE` | `"move"` | Step one cell in a direction. |
| `DRINK` | `"drink"` | Drink from the water you are standing in. |
| `EAT` | `"eat"` | Eat one ration from your pack. |
| `HUNT` | `"hunt"` | Try to catch food on the terrain you stand on. |
| `PICK_UP` | `"pick_up"` | Take whatever supplies are in your cell. |
| `HEAL` | `"heal"` | Use one medkit from your pack. |
| `ATTACK` | `"attack"` | Fight a player within reach. |
| `FLEE` | `"flee"` | Step away from a threat. Mechanically identical to `MOVE`. |

`FLEE` exists so a learning brain can express *why* it moved. The referee treats `MOVE` and `FLEE` the same way (both call `Player.move`). A reward function could give different credit to the two.

```python
from hunger_games.actions import ActionType

print(ActionType.DRINK.value)      # "drink"
print(ActionType("hunt"))          # ActionType.HUNT
print(list(ActionType)[0])         # ActionType.REST
```

### `class Action` (frozen dataclass)

Signature of the generated constructor:

```python
Action(kind: ActionType, dx: int = 0, dy: int = 0, target_id: int | None = None)
```

| Field | Type | Default | Used by |
| --- | --- | --- | --- |
| `kind` | `ActionType` | required | every action |
| `dx` | `int` | `0` | `MOVE`, `FLEE` (horizontal step, -1, 0 or +1) |
| `dy` | `int` | `0` | `MOVE`, `FLEE` (vertical step, -1, 0 or +1) |
| `target_id` | `int \| None` | `None` | `ATTACK` (the victim's `player_id`) |

The convention for `dy` follows numpy grids: negative `dy` is *up* (a smaller row number). Simple actions like `REST` leave every extra field at its default, so `Action(ActionType.REST) == Action(ActionType.REST)` is always `True`.

#### `Action.move(dx: int, dy: int) -> Action`

Static shortcut. Returns `Action(ActionType.MOVE, dx=dx, dy=dy)`.

#### `Action.flee(dx: int, dy: int) -> Action`

Static shortcut. Returns `Action(ActionType.FLEE, dx=dx, dy=dy)`.

#### `Action.attack(target_id: int) -> Action`

Static shortcut. Returns `Action(ActionType.ATTACK, target_id=target_id)`.

```python
from hunger_games.actions import Action, ActionType

step = Action.move(1, 0)          # one cell to the right
run = Action.flee(-1, -1)         # up and to the left
hit = Action.attack(7)            # fight player 7
rest = Action(ActionType.REST)

print(step)                       # Action(kind=<ActionType.MOVE: 'move'>, dx=1, dy=0, target_id=None)
print(step == Action.move(1, 0))  # True: value equality
print({step: 3.0}[Action.move(1, 0)])  # 3.0: usable as a dict key
```

### `DIRECTIONS`

```python
DIRECTIONS = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
```

The eight compass directions as `(dx, dy)` tuples, in a fixed reading order: the top row left to right, then the middle row (skipping the centre), then the bottom row. The order is part of the contract. The neural brain's output layer has one neuron per direction and maps neuron `i` to `DIRECTIONS[i]`, so reordering this list would silently scramble any saved genome. The same list appears in `arena.py` as `NEIGHBOUR_STEPS`.

### `SIMPLE_ACTIONS`

```python
SIMPLE_ACTIONS = [
    Action(ActionType.REST),
    Action(ActionType.DRINK),
    Action(ActionType.EAT),
    Action(ActionType.HUNT),
    Action(ActionType.PICK_UP),
    Action(ActionType.HEAL),
]
```

The six actions that need no extra details. `ATTACK`, `MOVE` and `FLEE` are missing on purpose because they need a target or a direction. Brains that pick from a menu build it as `SIMPLE_ACTIONS + moves`: the random brain uses six simple actions plus eight moves (14 items), and the neural brain uses six simple plus "attack nearest", "flee nearest" and eight moves (`MENU_SIZE = 16`).

## How to use it / experiment

**Build a menu the way the brains do.**

```python
from hunger_games.actions import DIRECTIONS, SIMPLE_ACTIONS, Action

menu = SIMPLE_ACTIONS + [Action.move(dx, dy) for dx, dy in DIRECTIONS]
print(len(menu))          # 14
for index, action in enumerate(menu):
    print(index, action.kind.value, action.dx, action.dy)
```

**Feed a hand-made action into a real game.** The referee's `_resolve_action` is a private method, but calling it is the quickest way to see what an action does.

```python
from hunger_games import Game, SimulationConfig
from hunger_games.actions import Action, ActionType

game = Game(SimulationConfig(width=60, height=60, seed=5))
player = game.players[0]
print(player.position)
game._resolve_action(player, Action.move(1, 0))
print(player.position)    # one cell right, unless the terrain roll failed
```

**Add a new action.** Say you want `SHOUT` to scare neighbours. You would touch four places:

1. Add `SHOUT = "shout"` to `ActionType` here, and append `Action(ActionType.SHOUT)` to `SIMPLE_ACTIONS` if it needs no details.
2. Give the body a method in [player.md](player.md), for example `Player.shout(...)`.
3. Add an `elif action.kind is ActionType.SHOUT:` branch to `Game._resolve_action` in [game.md](game.md).
4. Teach at least one brain to choose it. The neural brain's `MENU_SIZE` grows automatically if you extended `SIMPLE_ACTIONS`, but any saved genome from before the change is now the wrong size.

**Neural network brain.** The network never sees this file's strings. It outputs one score per menu slot and `NeuralBrain.menu_to_action` turns the winning slot into an `Action` using `SIMPLE_ACTIONS`, `DIRECTIONS` and the perception. If you change the menu, change that mapping too.

**Genetic algorithm.** Genomes live in the brains, not here. This file only matters to a GA through menu ordering, as above.

**Reward function.** A reward wrapper around `Game.step()` will inspect `player.last_action` (an `Action`) to decide how much credit to give. `FLEE` versus `MOVE` is the hook for rewarding escapes differently from ordinary walking.

## Gotchas

- **Nothing validates `dx` and `dy`.** The comments say -1, 0 or +1, but `Action.move(5, 0)` is accepted and `Player.move` will happily teleport five cells if the destination is walkable. Brains must be well behaved.
- **`Action.move(0, 0)` is legal** and is "step into my own cell". The body still rolls the terrain's move-success chance, so it can even "fail".
- **Frozen means frozen.** `action.dx = 1` raises `FrozenInstanceError`. Build a new action instead.
- **Equality is by value, not identity.** Two separately built `Action.move(1, 0)` objects are `==` and share a hash. This is a feature for vote counting, but do not rely on `is` between actions.
- **`FLEE` and `MOVE` are identical to the referee.** Only a future reward function or logger would tell them apart.
- **`target_id=None` on an `ATTACK`** makes the attack fizzle silently in `Game._resolve_attack`. So does a dead or out-of-reach target.
- **`SIMPLE_ACTIONS` and `DIRECTIONS` are shared module-level lists.** Never mutate them in place (`SIMPLE_ACTIONS.append(...)`); every brain would see the change.
- **Order is a contract.** Changing the order of `DIRECTIONS` or `SIMPLE_ACTIONS` changes what every trained neural genome means.
