# `random_brain.py`

**Source:** [hunger_games/brain/random_brain.py](../../hunger_games/brain/random_brain.py)
**Depends on:** `numpy`; [hunger_games/actions.py](../actions.md) (`DIRECTIONS`, `SIMPLE_ACTIONS`, `Action`); [brain/base.py](base.md) (`Brain`); [hunger_games/perception.py](../perception.md) (`Perception`, type hint only)
**Used by:** [brain/__init__.py](init.md) (registered as `"random"`)

## Purpose

This is the simplest possible brain: every tick it ignores what the player senses and picks one action from a fixed menu at random. It is deliberately stupid.

Its job is to be a baseline. When you write a smarter brain, or start training one, you need a floor to measure against. If the smart brain does not out-survive coin flips, something is broken. Machine-learning projects call this a "random policy" and it is usually the first thing you run.

It is also the shortest complete example of the `Brain` contract. If you are writing your own brain, start by reading this file; it shows exactly how little is required.

## Concepts you need

**Inheritance.** `class RandomBrain(Brain)` inherits `__init__` (which stores `chaos`), `genome()`, `set_genome()`, `observe()`, `on_game_end()` and `clone()` from the base class without rewriting them. Only `name` and `decide()` are new.

**Frozen dataclasses as menu items.** `Action` is a `@dataclass(frozen=True)`, so the `Action` objects in `SIMPLE_ACTIONS` are immutable and safe to hand out again and again. Returning the same object twice is fine because nobody can change it.

**`rng.integers(n)`.** Returns a random integer from `0` to `n - 1` inclusive. It comes back as a numpy integer, which is why the code wraps it in `int()` before using it as a list index.

## Walkthrough

### `class RandomBrain(Brain)`

#### `name = "random"`

The label written to the players CSV and the key in `BRAIN_REGISTRY`.

#### `decide(self, perception: Perception, rng: np.random.Generator) -> Action`

Step by step:

1. Build the menu: `SIMPLE_ACTIONS + [Action.move(dx, dy) for dx, dy in DIRECTIONS]`. That is the six simple actions (REST, DRINK, EAT, HUNT, PICK_UP, HEAL) followed by one MOVE for each of the eight compass directions. Fourteen items in all.
2. Draw an index with `rng.integers(len(menu))`.
3. Return that menu item.

The `perception` argument is accepted (the contract requires it) but never read. The menu contains no ATTACK or FLEE, because those need a target taken from the perception, and this brain does not look.

Parameters: `perception` (ignored), `rng` (the game's generator). Returns an `Action`.

```python
import numpy as np
from hunger_games.brain.random_brain import RandomBrain

brain = RandomBrain()
rng = np.random.default_rng(1)
for _ in range(3):
    print(brain.decide(None, rng))    # perception is never touched, so None works here
```

Why rebuild the menu on every call instead of once at module level? Simplicity: the list is fourteen cheap objects, and building it inline keeps the whole brain in one method a beginner can read top to bottom. If you profile a batch of 500 games and this shows up, hoist it to a module constant.

Note that `chaos` has no effect on this brain. It is already as random as it can be.

## How to use it / experiment

**Run it from the command line.**

```bash
python -m hunger_games watch --brain random --seed 5
python -m hunger_games simulate --brain random --games 50
```

**Use it as a baseline in a comparison.** Run the same batch once per brain and compare the tables `Runner` returns.

```python
from hunger_games.config import SimulationConfig
from hunger_games.runner import Runner

for name in ("random", "voting", "neural"):
    config = SimulationConfig(brain_name=name, width=60, height=60, seed=42)
    eliminations, players, games = Runner(config, num_games=30, output_dir=f"output/{name}").run(show_progress=False)
    natural = (eliminations["method"] == "natural_causes").mean()
    print(f"{name:7s} mean days {players['days_survived'].mean():5.2f}  "
          f"natural deaths {natural:4.0%}  mean kills {players['kills'].mean():.2f}")
```

A brain that is learning should push mean days survived above the random line and push the natural-causes fraction below it.

**Put random players in the same arena as smart ones.** Mixing brains with `brain_factory` gives a fairer comparison because both kinds face the same map, the same game makers, and each other.

```python
import numpy as np
from hunger_games.brain.random_brain import RandomBrain
from hunger_games.brain.voting import VotingBrain
from hunger_games.config import SimulationConfig
from hunger_games.game import Game

config = SimulationConfig(width=60, height=60, seed=9)
factory = lambda index, rng: RandomBrain(chaos=0.5) if index < 12 else VotingBrain(chaos=0.5)
result = Game(config, brain_factory=factory).run()
for row in sorted(result.players, key=lambda r: r.placement):
    print(row.placement, row.brain, row.name)
```

**Write your own baseline.** Copy this file, rename the class and `name`, and replace the menu. For example, a "walker" that only ever moves is a useful test of the game makers' shrinking circle.

```python
from hunger_games.actions import DIRECTIONS, Action
from hunger_games.brain.base import Brain

class WalkerBrain(Brain):
    name = "walker"
    def decide(self, perception, rng):
        dx, dy = DIRECTIONS[int(rng.integers(len(DIRECTIONS)))]
        return Action.move(dx, dy)
```

**Genomes and rewards.** `RandomBrain` inherits the empty `genome()` and the do-nothing `observe()`, so there is nothing to evolve or train. That is exactly what makes it a baseline. See [base.md](base.md) for how to add those hooks to a brain of your own.

## Gotchas

- `chaos` is stored but unused. Changing `--chaos` changes the world (terrain, luck in fights, starting thirst) but not this brain's choices.
- The menu never includes ATTACK or FLEE, so random players never kill anyone on purpose. If the eliminations table for a random batch shows `player_vs_player` rows, look at the other brains in the game.
- A random player standing on dry land will happily choose DRINK, and one with an empty pack will choose EAT. Those actions simply fail in `Player.drink()` and `Player.eat()`. Nothing crashes; the tick is wasted.
- `decide()` never reads the perception, so passing `None` works in a quick test. Do not rely on that for other brains.
