# `base.py`

**Source:** [hunger_games/brain/base.py](../../hunger_games/brain/base.py)
**Depends on:** `abc`, `copy` (standard library); `numpy`; [hunger_games/actions.py](../actions.md) (`Action`); [hunger_games/perception.py](../perception.md) (`Perception`)
**Used by:** [brain/__init__.py](init.md), [brain/voting.py](voting.md), [brain/random_brain.py](random_brain.md), [brain/neural.py](neural.md), [hunger_games/player.py](../player.md) (type hint for `Player.brain`), [hunger_games/game.py](../game.md) (through the package import)

## Purpose

Chapter 4 of the video splits a tribute into a *body* (what they can physically do) and a *brain* (what they decide to do). This file defines the brain half. It is a contract, not a working brain. It says: "anything that can take a `Perception` and return an `Action` may drive a player."

The contract is small on purpose. There is one required method, `decide()`. Everything else is an optional hook that does nothing by default. That means a beginner can write a working brain in five lines, while a training script can later use the hooks to evolve genomes or feed back rewards without changing `game.py` or `player.py`.

Because the game only ever talks to a brain through this class, you can swap the voting brain for a neural network, a genetic algorithm genome, or a hand-written rule set, and the rest of the simulator never notices. This is the single most important extension point in the package.

## Concepts you need

**Abstract base class (ABC).** `class Brain(ABC)` with an `@abstractmethod` means Python refuses to create a `Brain()` directly, and refuses to create any subclass that forgets to write `decide()`. It is a compile-time-style safety net: forget the method and you get a clear `TypeError` the moment you call the constructor, not a confusing crash deep inside the game loop.

**Class attribute versus instance attribute.** `name = "base"` is written at class level, so every subclass inherits it and can override it with one line (`name = "voting"`). `self.chaos` is set in `__init__`, so every brain object has its own copy.

**Genome.** A genome is just a flat one-dimensional numpy array of the numbers that shape a brain's behaviour. A genetic algorithm does not need to know what the numbers mean. It only needs `genome()` to read them and `set_genome()` to write them back. The base class returns an empty array, meaning "nothing to tune".

**Deep copy.** `copy.deepcopy` copies an object and every object inside it, including numpy arrays. A shallow copy would share the arrays, so mutating one clone's genome would silently mutate the other.

## Walkthrough

### `class Brain(ABC)`

The base class for every decision-maker. Subclasses must implement `decide()`. They may override any of the hooks.

#### `name = "base"`

A short label. `Game.result()` writes `player.brain.name` into the `brain` column of the players CSV, so batches that mix brains can be compared later with pandas. `BRAIN_REGISTRY` in [init.md](init.md) also uses it as the dictionary key. Every subclass should override it with a unique string.

#### `__init__(self, chaos: float = 0.0) -> None`

Stores the chaos dial as `self.chaos`. `0.0` means "always pick the best action", `1.0` means "very unpredictable". The base class does nothing else with it; each subclass decides how to turn chaos into noise (the voting brain blends probabilities, the neural brain uses it as a softmax temperature).

```python
from hunger_games.brain.random_brain import RandomBrain
brain = RandomBrain(chaos=0.3)
print(brain.chaos)   # 0.3
```

#### `decide(self, perception: Perception, rng: np.random.Generator) -> Action` (abstract)

The one method every brain must write. Called once per tick for every living player.

- `perception`: a `Perception` snapshot of what the player senses (see [../perception.md](../perception.md)).
- `rng`: the game's single numpy random generator. Always draw randomness from this, never from `random` or a fresh generator, or the game stops being reproducible from its seed.
- Returns an `Action` (see [../actions.md](../actions.md)).

The body (`Player.decide`) calls this and remembers the result in `player.last_action`; the referee (`Game._resolve_action`) then carries it out.

#### `genome(self) -> np.ndarray`

Returns the brain's tunable numbers as a flat float vector. Default: `np.zeros(0, dtype=float)`, an empty array. Override it when your brain has parameters worth evolving.

#### `set_genome(self, genome: np.ndarray) -> None`

The reverse of `genome()`. Loads a flat vector back into the brain. Default: does nothing and returns `None`. A genetic algorithm calls this on a fresh brain before each game.

```python
import numpy as np
from hunger_games.brain.voting import VotingBrain
brain = VotingBrain()
genes = brain.genome()          # a copy of the 8 default genes
genes[0] *= 2                   # make thirst twice as loud
brain.set_genome(genes)         # load it back
```

#### `observe(self, perception: Perception, action: Action, reward: float) -> None`

A reinforcement-learning hook. Intended to be called after each action with a reward number. Default: does nothing. Important: **the base game never calls this**. `Game.step()` does not compute rewards, because what counts as a good tick is a training decision, not a rules decision. A training script wraps `Game` and calls `observe()` itself. See "How to use it" below.

#### `on_game_end(self, placement: int, kills: int, days_survived: float) -> None`

Called exactly once per brain by `Game._finish()` when the game ends. `placement` is 1 for the victor and higher for earlier deaths (24 means first out of 24). `kills` is how many other tributes this player eliminated. `days_survived` is ticks alive divided by `ticks_per_day`. Default: does nothing. This is the natural place for a learner to score a whole episode.

#### `clone(self) -> "Brain"`

Returns `copy.deepcopy(self)`: a fresh, independent brain with the same parameters. Useful when you want to give a winner's brain to several players in the next generation without them sharing one genome array.

```python
child = brain.clone()
child.set_genome(child.genome() + 0.05)   # only the child changes
```

## How to use it / experiment

**Write a new brain.** Subclass `Brain`, give it a `name`, implement `decide()`.

```python
import numpy as np
from hunger_games.actions import Action, ActionType
from hunger_games.brain.base import Brain
from hunger_games.perception import Perception

class ThirstyBrain(Brain):
    """Drinks when in water, otherwise walks toward water, otherwise rests."""
    name = "thirsty"

    def decide(self, perception: Perception, rng: np.random.Generator) -> Action:
        if perception.in_water:
            return Action(ActionType.DRINK)
        if perception.water_direction != (0, 0):
            return Action.move(*perception.water_direction)
        return Action(ActionType.REST)
```

**Register it** so `--brain thirsty` and `SimulationConfig(brain_name="thirsty")` work. `create_brain()` calls `cls(chaos=chaos)` for every brain except the neural one, so your `__init__` must accept `chaos`.

```python
from hunger_games.brain import BRAIN_REGISTRY
BRAIN_REGISTRY[ThirstyBrain.name] = ThirstyBrain
```

**Bypass the registry** with a brain factory. `Game` accepts `brain_factory=(player_index, rng) -> Brain`, so you can hand out any brain object, including ones with custom constructor arguments.

```python
from hunger_games.config import SimulationConfig
from hunger_games.game import Game

config = SimulationConfig(width=60, height=60, seed=3)
game = Game(config, brain_factory=lambda index, rng: ThirstyBrain(chaos=0.2))
result = game.run()
print(result.winner_name)
```

**Hook a reward into `observe()`.** Subclass, remember the last perception and action in `decide()`, then have your loop call `observe()` after every `game.step()`.

```python
class LearningBrain(ThirstyBrain):
    name = "learning"

    def __init__(self, chaos: float = 0.0) -> None:
        super().__init__(chaos)
        self.last: tuple[Perception, Action] | None = None
        self.total_reward = 0.0

    def decide(self, perception, rng):
        action = super().decide(perception, rng)
        self.last = (perception, action)
        return action

    def observe(self, perception, action, reward):
        self.total_reward += reward     # a real learner would update weights here

game = Game(config, brain_factory=lambda i, rng: LearningBrain())
previous = {p.player_id: p.thirst + p.hunger + p.health for p in game.players}
while not game.is_over:
    game.step()
    for player in game.players:
        brain = player.brain
        if brain.last is None:
            continue
        wellbeing = player.thirst + player.hunger + player.health
        reward = wellbeing - previous[player.player_id] - (0.0 if player.alive else 5.0)
        previous[player.player_id] = wellbeing
        brain.observe(*brain.last, reward)
        brain.last = None
```

**Compare brains** by running `Runner` once per brain name and reading the `brain` column. See [random_brain.md](random_brain.md) for a full snippet.

## Gotchas

- Forgetting `decide()` in a subclass raises `TypeError: Can't instantiate abstract class ...` when you construct it, not when you import it.
- `observe()` is never called by `Game`. If your learner seems to receive no rewards, that is why. You must call it from your own loop.
- `on_game_end()` is called for every player, dead or alive, once per game. `Game._finish()` guards against running twice.
- `create_brain()` only passes `chaos=`. If your `__init__` needs other arguments, give them defaults or use `brain_factory`.
- `genome()` in the base class returns an empty array. `set_genome()` on such a brain silently does nothing, so a genetic algorithm that "evolves" a `RandomBrain` will run without error and change nothing.
- Draw all randomness from the `rng` passed to `decide()`. A brain that uses `np.random.default_rng()` internally breaks seed reproducibility.
