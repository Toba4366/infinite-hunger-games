# `__init__.py`

**Source:** [hunger_games/brain/__init__.py](../../hunger_games/brain/__init__.py)
**Depends on:** `numpy` (the generator type hint); [brain/base.py](base.md) (`Brain`); [brain/neural.py](neural.md) (`NeuralBrain`); [brain/random_brain.py](random_brain.md) (`RandomBrain`); [brain/voting.py](voting.md) (`VotingBrain`); [hunger_games/config.py](../config.md) (`NeuralConfig`)
**Used by:** [hunger_games/game.py](../game.md) (`Game._make_brain` calls `create_brain` with the config's `neural` and `endgame_instinct`); [training/genetic.py](../training/genetic.md) (`play_evaluation_game`, `play_validation_game`, `GeneticTrainer.__init__`, `GeneticTrainer.champion_brain`); [training/reinforce.py](../../hunger_games/training/reinforce.py) (`play_rl_episode` builds the opponents); [hunger_games/ui/app.py](../ui/app.md) (`BRAIN_REGISTRY` fills the brain drop-down); `tests/test_brains.py`

## Purpose

This file is the front door of the `brain` package. It does two small jobs:

1. It gathers the three built-in brains into one dictionary, `BRAIN_REGISTRY`, keyed by their `name` strings. That is what lets a config say `brain_name="neural"` or the command line say `--brain voting`.
2. It provides `create_brain()`, the one function the rest of the simulator calls to build a brain from a name. Callers never write `VotingBrain(...)` or `NeuralBrain(...)` themselves.

Two brains need more than the chaos dial. The neural brain needs a `NeuralConfig` (its layer widths, activation and initializer) and a random generator for its starting weights. The voting brain has an optional endgame instinct switched on by a boolean. `create_brain()` hides both special cases so every caller uses the same five-argument call.

## Concepts you need

**Package `__init__.py`.** When you write `import hunger_games.brain`, Python runs this file. Anything it imports or defines becomes available as `hunger_games.brain.Something`. That is why `from hunger_games.brain import Brain, create_brain` works even though `Brain` really lives in `base.py`.

**Registry pattern.** A dictionary from string to class. Instead of a chain of `if name == "voting": ... elif name == "random": ...`, you look the class up and call it. Adding a brain is one dictionary entry, not a code change in every caller.

**`type[Brain]`.** The type hint `dict[str, type[Brain]]` says the values are *classes* (the blueprint), not instances (a built brain). Calling a class builds an instance.

**Optional arguments.** `neural: NeuralConfig | None = None` and `endgame: bool = False` have defaults, so old two-or-three-argument calls still work. `None` means "use `NeuralConfig()`", and `False` means "no endgame instinct".

## Walkthrough

### Imports

`Brain` (the abstract base), the three concrete brains, and `NeuralConfig` from [config.py](../config.md). Importing the brains here also re-exports them, so `from hunger_games.brain import VotingBrain` works.

### `BRAIN_REGISTRY: dict[str, type[Brain]]`

```python
BRAIN_REGISTRY = {
    VotingBrain.name: VotingBrain,   # "voting"
    RandomBrain.name: RandomBrain,   # "random"
    NeuralBrain.name: NeuralBrain,   # "neural"
}
```

The keys come from each class's `name` attribute, so the string in the config, the key here, and the `brain` column in the results CSV always match. Order matters a little: the dashboard lists brains in this order, and the error message from `create_brain` lists them in this order.

### `create_brain(name: str, chaos: float, rng: np.random.Generator, neural: NeuralConfig | None = None, endgame: bool = False) -> Brain`

Builds one fresh brain.

| Parameter | Meaning | Who uses it |
| --- | --- | --- |
| `name` | A key of `BRAIN_REGISTRY` | Everyone |
| `chaos` | The randomness dial, 0.0 to 1.0 | Every brain's constructor |
| `rng` | The game's numpy generator | Only `"neural"`, to draw its starting weights |
| `neural` | A `NeuralConfig`, or `None` for the defaults | Only `"neural"` |
| `endgame` | Switch on the head-for-the-centre instinct | Only `"voting"` |

The logic, in order:

1. If `name` is not in the registry, raise `KeyError("Unknown brain 'x'. Choose from: voting, random, neural")`.
2. If `name == "neural"`, return `NeuralBrain(chaos=chaos, config=neural, rng=rng)`.
3. If `name == "voting"`, return `VotingBrain(chaos=chaos, endgame=endgame)`.
4. Otherwise return `BRAIN_REGISTRY[name](chaos=chaos)`, which today means `RandomBrain(chaos=chaos)`.

Returns a `Brain` instance.

```python
import numpy as np
from hunger_games.brain import create_brain
from hunger_games.config import NeuralConfig

rng = np.random.default_rng(0)
voter = create_brain("voting", 0.5, rng, endgame=True)
net = create_brain("neural", 0.0, rng, NeuralConfig(hidden_layers=(32, 16), activation="relu", initializer="he_uniform"))
print(voter.name, voter.genome().size, voter.endgame)   # voting 8 True
print(net.describe())                                    # 50 -> 32 -> 16 -> 16, relu, he_uniform, 2432 params
```

Design reasoning: `Game` builds one brain per tribute and knows only the string in the config or the roster. The trainers build hundreds of brains and know only `TrainingConfig.brain_name`. None of them should need to know which brain takes which extra argument. Putting that knowledge here keeps it in exactly one place. `Game._make_brain` passes `self.config.neural` and `self.config.endgame_instinct`, so the dashboard's checkboxes reach the brains through this function.

### `__all__`

`["Brain", "VotingBrain", "RandomBrain", "NeuralBrain", "BRAIN_REGISTRY", "create_brain"]`. This is what `from hunger_games.brain import *` pulls in. It is also a good list of the public API.

## How to use it / experiment

**Pick a brain from the config.**

```python
from hunger_games.config import SimulationConfig, NeuralConfig
config = SimulationConfig(brain_name="neural", neural=NeuralConfig(hidden_layers=(24,), activation="selu", initializer="lecun_normal"))
config = SimulationConfig(brain_name="voting", endgame_instinct=True)
```

`Game._make_brain` forwards `config.neural` and `config.endgame_instinct`, so the whole roster gets the settings you chose.

**Register your own brain.** Subclass `Brain` (see [base.md](base.md)), give it a unique `name`, make sure its `__init__` accepts `chaos`, and add it to the registry before building a game.

```python
from hunger_games.actions import Action, ActionType
from hunger_games.brain import BRAIN_REGISTRY
from hunger_games.brain.base import Brain

class CowardBrain(Brain):
    name = "coward"
    def decide(self, perception, rng):
        threat = perception.nearest_threat
        if threat is not None:
            return Action.flee(*threat.direction_away())
        return Action(ActionType.REST)

BRAIN_REGISTRY[CowardBrain.name] = CowardBrain
```

Now `SimulationConfig(brain_name="coward")` works. `TrainingConfig(brain_name="coward")` will be refused by the genetic trainer, because the brain has no genome to evolve.

**Mix brains in one game.** `create_brain` builds one brain at a time, so a `brain_factory` can call it with a different name per tribute:

```python
from hunger_games.game import Game
names = ["voting"] * 12 + ["neural"] * 12
game = Game(config, brain_factory=lambda index, rng: create_brain(names[index], config.chaos, rng, config.neural, config.endgame_instinct))
result = game.run()
```

The `brain` column of `result.players` then tells the two groups apart.

**Load a saved genome by name.** `GeneticTrainer.load_champion` returns `brain_name`, `neural` and `genome`. Build the brain with the first two and call `set_genome` with the third (see [../training/genetic.md](../training/genetic.md)).

## Gotchas

- `create_brain` only passes `chaos=` to brains other than `"neural"` and `"voting"`. A custom brain whose `__init__` needs more arguments must give them defaults or be built through `brain_factory` instead.
- `neural` is silently ignored for `"voting"` and `"random"`; `endgame` is silently ignored for `"neural"` and `"random"`. Passing them does no harm.
- Passing `neural=None` for `"neural"` does not error. You get the default architecture (50 -> 16 -> 16), which may not match a genome you are about to load. `set_genome` will then raise `ValueError` about the size. Always pass the same `NeuralConfig` the genome was trained with.
- `GeneticTrainer.__init__` and `champion_brain` call `create_brain` without `endgame`, so a voting brain built there has the instinct off even if the config has it on. The evaluation games (`play_evaluation_game`) do pass it.
- The `rng` you pass decides the neural brain's starting weights. `Game` passes its single seeded generator, so a seeded game gives every neural tribute reproducible starting weights.
- Registering a brain after a `Game` has been built has no effect on that game. The registry is read at construction time.
- `BRAIN_REGISTRY` is a plain module-level dict. Editing it changes it for the whole process, including the dashboard's drop-down if the dashboard is running in the same process.
