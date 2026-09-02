# `test_brains.py`

**Source:** [tests/test_brains.py](../../tests/test_brains.py)
**Tests:** [../brain/init.md](../brain/init.md) (`BRAIN_REGISTRY`, `create_brain`), [../brain/base.md](../brain/base.md) (`Brain`), [../brain/voting.md](../brain/voting.md) (`VotingBrain`, `DEFAULT_GENES`), [../brain/random_brain.md](../brain/random_brain.md) (`RandomBrain`), [../brain/neural.md](../brain/neural.md) (`NeuralBrain`, which wraps the `MLP` in `brain/mlp.py`), [../perception.md](../perception.md) (`Perception.to_vector`, `VECTOR_SIZE`), with `Game` from [../game.md](../game.md) used only to build a realistic perception

## Purpose

Chapter 4 splits a tribute into a body and a brain. The body (`Player`) gathers a `Perception` and carries out an `Action`. The brain decides which action. Because the brain only ever sees a `Perception`, you can swap in any decision-maker: the chapter 4 voting brain, a dice-rolling baseline, or a small neural network. This file guards that contract.

The first test pins the length of `Perception.to_vector()` to the `VECTOR_SIZE` constant. A neural network sizes its input layer from that constant, so if the two drift apart the network crashes on its first decision. The second test walks every brain in `BRAIN_REGISTRY` and checks that `decide()` returns an `Action` at both ends of the chaos dial. The third checks that `genome()` and `set_genome()` are true inverses, which a genetic algorithm needs. The last test checks one specific piece of behaviour in the voting brain: a very thirsty tribute standing in water drinks.

The bugs each would catch: a new field added to `Perception` without updating `VECTOR_SIZE`; a brain that returns `None` or a raw string when chaos is 0 because the sampling branch is never reached; a `set_genome` that stores a reference instead of a copy, or slices the flat vector at the wrong offsets; and a change to the voting weights that lets a small greed or idle vote beat a life-or-death thirst vote.

## Concepts you need

**Test discovery.** pytest runs every function whose name starts with `test_`. `first_perception` has no such prefix, so it is a helper and only runs when a test calls it.

**Helper functions.** `first_perception()` builds a whole `Game` and returns what player 0 can see on tick 0. Three tests call it. Each call builds a fresh game, so tests never share state.

**Plain `assert` and `isinstance`.** `assert isinstance(x, Action)` fails unless `x` is an `Action` or a subclass of it.

**Nested loops inside a test.** `test_every_registered_brain_returns_an_action` loops over three brain names and two chaos values, six combinations in one test. If any combination fails, the whole test fails and the traceback shows the values of `name` and `chaos` at that moment.

**Editing a dataclass.** `Perception` is a plain dataclass, so a test can overwrite fields like `perception.thirst = 0.05` to stage a situation without needing the game to reach it naturally.

**numpy comparisons.** `np.allclose(a, b)` is True when every element matches within a small tolerance. `array.shape == (50,)` checks a one-dimensional array of length 50.

**Running a subset.** `python -m pytest tests/test_brains.py -k genome`.

## Walkthrough

### `first_perception()`

```python
def first_perception():
    game = Game(SimulationConfig(width=60, height=60, seed=5))
    player = game.players[0]
    return player.perceive(game.arena, game.players, False, 0.0, 1.0, game.config.vision_radius)
```

Builds a 60 by 60 game with `seed=5`, takes the first tribute, and asks their body to look around. The arguments to `perceive` are: the arena, the full player list (the body skips itself and the dead), `lethal_here=False` (no game maker hazard), `day_fraction=0.0` (the start), `alive_fraction=1.0` (everyone alive), and the default `vision_radius` of 8. The optional arguments (`landmark_radius`, `hazard_distance`, `hazard_closing`, `field`) keep their defaults, so the field is unknown and the hazard is far away. With this seed, player 0 stands on sand at (21, 57) with nobody in sight and one supply one cell away. All three bars are full, because the default `start_*_min` settings are `1.0`. That detail does not matter to the tests, which is the point: they only need a realistic, fully populated `Perception` object.

### `test_vector_size_matches_constant()`

```python
def test_vector_size_matches_constant():
```

**Setup.** Calls `first_perception().to_vector()`.

**`assert ... .shape == (VECTOR_SIZE,)`.** `to_vector` assembles a fixed list of floats: 11 body and terrain values, 2 for the downhill step, 3 each for water, grass and centre, 3 for the supply underfoot, 4 for the nearest supply, 5 for the nearest player, 1 crowd count, 3 hazard values (inside, distance, closing), 2 for the safe direction, 2 for the clock, 4 for what the sky told you about the field, and 4 one-hot terrain flags. That is 50, and `VECTOR_SIZE = 50` is written by hand in `perception.py`, with the count in a comment beside it. `NeuralBrain.__init__` uses `VECTOR_SIZE` as the first entry of `layer_sizes`, so the `MLP` shapes its first weight matrix from it. A failure would mean someone changed `to_vector` without updating the constant, and every neural brain would raise a shape error on its first matrix multiply.

### `test_every_registered_brain_returns_an_action()`

```python
def test_every_registered_brain_returns_an_action():
```

**Setup.** One perception, one `default_rng(0)`, then a loop over every name in `BRAIN_REGISTRY` ("voting", "random", "neural") and over `chaos in (0.0, 1.0)`. `create_brain(name, chaos, rng)` builds the brain. Its two optional arguments, `neural` (a `NeuralConfig`) and `endgame`, are left at `None` and `False`, so the neural brain uses the default architecture and the voting brain has the endgame instinct off. The generator is passed to `NeuralBrain` so its starting weights are seeded.

**`assert isinstance(brain.decide(perception, rng), Action)`.** This is the whole brain interface in one line: give it a perception and a generator, get back an `Action`. Both chaos values are tested because each brain has two code paths. `Ballot.winner` in `voting.py` returns the favourite directly at chaos 0 and samples from a blended probability at chaos 1. `NeuralBrain.decide_index` builds a one-hot at chaos 0 and takes its `argmax`, and above zero draws from a softmax with chaos as the temperature. A failure at only one chaos value would point straight at the branch that broke. A `KeyError` would mean a brain class had been added to the registry under the wrong name.

**Why one shared `rng`.** Reusing the generator across all six calls keeps the test deterministic without needing six seeds.

### `test_genome_round_trip()`

```python
def test_genome_round_trip():
```

**Setup.** For every registered brain, build it with chaos 0 and `default_rng(1)`, read its genome, add 0.1 to every value, and load that back with `set_genome`.

**`assert np.allclose(brain.genome(), mutated)`.** What you store must be what you get back. This is the contract a genetic algorithm depends on: mutate the vector, load it, play a game, read it out again. The three brains exercise three different implementations. `VotingBrain` stores 8 genes and copies on both read and write. `NeuralBrain` delegates to `MLP.genome` and `MLP.set_genome`, which flatten each layer's weight matrix followed by its bias vector and slice them back out by a running cursor; with the default `(16,)` hidden layer that is 1088 numbers, and an off-by-one in any slice would fail here. `RandomBrain` inherits the base class behaviour and returns an empty array, so `mutated` is also empty and the comparison is trivially true. A failure would mean a slice boundary was wrong, a `.copy()` was missing, or the vector order changed between `genome` and `set_genome`.

**Why `+ 0.1`.** Adding a constant changes every element, so a `set_genome` that silently ignored part of the vector would be caught.

### `test_voting_brain_drinks_when_thirsty_in_water()`

```python
def test_voting_brain_drinks_when_thirsty_in_water():
```

**Setup.** Starts from `first_perception()` and edits it to stage the situation: `thirst = 0.05` (almost dead), `in_water = True`, no nearby players, no danger zone, nothing underfoot, and no supply in sight (`nearby_resource_distance = inf`). Everything else, such as hunger and health, keeps its real value from the game, which is full. Then `VotingBrain(chaos=0.0).decide(...)` is called. Chaos 0 means the top-voted action always wins, so the test is exact.

**`assert action.kind.value == "drink"`.** Here is how the ballot fills in. The thirst instinct computes `urgency(0.05)`: `(1 - 0.05) ** 2 * 10 = 9.025`, plus `CRITICAL_BONUS` of 20 because the bar is below `CRITICAL_LEVEL` (0.2), so about 29 votes, and because `in_water` is True it casts them for `DRINK`. The survival instinct adds a little more for `DRINK` because the tribute is in water with thirst below 0.95. The next best action is a move toward the centre from the greed instinct, at about 1.1 votes, because the tribute has no weapon. Idle adds 0.5 for `REST` and 0.3 for a random step. Hunger and health are full, so those instincts stay silent, and nobody is in sight, so the danger instinct stays silent too. `DRINK` wins by a factor of more than twenty. The test compares `action.kind.value`, the string `"drink"`, rather than importing `ActionType`. A failure would mean the thirst instinct no longer fires when in water, or the weights had been rebalanced so badly that a trickle vote beats an emergency.

**Why `thirst = 0.05` and chaos 0.** A near-empty bar gives near-maximum urgency plus the critical bonus, and chaos 0 removes sampling, so the outcome is fully determined.

**`assert len(DEFAULT_GENES) == 8`.** `GENE_NAMES` lists eight names (`thirst_weight`, `hunger_weight`, `survival_weight`, `danger_weight`, `greed_weight`, `aggression`, `caution`, `urgency_power`) and `DEFAULT_GENES` must have one value per name, because `gene(name)` looks up by index. If someone adds a gene name and forgets its default, this fails before any game does.

## How to run and extend

```bash
python -m pytest tests/test_brains.py
python -m pytest tests/test_brains.py::test_genome_round_trip
python -m pytest tests -k "brain and not genome"
python -m pytest tests/test_brains.py -v
```

Ideas for new tests in this area:

**1. A custom `Brain` subclass works with `Game`.** The registry is not required; `Game` accepts a `brain_factory`.

```python
from hunger_games.actions import Action, ActionType
from hunger_games.brain import Brain

class Sleeper(Brain):
    name = "sleeper"
    def decide(self, perception, rng):
        return Action(ActionType.REST)

def test_custom_brain_drives_a_game():
    config = SimulationConfig(width=60, height=60, seed=5, max_days=2)
    game = Game(config, brain_factory=lambda index, rng: Sleeper())
    result = game.run()
    assert all(row.brain == "sleeper" for row in result.players)
```

**2. The danger zone overrides everything.** Fifty hazard votes should beat even maximal thirst.

```python
def test_hazard_vote_beats_thirst():
    perception = first_perception()
    perception.thirst = 0.0
    perception.in_water = True
    perception.in_danger_zone = True
    perception.safe_direction = (1, 0)
    action = VotingBrain(chaos=0.0).decide(perception, np.random.default_rng(0))
    assert action == Action.move(1, 0)
```

**3. A wrong-sized genome is rejected.** Both the voting brain and the `MLP` raise `ValueError`.

```python
import pytest

def test_wrong_genome_sizes_are_rejected():
    with pytest.raises(ValueError):
        VotingBrain().set_genome(np.zeros(7))
    with pytest.raises(ValueError):
        create_brain("neural", 0.0, np.random.default_rng(0)).set_genome(np.zeros(3))
```

**4. Unknown brain names give a clear error.**

```python
def test_unknown_brain_name():
    with pytest.raises(KeyError):
        create_brain("psychic", 0.0, np.random.default_rng(0))
```

## Gotchas

**Building a `Game` for a perception.** `first_perception` runs the full `Game.__init__`: noise, terrain, layout, 24 players. It takes about 0.03 seconds. That is fine for four tests, but if you write many brain tests, consider building the perception once at module level or constructing a `Perception` by hand.

**Starting bars are full by default.** `start_thirst_min`, `start_hunger_min` and `start_health_min` all default to `1.0`, so player 0 begins with every bar at `1.0` regardless of chaos. The drink test overrides thirst only. If those defaults ever drop below the instinct thresholds (0.85 for thirst, 0.8 for hunger), competing votes appear, though 29 thirst votes would still win.

**Chaos above zero samples.** If you write a behaviour test with `chaos=1.0`, the winner is drawn from a probability distribution and the test may pass or fail depending on the generator state. Use `chaos=0.0` for exact tests.

**`RandomBrain` has no genome.** Its round trip passes on an empty array. Do not mistake this for coverage of a real load path.

**`NeuralBrain` needs a generator.** `create_brain` passes one. If you build `NeuralBrain()` directly without `rng`, the `MLP` seeds from fresh entropy and its decisions will change between runs.

**The endgame instinct is off here.** `create_brain` passes `endgame=False` unless told otherwise, and `VotingBrain(chaos=0.0)` in the drink test uses the same default. A game built from a config with `endgame_instinct=True` gets a brain that also votes to head for the centre once fewer than half the tributes remain.
