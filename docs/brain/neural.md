# `neural.py`

**Source:** [hunger_games/brain/neural.py](../../hunger_games/brain/neural.py)
**Depends on:** `numpy`; [hunger_games/actions.py](../actions.md) (`DIRECTIONS`, `SIMPLE_ACTIONS`, `Action`, `ActionType`); [brain/base.py](base.md) (`Brain`); [brain/mlp.py](mlp.md) (`MLP`); [hunger_games/config.py](../config.md) (`NeuralConfig`); [hunger_games/perception.py](../perception.md) (`VECTOR_SIZE`, `Perception`)
**Used by:** [brain/__init__.py](init.md) (registered as `"neural"`); [training/genetic.py](../training/genetic.md) (through `create_brain`, `genome`, `set_genome`); [training/reinforce.py](../training/reinforce.md) (`NeuralBrain`, `softmax`; reads `.network` and `last_index`); [training/imitation.py](../training/imitation.md) (`NeuralBrain`, `softmax`, `action_to_menu_index` for the labels); [hunger_games/ui/session.py](../ui/session.md) (`network_snapshot` reads `layer_sizes`, `network`, `probabilities`, `last_index`, `MENU_NAMES`); [hunger_games/ui/app.py](../ui/app.md) (`MENU_NAMES`, `MENU_SIZE`, `describe`); [hunger_games/ui/visualizer.py](../ui/visualizer.md) (`MENU_NAMES`); `tests/test_initializers.py`, `tests/test_brains.py`, `tests/test_imitation.py`

## Purpose

A brain that decides with a neural network instead of hand-written instincts. The network reads the 50-number perception vector and produces 16 scores, one per item on a fixed menu of actions. The highest score wins, or, with some chaos, the scores become probabilities and one is drawn.

A fresh `NeuralBrain` knows nothing. Its weights are random, so it plays about as well as the dice-rolling [random brain](random_brain.md). Three things can teach it:

- The **imitation trainer** ([../training/imitation.md](../training/imitation.md)) records the voting brain's decisions and trains the network to predict them. This is the recommended first step, because a random network dies of thirst before the other two methods can reward or select anything.
- The **genetic algorithm** ([../training/genetic.md](../training/genetic.md)) treats the whole network as one flat genome, breeds and mutates it, and keeps the genomes that win games.
- The **REINFORCE trainer** ([../training/reinforce.md](../training/reinforce.md)) rewards each action and backpropagates through the same network to make well-rewarded actions more likely.

This file is thin on purpose. The maths lives in [mlp.py](mlp.md). What this file adds is the mapping from the game to the network (perception in, menu index out), the mapping back (menu index to a concrete `Action`), and the reverse mapping the imitation trainer needs (a teacher's `Action` to a menu index).

## Concepts you need

**Logits.** The raw output scores of the network. Any real number, positive or negative. They are not probabilities yet.

**Softmax.** Turns logits into probabilities: exponentiate each, then divide by the sum. Bigger logits get bigger shares, and the shares add to one.

**Temperature.** Dividing the logits by a number before softmax. A temperature below 1 sharpens the distribution (the leader takes nearly all the mass); above 1 flattens it toward uniform. This brain uses the chaos dial as the temperature.

**Argmax.** The index of the largest value. With chaos 0 the brain is deterministic and just takes the argmax.

**Action menu.** The network cannot output an arbitrary `Action` object, so it chooses from a fixed list of 16 options. Two of them, "attack nearest" and "flee nearest", are filled in with a real target using the perception at decision time.

**Labels.** Supervised learning needs a correct answer per input. For imitation the input is a perception vector and the answer is the menu index of the action the teacher chose. `action_to_menu_index` produces that index from any `Action`.

**Genome.** Every weight and bias as one flat vector. See [base.md](base.md) and [mlp.md](mlp.md).

**Policy.** In reinforcement learning, the function from state to action probabilities. Here that is `softmax(network(perception))`. The trainer needs the probabilities and the index actually chosen, which is why `decide_index`, `last_index` and `last_probabilities` exist alongside `decide`.

## Walkthrough

### `MENU_SIZE`

`len(SIMPLE_ACTIONS) + 2 + len(DIRECTIONS)` = `6 + 2 + 8` = `16`. The width of the output layer.

### `ATTACK_INDEX`, `FLEE_INDEX`, `FIRST_MOVE_INDEX`

`6`, `7` and `8`. The menu is laid out as the six simple actions, then attack, then flee, then the eight moves.

### `MENU_NAMES`

Human-readable names in menu order, built from `SIMPLE_ACTIONS` and the direction names. The dashboard and the research plots label outputs with these.

| Index | Name | What `menu_to_action` returns |
| --- | --- | --- |
| 0 | `rest` | `Action(REST)` |
| 1 | `drink` | `Action(DRINK)` |
| 2 | `eat` | `Action(EAT)` |
| 3 | `hunt` | `Action(HUNT)` |
| 4 | `pick_up` | `Action(PICK_UP)` |
| 5 | `heal` | `Action(HEAL)` |
| 6 | `attack` | attack the nearest player if in reach, else step toward them, else rest |
| 7 | `flee` | flee one step away from the nearest player, else rest |
| 8 | `move up-left` | `Action.move(-1, -1)` |
| 9 | `move up` | `Action.move(0, -1)` |
| 10 | `move up-right` | `Action.move(1, -1)` |
| 11 | `move left` | `Action.move(-1, 0)` |
| 12 | `move right` | `Action.move(1, 0)` |
| 13 | `move down-left` | `Action.move(-1, 1)` |
| 14 | `move down` | `Action.move(0, 1)` |
| 15 | `move down-right` | `Action.move(1, 1)` |

### `softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray`

1. Divide by `max(temperature, 1e-6)` (the floor stops division by zero).
2. Subtract the row maximum. This changes nothing mathematically but keeps `exp` from overflowing on large logits.
3. Exponentiate and divide by the row sum.

Works on a single vector or a batch of rows (`axis=-1`). The REINFORCE and imitation trainers call it on whole batches of logits.

```python
import numpy as np
from hunger_games.brain.neural import softmax
print(softmax(np.array([2.0, 1.0, 0.0])))        # [0.665 0.245 0.090]
print(softmax(np.array([2.0, 1.0, 0.0]), 0.5))   # [0.867 0.117 0.016]  sharper
print(softmax(np.array([2.0, 1.0, 0.0]), 5.0))   # [0.402 0.329 0.269]  flatter
```

### `class NeuralBrain(Brain)`

Perception vector to hidden layers to one score per action.

#### `name = "neural"`

The registry key and the results CSV label.

#### `__init__(self, chaos: float = 0.0, config: NeuralConfig | None = None, genome: np.ndarray | None = None, rng: np.random.Generator | None = None) -> None`

- `chaos`: stored by the base class, used as the softmax temperature.
- `config`: a `NeuralConfig`, or `None` for `NeuralConfig()` (two hidden layers of 64 and 32, `tanh`, `xavier_uniform`, `init_scale 0.05`, `sparsity 0.1`).
- `genome`: optional flat vector to load straight away.
- `rng`: the generator for the starting weights.

It sets `self.layer_sizes = [VECTOR_SIZE, *config.hidden_layers, MENU_SIZE]`, so the input width is always 50 and the output width always 16, with whatever hidden layers the config asks for. Then it builds `self.network = MLP(layer_sizes, activation, initializer, init_scale, sparsity, rng)`.

Two pieces of memory for the tools: `self.last_probabilities` (`None` until the first decision) and `self.last_index` (`0` until the first decision).

**Weight count.** `parameter_count` is the sum over layers of `n_in * n_out + n_out`.

| `hidden_layers` | Layer sizes | Parameters |
| --- | --- | --- |
| `(64, 32)` (default) | `50 -> 64 -> 32 -> 16` | `50*64 + 64 + 64*32 + 32 + 32*16 + 16 = 5872` |
| `(16,)` (the old default) | `50 -> 16 -> 16` | `50*16 + 16 + 16*16 + 16 = 1088` |
| `(32, 16)` | `50 -> 32 -> 16 -> 16` | `1600 + 32 + 512 + 16 + 256 + 16 = 2432` |
| `(8,)` | `50 -> 8 -> 16` | `400 + 8 + 128 + 16 = 552` |

The default moved from `(16,)` to `(64, 32)` because, measured under imitation pretraining, the bigger network copies the voting brain 80 percent of the time where the 16-neuron layer manages 64 percent. See [../config.md](../config.md).

#### `parameter_count` (property) `-> int`

Delegates to `self.network.parameter_count`.

#### `weights` (property) `-> list[np.ndarray]` and `biases` (property) `-> list[np.ndarray]`

Direct views of the network's lists, so the dashboard can draw them.

#### `genome(self) -> np.ndarray`

`self.network.genome()`: every weight and bias flattened, layer by layer. This is what the genetic algorithm evolves.

#### `set_genome(self, genome: np.ndarray) -> None`

`self.network.set_genome(genome)`. A wrong-length vector raises `ValueError` with both sizes in the message.

#### `forward(self, inputs: np.ndarray) -> np.ndarray`

`self.network.forward(inputs)`: one raw logit per menu item.

#### `probabilities(self, logits: np.ndarray) -> np.ndarray`

The chance of each menu item.

- `chaos <= 0`: a one-hot vector with `1.0` at the argmax. Deterministic.
- otherwise: `softmax(logits, chaos)`. Chaos 1.0 is a plain softmax; chaos 0.3 is sharp; chaos 3.0 is close to uniform.

This is where the neural brain's use of chaos differs from the voting brain's. The voting brain blends "certain" and "proportional" linearly; this brain uses chaos as a temperature.

#### `decide_index(self, perception: Perception, rng: np.random.Generator) -> int`

1. `logits = self.forward(perception.to_vector())`
2. `probabilities = self.probabilities(logits)`, stored in `self.last_probabilities`
3. With chaos 0 take the argmax; otherwise `rng.choice(MENU_SIZE, p=probabilities)`
4. Store in `self.last_index` and return it

The REINFORCE trainer needs this index, not the `Action`, because its loss is written in terms of the probability of the chosen menu item. It reads `brain.last_index` from a decision hook after every tick.

#### `decide(self, perception: Perception, rng: np.random.Generator) -> Action`

`self.menu_to_action(self.decide_index(perception, rng), perception)`. This is the method the game calls.

#### `menu_to_action(index: int, perception: Perception) -> Action` (static)

Turns a menu index into a concrete action, filling in targets from the perception.

- `index < 6`: `SIMPLE_ACTIONS[index]`.
- `index == ATTACK_INDEX`: if nobody is in sight, rest. If the nearest player is within `perception.reach`, `Action.attack(threat.player_id)`. Otherwise `Action.move(*threat.direction_toward())`.
- `index == FLEE_INDEX`: if nobody is in sight, rest. Otherwise `Action.flee(*threat.direction_away())`.
- `index >= 8`: `Action.move(*DIRECTIONS[index - FIRST_MOVE_INDEX])`.

It is public and static so training code and analysis scripts can translate recorded indices without a brain instance.

#### `action_to_menu_index(action: Action) -> int` (static)

The reverse of `menu_to_action`: which menu item an `Action` corresponds to. The imitation trainer calls it on every action the teacher returns, to make the label for that decision.

1. If `action.kind` is the kind of one of the `SIMPLE_ACTIONS`, return its position (`rest` 0 through `heal` 5).
2. `ATTACK` returns `ATTACK_INDEX` (6), whatever the target.
3. `FLEE` returns `FLEE_INDEX` (7), whatever the direction.
4. Otherwise, if `(action.dx, action.dy)` is in `DIRECTIONS`, return `FIRST_MOVE_INDEX + DIRECTIONS.index(...)`. So `Action.move(1, 0)` is 12.
5. Anything else, including a `MOVE` with `(0, 0)`, returns 0 (`rest`).

It is not a perfect inverse of `menu_to_action`, and cannot be: menu item 6 can become an attack, a move toward the nearest player, or a rest, depending on the perception. Going back, an attack maps to 6, but the approach move maps to a move item and the rest maps to 0. For the six simple actions and the eight moves the two functions are exact inverses, which `tests/test_imitation.py` checks.

#### `describe(self) -> str`

`self.network.describe()`, for example `"50 -> 64 -> 32 -> 16, tanh, xavier_uniform, 5872 params"`.

### The 50 inputs

`Perception.to_vector()` is the network's input, in this fixed order (names from `VECTOR_NAMES` in [perception.py](../perception.md)). Everything is scaled to roughly -1..1.

| Slots | Names | Notes |
| --- | --- | --- |
| 0-2 | thirst, hunger, health | bars, 1.0 full |
| 3-4 | survival score, training score | 0..1 |
| 5-6 | weapon quality, reach | reach divided by 3 |
| 7-8 | food carried, medkits carried | capped at 5 and 3, scaled to 1 |
| 9-10 | in water, hunt difficulty | flag, 0..1 |
| 11-12 | downhill dx, dy | -1, 0 or 1 |
| 13-15 | water dx, dy, distance | distance scaled by vision radius, 1.0 if unseen |
| 16-18 | grass dx, dy, distance | same |
| 19-21 | centre dx, dy, distance | distance 0 middle, 1 edge |
| 22-24 | loot here kind, qty, quality | kind divided by 3 |
| 25-28 | nearby loot dx, dy, distance, kind | |
| 29-33 | threat dx, dy, distance, level, health | zeros (distance 1.0) when alone |
| 34 | players in sight | count divided by 5, capped |
| 35-37 | in danger zone, hazard distance, hazard closing | distance clipped to -1..1 |
| 38-39 | safe dx, dy | |
| 40-41 | day fraction, alive fraction | the cannon tells everyone `alive fraction` |
| 42-45 | field known, field strength, strongest remaining, my rank | from the nightly sky, zeros and 0.5 when unknown |
| 46-49 | on water, on sand, on grass, on rock | one-hot terrain |

`VECTOR_SIZE` is 50 and the test `test_vector_size_matches_constant` guards it. Add a feature to the perception and this number, every saved genome and every trained champion file change with it.

### How the three trainers use this class

**Imitation.** The trainer keeps `NeuralBrain(chaos=0.0, config=...).network` as the student `MLP`. Demonstration games are played by the teacher through `create_brain`, and a decision hook records `perception.to_vector()` and `NeuralBrain.action_to_menu_index(action)` for every tribute. Training calls `policy.forward_cached(x)`, `softmax(logits)`, builds the gradient `(probs - one_hot) / batch`, then `policy.backward`. Validation games build learner brains with `chaos=0.0` (argmax) through `play_rl_episode`. `champion_brain(chaos)` returns a `NeuralBrain` loaded with the best genome.

**Genetic algorithm.** `create_brain("neural", chaos, rng, neural)` builds a fresh brain per tribute; the trainer reads `template.genome().size` to learn the genome length, fills a population with `genome()` from fresh brains (or from an `initial_genome` plus relatives), and loads each candidate with `set_genome` before a game. Chaos comes from the simulation config. The trainer never calls `forward` or looks at probabilities; only wins matter.

**REINFORCE.** The trainer keeps `NeuralBrain(chaos=1.0, config=...).network` as its policy `MLP`, optionally loaded from an `initial_genome`. During collection each learner gets a `NeuralBrain` with `chaos=1.0` (so sampling follows the plain softmax exactly) loaded with the current genome, and a decision hook records `perception.to_vector()` and `brain.last_index` every tick. During the update it calls `policy.forward_cached(states)`, `softmax(logits)` and builds the gradient at the logits, `(probs - one_hot) * advantage`, then `policy.backward`. Validation builds brains with `chaos=0.0` (argmax). `champion_brain(chaos)` returns a `NeuralBrain` loaded with the best genome.

## How to use it / experiment

**Play a game with untrained networks.**

```python
from hunger_games.config import SimulationConfig, NeuralConfig
from hunger_games.game import Game

config = SimulationConfig(brain_name="neural", chaos=0.5, seed=1, neural=NeuralConfig(hidden_layers=(32, 16), activation="relu", initializer="he_uniform"))
result = Game(config).run()
print(result.winner_name)
```

**Look at one decision.**

```python
import numpy as np
from hunger_games.brain.neural import NeuralBrain, MENU_NAMES

game = Game(SimulationConfig(width=60, height=60, seed=5))
player = game.players[0]
perception = player.perceive(game.arena, game.players, False, 0.0, 1.0, game.config.vision_radius)

brain = NeuralBrain(chaos=1.0, rng=np.random.default_rng(0))
index = brain.decide_index(perception, np.random.default_rng(0))
print(MENU_NAMES[index], brain.last_probabilities.round(3))
print(brain.menu_to_action(index, perception))
```

**Label a teacher's decision.**

```python
from hunger_games.brain import create_brain

teacher = create_brain("voting", 0.0, np.random.default_rng(0))
action = teacher.decide(perception, np.random.default_rng(0))
print(action, "->", MENU_NAMES[NeuralBrain.action_to_menu_index(action)])
```

**Pretrain it.** `ImitationTrainer(config, ImitationConfig())`, then `trainer.champion_brain()`; see [../training/imitation.md](../training/imitation.md).

**Evolve it.** `TrainingConfig(brain_name="neural")` with a `GeneticTrainer`, ideally with `initial_genome` from the student; see [../training/genetic.md](../training/genetic.md). Save with `save_champion` and reload with `load_champion`, then `NeuralBrain(config=data["neural"], genome=data["genome"])`.

**Train it with rewards.** `ReinforceTrainer(config, RLConfig(epochs=10), initial_genome=student.champion)`, then `trainer.champion_brain()`.

**Check the genome round-trips.** Straight from the tests:

```python
brain = NeuralBrain(config=NeuralConfig(hidden_layers=(32, 16)), rng=np.random.default_rng(0))
assert brain.layer_sizes == [50, 32, 16, 16]
assert brain.parameter_count == 2432 == brain.genome().size
doubled = brain.genome() * 2.0
brain.set_genome(doubled)
assert np.allclose(brain.genome(), doubled)
```

**Hand-wire a tiny policy.** Zero every weight, then set one output bias high, and the brain always picks that item at chaos 0. Useful for checking `menu_to_action` on its own. The last 16 entries of the genome are the output biases whatever the hidden layers are.

```python
brain = NeuralBrain(chaos=0.0, rng=np.random.default_rng(0))
g = np.zeros(brain.parameter_count)
g[-16 + 1] = 5.0          # item 1 is "drink"
brain.set_genome(g)
print(MENU_NAMES[brain.decide_index(perception, np.random.default_rng(0))])   # drink
```

## Gotchas

- The input width is fixed at `VECTOR_SIZE = 50` and the output at `MENU_SIZE = 16`. Only the hidden layers are configurable. A saved genome only fits a brain built with the same `NeuralConfig.hidden_layers`. Champion files from before the default changed to `(64, 32)` hold 1088 numbers and need `NeuralConfig(hidden_layers=(16,))`.
- `chaos` is a *temperature* here, not a blend. `chaos = 1.0` is the plain softmax, not "fully random". The REINFORCE trainer relies on exactly that.
- `probabilities()` at chaos 0 returns a one-hot even when several logits tie; the first maximum wins.
- `last_probabilities` and `last_index` describe the last call to `decide_index`, whichever tribute's perception that was. Each tribute has its own brain, so in a game this is fine. If you reuse one brain object for several players by hand, the values interleave.
- `menu_to_action` for `attack` can return a `MOVE` or a `REST`. If you count attacks from the recorded menu index you will overcount real attacks. Count from the resulting action instead.
- `action_to_menu_index` labels a teacher's approach move as a move, not as `attack`. A student trained by imitation learns `attack` only for attacks that were actually within reach, and learns the approach as ordinary moves.
- The network is untrained on construction. A neural roster with chaos 0 tends to repeat one action forever; use some chaos or a trained genome.
- `set_genome` shares memory with a float64 array you pass in (see [mlp.md](mlp.md) gotchas). The genetic trainer passes population entries directly; nothing writes in place, so this is safe today, but a brain that edited its weights in place would edit the population.
- Calling `forward` with a batch works, but `decide_index` and `probabilities` are written for one perception at a time (`probabilities` builds a length-16 one-hot at chaos 0).
