# `imitation.py`

**Source:** [hunger_games/training/imitation.py](../../hunger_games/training/imitation.py)
**Depends on:** `json`, `time`, `collections.abc.Callable`, `concurrent.futures.ProcessPoolExecutor`, `dataclasses` (`asdict`, `dataclass`, `field`), `pathlib` (standard library); `numpy`; [brain/__init__.py](../brain/init.md) (`Brain`, `create_brain`); [brain/mlp.py](../brain/mlp.md) (`Adam`); [brain/neural.py](../brain/neural.md) (`NeuralBrain`, `softmax`); [hunger_games/config.py](../config.md) (`SimulationConfig`); [hunger_games/game.py](../game.md) (`Game`); [hunger_games/recorder.py](../recorder.md) (`Recording`); [hunger_games/scenario.py](../scenario.md) (`Scenario`); [training/reinforce.py](reinforce.md) (`play_rl_episode`); [hunger_games/research/telemetry.py](../research/telemetry.md) (`BehaviorTelemetry`, imported inside `_validate`)
**Used by:** [training/__init__.py](init.md) (re-exports `ImitationConfig`, `ImitationStats`, `ImitationTrainer`); [training/runs.py](runs.md) (`save_run` reads `settings`, `config`, `history`, `history_rows()`, `champion`, `save_champion`); [hunger_games/ui/session.py](../ui/session.md) (`Session.start_training` builds an `ImitationTrainer` for the `"imitation"` method, `Session.save_champion` calls `save_champion`); [hunger_games/ui/app.py](../ui/app.md) (`ImitationConfig` backs the Train tab's imitation group and its accuracy chart); `tests/test_imitation.py`

## Purpose

This file pretrains the neural brain by copying the voting brain. The technique is called imitation learning, or behaviour cloning. Play games with a competent teacher, record every (perception, action) pair the teacher produces, and train the network to predict the teacher's action from the perception. That is ordinary supervised learning: a cross-entropy loss, mini-batches, Adam, and the `MLP`'s backpropagation.

**Why it exists.** A fresh network chooses actions at random. In the arena that means it usually dies of thirst on day three, before any fitness score or reward can teach it anything. Measured on the default config: of 12 learner deaths in a validation run, 10 were dehydration. Neither of the other trainers fixes this quickly. The genetic algorithm needs a genome that wins a game before selection has anything to select, and the policy gradient needs a reward that arrives before death, which is rare when death comes from a slowly draining bar.

Imitation gives the network working instincts first. Measured with a deterministic teacher (`teacher_chaos = 0.0`) and the default `(64, 32)` network: after 30 epochs the student picks the teacher's action 80 percent of the time, survives twice as long in validation games, and dehydration falls from 10 of 12 learner deaths to 2 of 12.

**The recommended flow.** Run imitation first. Then hand its champion to `GeneticTrainer` or `ReinforceTrainer` through their `initial_genome` argument (a "warm start"), and let those improve on it with real game outcomes. The dashboard's "start from the current champion" tick box does exactly that.

Every epoch logs training and validation loss and accuracy, then plays a greedy validation game, so survival, win rate, behaviour telemetry and a showcase recording are available exactly as they are for the other two trainers.

## Concepts you need

**Supervised learning.** You have inputs `x` and correct answers `y`, and you tune the network so its output for `x` matches `y`. Here `x` is the 50-number perception vector and `y` is the index of the menu item the teacher chose.

**Labels from actions.** The teacher returns an `Action` object, but the network outputs one score per item on a 16-item menu. `NeuralBrain.action_to_menu_index` maps an `Action` back to the menu index it corresponds to. It is the reverse of `menu_to_action`. See [../brain/neural.md](../brain/neural.md).

**Cross-entropy.** For one sample the loss is `-log p_y`, where `p_y` is the probability the network assigns to the correct answer. A confident correct answer costs nearly 0; a confident wrong answer costs a lot. A uniform guess over 16 items costs `ln 16 = 2.77`.

**The softmax gradient.** With logits `z`, `p = softmax(z)` and correct index `y`, the derivative of `-log p_y` with respect to `z_j` is `p_j - 1[j = y]`. In vector form that is `p - onehot(y)`. This is the same identity the REINFORCE trainer uses, minus the advantage. See [reinforce.md](reinforce.md).

**Accuracy.** The fraction of samples where the network's argmax equals the label. Chance level for 16 items is 1/16, about 6 percent.

**Train and validation split.** A slice of the demonstrations is held out and never trained on. The loss on that slice says whether the student generalises or has memorised. The champion is chosen by validation loss.

**Warm start.** Starting a trainer from an existing genome instead of random weights. Every trainer in this package accepts `initial_genome`.

**Process pools.** With `workers > 1`, demonstration games and validation games run in separate processes. The same `spawn` rules as the other trainers apply: top-level job functions, a `__main__` guard on macOS. See [genetic.md](genetic.md).

## Walkthrough

### `ImitationConfig`

```python
@dataclass
class ImitationConfig:
```

Every knob of the imitation learner.

| Field | Default | Meaning |
| --- | --- | --- |
| `teacher` | `"voting"` | Which brain to copy (any name `create_brain` accepts) |
| `teacher_chaos` | `0.0` | The chaos dial while the teacher demonstrates. 0 makes it pick its favourite action every time, which gives clean labels. A teacher that sometimes acts at random is hard to copy |
| `demonstration_games` | `12` | Teacher games recorded for demonstrations (12 games give about 40,000 decisions) |
| `epochs` | `30` | Passes over the demonstration data |
| `batch_size` | `256` | Samples per gradient step |
| `learning_rate` | `1e-3` | Adam step size |
| `validation_fraction` | `0.2` | Fraction of the demonstrations held out for validation loss and accuracy |
| `validation_games` | `1` | Greedy games the student plays per epoch on fixed seeds (survival, win rate, telemetry, showcase) |
| `validation_seed` | `90000` | The first validation seed (game `i` uses `validation_seed + i`) |
| `learners_per_game` | `6` | Tributes driven by the student in validation games; the rest use the config's brain |
| `workers` | `1` | CPU cores for collecting demonstrations and playing validation games |
| `seed` | `None` | The trainer's own seed (data shuffling, network init) |
| `record_showcase` | `True` | Whether to record the first validation game of each epoch for the dashboard's training feed |

Design reasoning: `teacher_chaos` is separate from `SimulationConfig.chaos` because demonstrations want a deterministic teacher (clean labels) while the validation games want the config's chaos for the opponents (a realistic test).

### `ImitationStats`

```python
@dataclass
class ImitationStats:
```

What happened in one epoch. Built by `step_epoch` and kept in `trainer.history`.

| Field | Type | Meaning |
| --- | --- | --- |
| `epoch` | `int` | Which epoch (0 first) |
| `train_loss` | `float` | Mean cross-entropy on the training demonstrations |
| `val_loss` | `float` | Mean cross-entropy on the held-out demonstrations |
| `train_accuracy` | `float` | Fraction of training demonstrations the student gets right |
| `val_accuracy` | `float` | Fraction of held-out demonstrations the student gets right |
| `val_survival` | `float` | Mean ticks the student survived in the validation games |
| `val_win_rate` | `float` | Fraction of validation games the student won |
| `seconds` | `float` | Seconds this epoch took |
| `cumulative_seconds` | `float` | Seconds since training started |
| `genome` | `np.ndarray` (default `None`, `repr=False`) | The student's genome after this epoch (a copy) |
| `telemetry` | `dict` (default `{}`, `repr=False`) | Merged behaviour telemetry from the validation games |
| `showcase` | `Recording | None` (default `None`, `repr=False`) | A recording of the first validation game, or `None` |

#### `to_row()`

```python
def to_row(self) -> dict:
```

Every field except `genome`, `telemetry` and `showcase`, as a plain dictionary. It walks `self.__dict__` and skips those three keys, the same way `GenerationStats.to_row` and `EpochStats.to_row` do. These rows become `history.json` and feed the plots.

### `collect_demonstration_game(...)`

```python
def collect_demonstration_game(config: SimulationConfig, scenario: Scenario | None, teacher: str, seed: int, teacher_chaos: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
```

Plays one game with the teacher brain and returns every (perception vector, menu index) pair. A top-level function so worker processes can run it.

1. Copy the config with `seed = seed` and `chaos = teacher_chaos` (via `to_dict_raw()`).
2. Define a `factory(index, rng)` that returns `create_brain(teacher, game_config.chaos, rng, game_config.neural, game_config.endgame_instinct)`. Every tribute is the teacher.
3. Build `Game(game_config, 0, brain_factory=factory, scenario=scenario)`.
4. Append a decision hook `on_decision(player, perception, action)` that stores `perception.to_vector()` in `vectors` and `NeuralBrain.action_to_menu_index(action)` in `labels`.
5. `game.run()`.
6. Return `np.asarray(vectors, dtype=float)` (N by 50) and `np.asarray(labels, dtype=int)` (N).

Every decision by every tribute is a sample, so one game with 24 tributes over 576 ticks gives up to 13,824 pairs, fewer as tributes die.

The label is the action the teacher actually returned. When the voting brain steps toward an enemy it returns a `MOVE`, which is labelled as one of the eight move items, not as `attack`. The `attack` label appears only when the teacher returned an `ATTACK` action. A `MOVE` with `(0, 0)` and any action outside the menu map to index 0, `rest`.

### `_demo_job(args)` and `_validation_job(args)`

```python
def _demo_job(args: tuple) -> tuple[np.ndarray, np.ndarray]:
def _validation_job(args: tuple) -> dict:
```

Tuple-unpacking wrappers for `ProcessPoolExecutor.map`. `_demo_job` forwards to `collect_demonstration_game`; `_validation_job` forwards to `play_rl_episode` from [reinforce.md](reinforce.md).

### `ImitationTrainer`

```python
class ImitationTrainer:
```

Trains a `NeuralBrain` to copy a teacher brain's decisions.

#### `__init__(config, imitation, scenario=None, initial_genome=None)`

```python
def __init__(self, config: SimulationConfig, imitation: ImitationConfig, scenario: Scenario | None = None, initial_genome: np.ndarray | None = None) -> None:
```

Stores `config`, `imitation` and `scenario`, and seeds `self.rng = np.random.default_rng(imitation.seed)`. The student is `NeuralBrain(chaos=0.0, config=config.neural, rng=self.rng).network`, an `MLP` of shape `[50, *hidden_layers, 16]` (5872 parameters for the default `(64, 32)`). If `initial_genome` is given it is loaded with `set_genome`, so a student can continue from an earlier champion. The optimiser is `Adam(self.policy, imitation.learning_rate)`.

Also sets up `train_x`, `train_y`, `val_x`, `val_y` (all `None` until `collect()`), `history`, `epoch = 0`, `_stop`, `_started`, `best_genome = None` and `best_val_loss = inf`.

#### `settings` (property)

```python
@property
def settings(self):
```

Returns `self.imitation`. Every trainer exposes this name so `save_run` can write the trainer's settings without knowing which trainer it has.

#### `_learner_ids()`

```python
def _learner_ids(self) -> list[int]:
```

Which tribute slots the student drives in validation games: `count = min(learners_per_game, num_players)` slots at `int(i * num_players / count)`. With 6 of 24 that is `[0, 4, 8, 12, 16, 20]`, the same spread the REINFORCE trainer uses.

#### `collect(on_progress=None)`

```python
def collect(self, on_progress: Callable[[int, int], None] | None = None) -> int:
```

Plays the teacher games and splits the demonstrations. Draws `demonstration_games` seeds from `self.rng`, builds one job `(config, scenario, teacher, seed, teacher_chaos)` per seed, and runs them through a pool when `workers > 1` and there is more than one job, else in sequence. `on_progress(done, total)` is called after each game. The results are concatenated, shuffled with `self.rng.permutation`, and cut at `split = int(len(x) * (1.0 - validation_fraction))`: the first part is training, the rest validation. Returns the total number of samples.

`step_epoch` calls this on the first epoch if `train_x` is still `None`. You can call it yourself first to see the sample count, or call it again later to draw a fresh set of demonstrations.

#### `_loss_and_accuracy(x, y)`

```python
def _loss_and_accuracy(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
```

Mean cross-entropy and accuracy of the student on a set. Returns `(0.0, 0.0)` for an empty set. Otherwise `probabilities = softmax(self.policy.forward(x))`, `loss = -log(probabilities[range(N), y] + 1e-12).mean()` and `accuracy = (probabilities.argmax(axis=1) == y).mean()`. The `1e-12` stops `log(0)`.

#### `step_epoch(on_progress=None)`

```python
def step_epoch(self, on_progress: Callable[[int, int], None] | None = None) -> ImitationStats:
```

One pass over the demonstrations, then validation.

1. If `train_x` is `None`, `collect(on_progress)`.
2. Start the clocks.
3. `order = self.rng.permutation(len(train_x))`, a fresh shuffle each epoch.
4. For each slice of `batch_size` indices: `logits, cache = self.policy.forward_cached(x)`, `probabilities = softmax(logits)`, then the gradient of the mean cross-entropy with respect to the logits:

   ```python
   grad = probabilities.copy()
   grad[np.arange(len(y)), y] -= 1.0
   grad /= len(y)
   ```

   That is `(p - onehot) / batch`. `MLP.backward` sums over the batch and never divides, so dividing here makes the result the gradient of the *mean* loss. Then `self.optimizer.step(self.policy.backward(cache, grad))`.
5. Score the full training set and the full validation set with `_loss_and_accuracy`.
6. If `val_loss < best_val_loss` or there is no best yet, store a copy of the current genome as `best_genome`.
7. `val_survival, val_win_rate, telemetry, showcase = self._validate()`.
8. Build `ImitationStats` (with a copy of the current genome), append it to `history`, increment `epoch`, return it.

The last mini-batch of an epoch is whatever is left, so it can be smaller than `batch_size`. With about 32,000 training samples and a batch of 256, an epoch is about 125 gradient steps.

#### `_validate()`

```python
def _validate(self) -> tuple[float, float, dict, Recording | None]:
```

Plays the student greedily on the fixed validation seeds. Returns `(0.0, 0.0, {}, None)` if `validation_games <= 0`. Otherwise one job per game: `(config, scenario, genome, learners, validation_seed + i, True, i == 0 and record_showcase)`, run through `play_rl_episode` (in a pool when `workers > 1` and there is more than one game). `True` is `greedy`, so the student's brains use chaos 0 and take the argmax; the opponents use the config's brain and chaos. Only the first game records.

From the episodes it takes every learner outcome, averages `survival` and `won`, merges the telemetry with `BehaviorTelemetry.merge`, and returns `episodes[0].get("recording")` as the showcase. The reward numbers `play_rl_episode` also computes are ignored here.

#### `run(on_epoch=None, on_progress=None)`

```python
def run(self, on_epoch: Callable[[ImitationStats], None] | None = None, on_progress: Callable[[int, int], None] | None = None) -> list[ImitationStats]:
```

Clears `_stop`, then loops `step_epoch(on_progress)` while `epoch < imitation.epochs` and not stopped, calling `on_epoch(stats)` after each. Returns `history`. A second `run()` continues from the current epoch on the same demonstrations.

#### `stop()`

```python
def stop(self) -> None:
```

Sets `_stop`; a running `run()` returns after the current epoch. The dashboard calls this from the UI thread.

#### `champion` (property)

```python
@property
def champion(self) -> np.ndarray | None:
```

`best_genome` if one has been stored, otherwise the current policy genome. The best is the epoch with the lowest validation loss, not the highest survival or win rate.

#### `champion_brain(chaos=0.0)`

```python
def champion_brain(self, chaos: float = 0.0) -> NeuralBrain:
```

A `NeuralBrain` built from `config.neural` with the champion loaded. Chaos 0 makes it greedy, the same way it was validated.

#### `save_policy(path)`

```python
def save_policy(self, path: str | Path) -> None:
```

Writes the champion in the same JSON shape as the other trainers' champion files.

| Key | Value |
| --- | --- |
| `brain_name` | `"neural"` |
| `neural` | `asdict(config.neural)` |
| `genome` | The champion as a list |
| `fitness` | `-best_val_loss` if finite, else `0.0` (negated so that, as in every champion file, higher is better) |
| `epochs` | `len(history)` |
| `method` | `"imitation"` |
| `teacher` | `imitation.teacher` |

`GeneticTrainer.load_champion` reads it back, and the dashboard's "Load champion into all" accepts it.

#### `save_champion(path)`

```python
def save_champion(self, path: str | Path) -> None:
```

An alias of `save_policy`, so every trainer has the same method name. `save_run` and `Session.save_champion` call this one.

#### `history_rows()`

```python
def history_rows(self) -> list[dict]:
```

`[stats.to_row() for stats in self.history]`. No genomes, telemetry or recordings.

## How to use it / experiment

**Pretrain, then warm-start.** The whole recommended flow in one script:

```python
from hunger_games.config import SimulationConfig
from hunger_games.training import (
    GeneticTrainer, ImitationConfig, ImitationTrainer, ReinforceTrainer, RLConfig, TrainingConfig, save_run,
)

config = SimulationConfig(seed=0)
student = ImitationTrainer(config, ImitationConfig(seed=0))
student.run(on_epoch=lambda s: print(s.epoch, f"acc {s.val_accuracy:.2f}", f"loss {s.val_loss:.3f}", f"survival {s.val_survival:.0f}"))
save_run(student, "imitation", "student")

# Evolve from the student. A small mutation scale keeps its instincts.
ga = GeneticTrainer(config, TrainingConfig(brain_name="neural", mutation_scale=0.02, seed=0), initial_genome=student.champion)
ga.run()

# Or reinforce from the student.
rl = ReinforceTrainer(config, RLConfig(seed=0), initial_genome=student.champion)
rl.run()
```

**Read the numbers.**

| Metric | Healthy trend |
| --- | --- |
| `train_loss` | Falls from about 2.77 (uniform) and keeps falling |
| `val_loss` | Falls, then flattens. Rising while `train_loss` still falls is overfitting; more demonstration games help more than more epochs |
| `train_accuracy`, `val_accuracy` | Rise together. About 0.8 after 30 epochs with the defaults |
| `val_survival` | Rises early, then tracks accuracy loosely. This is the number that says the instincts work |
| `val_win_rate` | Usually stays near 0. With 6 learners in a 24-tribute game against the voting brain, winning is rare, and copying the teacher cannot beat the teacher |

**Watch an epoch.** Each `ImitationStats.showcase` is a recording of the first validation game, played greedily with the student in the `_learner_ids()` slots:

```python
from hunger_games.renderer import export_recording_gif
export_recording_gif(student.history[-1].showcase, "student.gif")
```

In the dashboard, set the training feed to "replay" and each epoch's game loads as the arena frees up.

**Copy a chaotic teacher.** Set `teacher_chaos=0.5` to see the labels get noisy: accuracy drops even though the loss still falls, because the same perception now has several "correct" answers.

**Continue from a champion.** `ImitationTrainer(config, ImitationConfig(), initial_genome=GeneticTrainer.load_champion("champion.json")["genome"])` starts the student from a saved genome of the same architecture.

**Use the cores.** `ImitationConfig(workers=4)` parallelises the demonstration games and the validation games, not the gradient steps. The script needs a `__main__` guard on macOS.

## Gotchas

- **Validation loss picks the champion.** A later epoch with better survival but a slightly higher validation loss is not the champion. `history[-1].genome` is the latest student if you want it instead.
- **`validation_fraction=0.0` freezes the champion at epoch 0.** With an empty validation set `_loss_and_accuracy` returns `0.0`, which beats `inf` on the first epoch and never beats `0.0` again. Keep a held-out slice.
- **`demonstration_games` must be at least 1.** With 0 games `collect()` calls `np.concatenate` on an empty list and raises `ValueError`.
- **Demonstrations are collected once.** `run()` twice trains twice on the same data. Call `collect()` again for fresh games.
- **Data size grows with the arena.** Each decision is a 50-float row. Twelve default games are about 40,000 rows, which is small. A big map with `max_days=48` and 48 tributes multiplies that; keep `demonstration_games` modest there.
- The student can only be as good as the teacher on the states the teacher visits. A deterministic teacher never explores odd situations, so the student is unsure in them. That is the distribution-shift problem of behaviour cloning, and it is why the recommended flow hands the champion to a trainer that learns from real outcomes.
- The showcase is a *greedy validation* game, not a training game. That differs from the REINFORCE trainer, whose showcase is a sampled training game.
- `val_win_rate` is a fraction over learner outcomes, so with 6 learners and 1 validation game it can only be 0 or about 0.17.
- The student's `NeuralBrain` is built with chaos 0, but only its `.network` is kept and training uses `softmax` at temperature 1 on the raw logits. The chaos value has no effect on training.
- `play_rl_episode` builds learners with `NeuralBrain(...)` directly, so `config.endgame_instinct` does not apply to the student in validation games. It does apply to the teacher during demonstrations, through `create_brain`.
- The same `spawn` rules as the other trainers apply when `workers > 1`: a `__main__` guard, a script file, and pickle-able `config` and `scenario`. The demonstration arrays are pickled back from the workers, and the first validation game's `Recording` too.
- Showcases stay in memory in `history` until the trainer is dropped. Set `record_showcase=False` for long runs nobody is watching.
- An `initial_genome` must match `config.neural`. A genome saved with `hidden_layers=(16,)` will not load into a student built with the default `(64, 32)`; `set_genome` raises `ValueError`.
