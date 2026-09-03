# `imitation.py`

**Source:** [hunger_games/training/imitation.py](../../hunger_games/training/imitation.py)
**Depends on:** `json`, `time`, `collections.abc.Callable`, `concurrent.futures.ProcessPoolExecutor`, `dataclasses` (`asdict`, `dataclass`, `field`), `pathlib` (standard library); `numpy`; [brain/__init__.py](../brain/init.md) (`Brain`, `create_brain`); [brain/mlp.py](../brain/mlp.md) (`Adam`); [brain/neural.py](../brain/neural.md) (`MENU_SIZE`, `NeuralBrain`, `softmax`); [hunger_games/config.py](../config.md) (`SimulationConfig`); [hunger_games/game.py](../game.md) (`Game`); [hunger_games/recorder.py](../recorder.md) (`Recording`); [hunger_games/scenario.py](../scenario.md) (`Scenario`); [training/common.py](common.md) (`Curriculum`, `EventLog`, `IterationStats`, `LearnerSpec`, `learner_ids`); [training/reinforce.py](reinforce.md) (`play_rl_episode`); [hunger_games/research/telemetry.py](../research/telemetry.md) (`BehaviorTelemetry`, imported inside `_validate`)
**Used by:** [training/__init__.py](init.md) (re-exports `ImitationConfig`, `ImitationStats`, `ImitationTrainer`); [training/runs.py](runs.md) (`save_run`); [research/comparison.py](../research/comparison.md) (the `"imitation"` method, usually as the warm start for the others); [hunger_games/ui/session.py](../ui/session.md) (`Session.start_training` builds an `ImitationTrainer` for the `"imitation"` method, the dashboard's default); [hunger_games/ui/app.py](../ui/app.md) (`ImitationConfig` backs the Train tab's imitation group); `tests/test_methods.py`; `tests/test_imitation.py`

## Purpose

This file pretrains the neural brain by copying the voting brain. The technique is called imitation learning, or behaviour cloning. Play games with a competent teacher, record every (perception, action) pair the teacher produces, and train the network to predict the teacher's action from the perception. That is ordinary supervised learning: a cross-entropy loss, mini-batches, Adam, and the `MLP`'s backpropagation.

**Why it exists.** A fresh network chooses actions at random. In the arena that means it usually dies of thirst on day three, before any fitness score or reward can teach it anything. Imitation gives the network working instincts first. Measured with a deterministic teacher and the default `(64, 32)` network: after 30 epochs the student picks the teacher's action 80 percent of the time, survives twice as long in validation games, and dehydration falls from 10 of 12 learner deaths to 2 of 12.

**Learning from winners.** `winners_top` keeps only the decisions of tributes that placed well in their demonstration game. That is the "show it a few winning games" idea: fewer samples, but from the tributes whose instincts worked.

**The recommended flow.** Run imitation first. Then hand its champion to any other trainer through `initial_genome` (a "warm start"). The dashboard's "start from the current champion" tick box does exactly that, and the method comparison's `warm_from` does it in scripts.

Every epoch logs training and validation loss and accuracy, plays greedy validation games, and appends the shared `IterationStats` to `learning_history`, so the dashboard and the shared learning curves work exactly as for the other trainers.

## Concepts you need

**Supervised learning.** You have inputs `x` and correct answers `y`, and you tune the network so its output for `x` matches `y`. Here `x` is the 50-number perception vector and `y` is the index of the menu item the teacher chose.

**Labels from actions.** The teacher returns an `Action` object, but the network outputs one score per item on a 16-item menu. `NeuralBrain.action_to_menu_index` maps an `Action` back to its menu index.

**Cross-entropy.** For one sample the loss is `-log p_y`. A uniform guess over 16 items costs `ln 16 = 2.77`.

**The softmax gradient.** With logits `z`, `p = softmax(z)` and correct index `y`, the derivative of `-log p_y` with respect to `z_j` is `p_j - 1[j = y]`, or `p - onehot(y)` in vector form. This is the same identity the REINFORCE trainer uses, minus the advantage.

**Accuracy.** The fraction of samples where the network's argmax equals the label. Chance level for 16 items is about 6 percent.

**Train and validation split.** A slice of the demonstrations is held out and never trained on. The champion is chosen by validation loss.

**Placement.** `result.players[i].placement` is 1 for the victor, 2 for the runner-up, and so on. `winners_top=3` keeps the decisions of the three best-placed tributes of each demonstration game.

## Walkthrough

### `ImitationConfig`

```python
@dataclass
class ImitationConfig:
```

| Field | Default | Meaning |
| --- | --- | --- |
| `teacher` | `"voting"` | Which brain to copy (any name `create_brain` accepts) |
| `teacher_chaos` | `0.0` | The chaos dial while the teacher demonstrates. 0 makes it pick its favourite action every time, which gives clean labels |
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
| `winners_top` | `0` | Learn only from tributes that placed this well or better in their demonstration game (0 = everyone) |

### `ImitationStats`

```python
@dataclass
class ImitationStats:
```

The trainer's own record per epoch, kept in `trainer.history`.

| Field | Type | Meaning |
| --- | --- | --- |
| `epoch` | `int` | Which epoch (0 first) |
| `train_loss` | `float` | Mean cross-entropy on the training demonstrations |
| `val_loss` | `float` | Mean cross-entropy on the held-out demonstrations |
| `train_accuracy` | `float` | Fraction of training demonstrations the student gets right |
| `val_accuracy` | `float` | Fraction of held-out demonstrations the student gets right |
| `val_survival` | `float` | Mean ticks the student survived in the validation games |
| `val_win_rate` | `float` | Fraction of validation learner outcomes that won |
| `seconds` | `float` | Seconds this epoch took |
| `cumulative_seconds` | `float` | Seconds since training started |
| `genome` | `np.ndarray` (default `None`, `repr=False`) | The student's genome after this epoch (a copy) |
| `telemetry` | `dict` (default `{}`, `repr=False`) | Merged behaviour telemetry from the validation games |
| `showcase` | `Recording | None` (default `None`, `repr=False`) | A recording of the first validation game, or `None` |

#### `to_row()`

```python
def to_row(self) -> dict:
```

Every field except `genome`, `telemetry` and `showcase`. These rows become `history.json`.

### `collect_demonstration_game(...)`

```python
def collect_demonstration_game(config: SimulationConfig, scenario: Scenario | None, teacher: str, seed: int, teacher_chaos: float = 0.0, winners_top: int = 0) -> tuple[np.ndarray, np.ndarray]:
```

Plays one game with the teacher brain and returns every (perception vector, menu index) pair, optionally keeping only the decisions of tributes that placed in the top `winners_top`. A top-level function so worker processes can run it.

1. Copy the config with `seed = seed` and `chaos = teacher_chaos`.
2. Every tribute is `create_brain(teacher, game_config.chaos, rng, game_config.neural, game_config.endgame_instinct)`.
3. A decision hook stores `perception.to_vector()` in `vectors`, `NeuralBrain.action_to_menu_index(action)` in `labels`, and `player.player_id` in `owners`.
4. `result = game.run()`.
5. Build `x` (N by 50) and `y` (N). If `winners_top > 0`, `keep_ids` is the set of player ids whose `placement` satisfies `0 < placement <= winners_top`, and only rows whose owner is in that set are kept.
6. Return `(x, y)`.

One default game gives up to 13,824 pairs; with `winners_top=3` it gives the decisions of three tributes, which are the ones who lasted longest, so still a good share.

### `_demo_job(args)` and `_validation_job(args)`

```python
def _demo_job(args: tuple) -> tuple[np.ndarray, np.ndarray]:
def _validation_job(args: tuple) -> dict:
```

Tuple-unpacking wrappers for `ProcessPoolExecutor.map`. `_demo_job` forwards to `collect_demonstration_game`; `_validation_job` to `play_rl_episode`.

### `ImitationTrainer`

```python
class ImitationTrainer:
```

#### `__init__(config, imitation, scenario=None, initial_genome=None, curriculum=None)`

```python
def __init__(self, config: SimulationConfig, imitation: ImitationConfig, scenario: Scenario | None = None, initial_genome: np.ndarray | None = None, curriculum: Curriculum | None = None) -> None:
```

Stores `config`, `imitation`, `curriculum` and `scenario`; makes `events = EventLog()`, an empty `learning_history` and `best_mean_score = -inf`. Seeds `self.rng` from `imitation.seed`. The student is `NeuralBrain(chaos=0.0, config=config.neural, rng=self.rng).network`. If `initial_genome` is given it is loaded with `set_genome`. The optimiser is `Adam(self.policy, imitation.learning_rate)`. Also `train_x`, `train_y`, `val_x`, `val_y` (all `None` until `collect()`), `history`, `epoch = 0`, `_stop`, `_started`, `best_genome = None` and `best_val_loss = inf`.

The `curriculum` argument is accepted so every trainer has the same constructor, and stored on `self.curriculum`. No method in this file reads it: the roster of the validation games is always `config.num_players`, and the shared record always says `stage=0`.

#### `settings` (property)

```python
@property
def settings(self):
```

Returns `self.imitation`.

#### `_learner_ids()`

```python
def _learner_ids(self) -> list[int]:
```

`learner_ids(config.num_players, imitation.learners_per_game)`; with 6 of 24 that is `[0, 4, 8, 12, 16, 20]`.

#### `learner_spec()` and `champion_spec()`

```python
def learner_spec(self) -> LearnerSpec:
def champion_spec(self) -> LearnerSpec:
```

`LearnerSpec("neural", genome, config.neural)` with a copy of the current student, or the champion.

#### `step(on_progress=None)`

```python
def step(self, on_progress: Callable[[int, int], None] | None = None) -> IterationStats:
```

Runs `step_epoch` and returns `learning_history[-1]`.

#### `collect(on_progress=None)`

```python
def collect(self, on_progress: Callable[[int, int], None] | None = None) -> int:
```

Draws `demonstration_games` seeds, builds one job `(config, scenario, teacher, seed, teacher_chaos, winners_top)` per seed, runs them (through a pool when `workers > 1`), concatenates, shuffles, and cuts at `int(len(x) * (1 - validation_fraction))` into training and validation sets. Logs an `"info"` event (`collected N demonstrations from G teacher games`). Returns the total number of samples.

`step_epoch` calls this on the first epoch if `train_x` is still `None`.

#### `_loss_and_accuracy(x, y)`

```python
def _loss_and_accuracy(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
```

Mean cross-entropy and accuracy of the student on a set; `(0.0, 0.0)` for an empty set.

#### `step_epoch(on_progress=None)`

```python
def step_epoch(self, on_progress: Callable[[int, int], None] | None = None) -> ImitationStats:
```

1. Collect if needed, start the clocks.
2. A fresh shuffle; for each mini-batch, `logits, cache = policy.forward_cached(x)`, `grad = (softmax(logits) - onehot) / batch`, backward, Adam step.
3. Score the full training and validation sets.
4. If `val_loss` beats `best_val_loss` (or no best yet), store a copy of the genome as `best_genome`.
5. `val_survival, val_win_rate, telemetry, showcase, val_returns = self._validate()`.
6. Build `ImitationStats`, append it to `history`, increment `epoch`.
7. Build the shared `IterationStats`: `scores` are the validation returns (`mean_score` their mean, `best_score` their max), `entropy` is the student's mean entropy on the held-out demonstrations (uniform if there are none), `mean_length = val_survival`, `win_rate = val_win_rate`, `val_score` the same mean return, `stage=0`, `opponents = num_players - 1`, `extra={"train_loss", "val_loss", "train_accuracy", "val_accuracy"}`, `learner = stats.genome`. Append it to `learning_history`.
8. Log a `"rollout"` event (`epoch E: accuracy A, loss L, validation score V`) and a `"record"` event when `mean_score` beats `best_mean_score`.
9. Return the `ImitationStats`.

Because `scores` come from validation games, `mean_score` and `val_score` are the same number here.

#### `_validate()`

```python
def _validate(self) -> tuple[float, float, dict, Recording | None, list[float]]:
```

Plays the student greedily on the fixed validation seeds. Returns `(0.0, 0.0, {}, None, [])` if `validation_games <= 0`. Otherwise one `play_rl_episode` job per game, passing the student's genome as a plain array (the episode player wraps it in a neural `LearnerSpec`), `greedy=True`, and `record` only for the first game. From every learner outcome it averages `survival` and `won`, merges the telemetry, and returns the first episode's recording and the list of every learner's `return`. The returns are what fill the shared `scores`.

#### `run(on_epoch=None, on_progress=None)`

```python
def run(self, on_epoch: Callable[[ImitationStats], None] | None = None, on_progress: Callable[[int, int], None] | None = None) -> list[ImitationStats]:
```

Clears `_stop`, loops `step_epoch` while `epoch < imitation.epochs` and not stopped. Returns `history`.

#### `stop()`

```python
def stop(self) -> None:
```

Sets `_stop`; `run()` returns after the current epoch.

#### `champion` (property)

```python
@property
def champion(self) -> np.ndarray | None:
```

`best_genome` if one has been stored, otherwise the current policy genome. The best is the epoch with the lowest validation loss.

#### `champion_brain(chaos=0.0)`

```python
def champion_brain(self, chaos: float = 0.0) -> NeuralBrain:
```

A `NeuralBrain` with the champion loaded.

#### `save_policy(path)` and `save_champion(path)`

```python
def save_policy(self, path: str | Path) -> None:
def save_champion(self, path: str | Path) -> None:
```

Writes JSON with `brain_name` (`"neural"`), `neural`, `genome`, `fitness` (`-best_val_loss` if finite, else `0.0`), `epochs`, `method` (`"imitation"`) and `teacher`. `save_champion` is an alias.

#### `history_rows()`

```python
def history_rows(self) -> list[dict]:
```

`[stats.to_row() for stats in self.history]`.

## How to use it / experiment

**Pretrain, then warm-start.**

```python
from hunger_games.config import SimulationConfig
from hunger_games.training import ImitationConfig, ImitationTrainer, PPOConfig, PPOTrainer, save_run

config = SimulationConfig(seed=0)
student = ImitationTrainer(config, ImitationConfig(seed=0))
student.run(on_epoch=lambda s: print(s.epoch, f"acc {s.val_accuracy:.2f}", f"survival {s.val_survival:.0f}"))
save_run(student, "imitation", "student")

trainer = PPOTrainer(config, PPOConfig(seed=0), initial_genome=student.champion)
trainer.run()
```

**Learn from winners only.** `ImitationConfig(winners_top=3, demonstration_games=24)` keeps the three best-placed tributes' decisions per game. Double the games to keep the sample count up; `tests/test_methods.py` checks that `collect()` returns fewer samples with `winners_top=3` than without.

**Read the numbers.**

| Metric | Healthy trend |
| --- | --- |
| `train_loss` | Falls from about 2.77 and keeps falling |
| `val_loss` | Falls, then flattens. Rising while `train_loss` still falls is overfitting |
| `train_accuracy`, `val_accuracy` | Rise together. About 0.8 after 30 epochs with the defaults |
| `val_survival` (and `mean_length` in the shared record) | Rises early. This is the number that says the instincts work |
| `mean_score` in the shared record | The validation episode return; comparable with the other methods' curves |

**Copy a chaotic teacher.** `teacher_chaos=0.5` makes the labels noisy: accuracy drops even though the loss still falls.

**Use the cores.** `ImitationConfig(workers=4)` parallelises the demonstration and validation games, not the gradient steps. The script needs a `__main__` guard on macOS.

## Gotchas

- **Validation loss picks the champion.** A later epoch with better survival but a slightly higher validation loss is not the champion. `history[-1].genome` is the latest student.
- **`validation_fraction=0.0` freezes the champion at epoch 0.** With an empty validation set `_loss_and_accuracy` returns `0.0`, which beats `inf` once and never again.
- **`demonstration_games` must be at least 1.** With 0 games `collect()` calls `np.concatenate` on an empty list and raises.
- **`winners_top` can empty a game.** If no tribute has a placement in the range (for example every game ends in a draw at the day cutoff with placements above `winners_top`), that game contributes zero rows. With every game empty, `collect()` fails on the concatenation or the split.
- **Demonstrations are collected once.** `run()` twice trains twice on the same data. Call `collect()` again for fresh games.
- The curriculum is stored but never applied. Pass it for uniformity, not for effect.
- `validation_games=0` gives an empty `scores` list, so `mean_score`, `best_score` and `val_score` are `0.0` and the shared score chart is flat.
- The showcase is a *greedy validation* game, not a training game.
- `val_win_rate` is a fraction over learner outcomes, so with 6 learners and 1 validation game it can only be 0 or about 0.17.
- The student's `NeuralBrain` is built with chaos 0, but only its `.network` is kept and training uses `softmax` at temperature 1 on the raw logits.
- `play_rl_episode` builds learners through `build_learner`, so `config.endgame_instinct` does not apply to the student in validation games. It does apply to the teacher during demonstrations, through `create_brain`.
- The same `spawn` rules as the other trainers apply when `workers > 1`.
- An `initial_genome` must match `config.neural`; `set_genome` raises `ValueError` otherwise.
