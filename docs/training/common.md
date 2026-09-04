# `common.py`

**Source:** [hunger_games/training/common.py](../../hunger_games/training/common.py)
**Depends on:** `time`, `dataclasses`, `typing.Any` (standard library); `numpy`; [brain/base.py](../brain/base.md) (`Brain`); [brain/neural.py](../brain/neural.md) (`NeuralBrain`); [hunger_games/config.py](../config.md) (`NeuralConfig`); [hunger_games/recorder.py](../recorder.md) (`Recording`); [brain/neat.py](../brain/neat.md) and [brain/voting.py](../brain/voting.md) (imported inside `build_learner` to avoid an import cycle); `psutil` (imported inside `SystemMonitor.__init__`, with a fallback when it is missing). `psutil>=5` is listed in `requirements.txt` and in the `dependencies` of `pyproject.toml`, so a normal `pip install` brings it in.
**Used by:** [training/genetic.py](genetic.md), [training/imitation.py](imitation.md), [training/neat.py](neat.md), [training/reinforce.py](reinforce.md) and through it [training/ppo.py](ppo.md) (every trainer fills `IterationStats`, keeps an `EventLog`, accepts a `Curriculum`, builds `LearnerSpec`s and calls `learner_ids`); [training/runs.py](runs.md) (`learning_history` rows and `events.events`); [training/__init__.py](init.md) (re-exports `Curriculum`, `CurriculumConfig`, `EventLog`, `IterationStats`, `LearnerSpec`, `SystemMonitor`); [research/comparison.py](../research/comparison.md) (`Curriculum`, `CurriculumConfig`, `LearnerSpec`, `learner_ids`); [ui/session.py](../ui/session.md) (`Curriculum`, `CurriculumConfig`, `SystemMonitor`, `events.tail`, `learning_history`); [ui/app.py](../ui/app.md) (`CurriculumConfig` backs the curriculum settings); `tests/test_methods.py`

## Purpose

Five training methods live in this package: imitation, the genetic algorithm, NEAT, REINFORCE and PPO. They change the learner in different ways. They do not differ in how an iteration is scored or reported. This file holds the pieces they share:

- `IterationStats`, the one record every method fills in after each iteration. The dashboard's graphs, `learning.json` and the shared learning-curve plots all read this shape.
- `EventLog`, the timestamped one-line messages behind the dashboard's event monitor.
- `CurriculumConfig` and `Curriculum`, which grow the number of opponents as the learner improves. This is modelled on the zombie video's ladder, where the agent faced 1, then 2, 4, 8 and 16 zombies. Since version 0.7.0 a stage is cleared by winning games, not by scoring points, and there is no timeout by default.
- `SystemMonitor`, CPU and memory readings for the dashboard.
- `LearnerSpec` and `build_learner`, a small description of a learner brain that a worker process can rebuild.
- `learner_ids`, the rule for which tribute slots the learner takes.

Every method here trains one learner network against opponents that use the voting brain from the video. The shared pieces are what let the dashboard, `save_run` and the method comparison treat all five trainers the same way.

## Concepts you need

**Iteration.** One round of a trainer: an epoch for imitation, REINFORCE and PPO, a generation for the genetic algorithm and NEAT. Every trainer's `step()` runs one iteration and returns an `IterationStats`.

**Score and episode return.** Every trainer scores the learner by its episode return, the total reward it collected in one game under `RewardConfig` (see [../config.md](../config.md)). That is what `scores`, `mean_score`, `best_score` and `val_score` hold. The genetic algorithm in `"self"` mode is the one exception; there the scores are placement fitness.

**Entropy.** `H = -sum p(a) log p(a)` in nats. High means the learner is still exploring, low means it is confident. A uniform choice over the 16-item menu has `H = ln 16 = 2.77`.

**Win rate, game-level.** The learner plays several copies of itself in one game (6 by default). A game has one victor, so at most one copy can win it. A game counts as won when any learner copy was the victor. `win_rate` and `val_win_rate` are fractions of games won, not fractions of learner copies that won. The old per-copy count capped six copies at one sixth; the game-level rate can reach 1.0.

**Curriculum.** Start easy and get harder. Here "easy" means few opponents. By default the learner is promoted when it has won at least half of its recent validation games. It can instead be judged on mean score. A timeout can be added, but there is none by default: the learner has to earn each stage.

**Worker processes.** With `workers > 1` the trainers play games in separate processes. A process cannot receive a live `Brain` object cheaply, so it receives a `LearnerSpec` (a kind plus a genome) and rebuilds the brain with `build_learner`.

**Genome shapes.** A neural learner's genome is a flat numpy array. A NEAT learner's genome is a dictionary from `NeatGenome.to_dict()`. A voting learner's genome is its eight genes. `LearnerSpec.kind` says which.

## Walkthrough

### `IterationStats`

```python
@dataclass
class IterationStats:
```

One iteration of any method, in the same shape for every method.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `iteration` | `int` | required | Which iteration (0 first) |
| `scores` | `list[float]` | required | The score of every learner episode this iteration (a return under the reward function) |
| `mean_score` | `float` | required | Their mean |
| `best_score` | `float` | required | The best of them |
| `entropy` | `float` | required | Policy entropy in nats |
| `mean_length` | `float` | required | Mean ticks the learner survived this iteration |
| `win_rate` | `float` | required | Fraction of training games a learner copy won (game-level: one victor per game) |
| `val_score` | `float` | required | Mean score in the greedy validation games on fixed seeds |
| `seconds` | `float` | required | Seconds this iteration took |
| `cumulative_seconds` | `float` | required | Seconds since training started |
| `val_win_rate` | `float` | `0.0` | Fraction of validation games a learner copy won (game-level) |
| `stage` | `int` | `0` | Curriculum stage index |
| `opponents` | `int` | `23` | Opponents faced |
| `extra` | `dict` | `{}` | Method-specific numbers (losses, species counts, accuracy, and so on) |
| `learner` | `Any` | `None` (`repr=False`) | The learner after this iteration: a genome array for neural brains, a NEAT genome dictionary for NEAT |
| `telemetry` | `dict` | `{}` (`repr=False`) | Behaviour telemetry summary of the learner's episodes |
| `showcase` | `Recording | None` | `None` (`repr=False`) | A recording of one real episode from this iteration (the dashboard's training feed) |

`val_win_rate` has a default so older code that builds the record by position still works. Every trainer fills it.

What goes in `extra` depends on the trainer:

| Trainer | `extra` keys |
| --- | --- |
| `GeneticTrainer` | `worst_fitness` |
| `ImitationTrainer` | `train_loss`, `val_loss`, `train_accuracy`, `val_accuracy` |
| `NeatTrainer` | `species`, `hidden_nodes`, `connections`, `threshold` |
| `ReinforceTrainer`, `PPOTrainer` | `policy_loss`, `value_loss` |

#### `to_row()`

```python
def to_row(self) -> dict:
```

A JSON-friendly dictionary without the big arrays. It walks `self.__dict__` and skips `learner`, `telemetry`, `showcase`, `scores` and `extra`. Then it flattens the extras in with an `extra_` prefix, so a NEAT row has `extra_species` and `extra_hidden_nodes`, and an RL row has `extra_policy_loss`. `val_win_rate` is a plain field, so it is in every row. `save_run` writes these rows to `learning.json`, `learning_curve_plots` reads them, and the method comparison reads `val_win_rate` from them for its win criterion and its win-rate charts.

### `EventLog`

```python
class EventLog:
```

Timestamped one-line messages about what training is doing. The dashboard's event monitor shows the tail of it.

#### `__init__(capacity=500)`

```python
def __init__(self, capacity: int = 500) -> None:
```

Starts with an empty `events` list (newest last), remembers `capacity`, and records `started = time.time()` so stamps are relative to when the log was made. Every trainer makes its log in its constructor, so stamps count from the trainer's creation, not from the first `step()`.

#### `add(kind, message)`

```python
def add(self, kind: str, message: str) -> None:
```

Appends one line of the form `[{stamp:7.1f}s] {kind:<10} {message}`, where `stamp` is seconds since the log started. The kind is padded to ten characters so the messages line up. The kinds the trainers use are `"rollout"`, `"evolution"`, `"curriculum"`, `"record"` and `"info"`. When the list grows past `capacity`, the oldest entries are dropped.

Example line: `[   12.4s] curriculum promoted to stage 1: 3 opponents`.

#### `tail(count=20)`

```python
def tail(self, count: int = 20) -> list[str]:
```

The most recent `count` events. `Session.training_events` calls this with 14 for the dashboard.

### `Stage`

```python
@dataclass
class Stage:
```

One lesson of a curriculum.

| Field | Default | Meaning |
| --- | --- | --- |
| `name` | required | A label for events and charts (`"survive"`, `"beat 3"`, `"generalise"`) |
| `opponents` | required | Voting opponents in every game of the stage; the learner's copies are added on top |
| `overrides` | `{}` | Dotted `SimulationConfig` overrides applied to every game of the stage, e.g. `{"gamemaker_enabled": False}` |
| `variants` | `()` | Override sets one of which is picked at random per training episode (the generalisation lesson) |
| `metric` | `"win_rate"` | What promotion is judged on: `"win_rate"`, `"survival"` (share of the game the copies stayed alive) or `"score"` |
| `threshold` | `0.5` | Promote when the last `window` iterations of that metric average at least this |

### `apply_overrides(config, overrides)`, `stage_config(base, stage, learners)`, `episode_config(config, stage, seed)`

`apply_overrides` copies a config with dotted overrides applied through its dictionary form (`to_dict`, set the keys, `from_dict`), so nested settings (`neural.hidden_layers`) and enums given as strings (`layout: "cornucopia"`, `shape: "round"`) both work. `stage_config` is the config every game of a stage is played on: `num_players = min(learners, 24) + stage.opponents`, then the stage's overrides. `episode_config` is the config of one training episode: the stage config with one of the stage's `variants` chosen by `np.random.default_rng(seed)`, so the same seed always gives the same rules; a stage without variants returns the config itself. Validation games never go through `episode_config`.

### `CurriculumConfig`

```python
@dataclass
class CurriculumConfig:
```

| Field | Default | Meaning |
| --- | --- | --- |
| `enabled` | `True` | Whether the curriculum is on |
| `opponents` | `(1, 3, 7, 11, 23)` | Opponents per stage (the learner's own copies are added on top) |
| `stages` | `None` | Explicit lessons; when given they replace the `opponents` ladder |
| `promote_on` | `"win_rate"` | What promotion is judged on: `"win_rate"` (the learner must actually win games) or `"score"` |
| `win_threshold` | `0.5` | With `promote_on="win_rate"`, promote when the last `window` win rates average at least this (a majority of games) |
| `threshold` | `3.0` | With `promote_on="score"`, promote when the last `window` mean scores average at least this |
| `window` | `5` | Iterations averaged for the promotion test |
| `max_iterations_per_stage` | `0` | Promote anyway after this many iterations in a stage; `0` means never |

#### `lessons(win_threshold=0.5, survival_threshold=0.6, rules_survival_threshold=0.3, window=5)` (classmethod)

The lesson curriculum, eight `Stage`s: `survive` (0 opponents, circle and sponsors off, promoted on a survival share of at least 0.6), `survive the rules` (0 opponents, circle and sponsors on, survival share of at least 0.3), `beat 1`, `beat 3`, `beat 7`, `beat 11`, `beat 23` (wins), and `generalise` (23 opponents with one of five variants per episode: cornucopia layout, ring layout, round arena, circle off, sponsors off; wins). The reason is in [../research/README.md](../research/README.md): every cold start of the first full experiment died of thirst before it met its single opponent. The two survival bars are measured, not chosen: six copies of the imitation champion survive 65 to 78 percent of a no-opponent game (they fight each other and hunger bites over 24 days) and 30 to 40 percent once the circle is on (the circle ends the game by driving them together), so the bars ask a learner to survive as long as the teacher would. A bar of 0.9, the first choice, was unreachable even for the teacher.

Design reasoning: the zombie video's ladder was 1, 2, 4, 8, 16 zombies. This arena has 24 tributes by default, and the learner keeps 6 copies of itself in every game, so the ladder here is 1, 3, 7, 11 and 23 opponents. The last stage is the full default roster of 23 opponents. Judging on wins rather than score means a stage is cleared only when the learner beats that field, not when it collects points while losing. The default of no timeout means a learner that never wins stays on stage 0; that is a result, not a failure of the curriculum.

### `Curriculum`

```python
class Curriculum:
```

Tracks the current stage and decides when to promote.

#### `__init__(config)`

```python
def __init__(self, config: CurriculumConfig) -> None:
```

Starts at `stage = 0` with `iterations_in_stage = 0` and an empty `recent` list. `recent` holds the judged metric of the recent iterations in this stage: whatever the current `Stage.metric` names.

#### `stages` (property), `stage_spec` (property), `describe()`

`stages` is the tuple of lessons: `config.stages` when given, else one `Stage("beat N", N)` per entry of `config.opponents`, judged on wins with `win_threshold` (or on score with `threshold` when `promote_on="score"`). `stage_spec` is the current lesson, the last one when the curriculum is off, clamped at the end. `describe()` is `"beat 7 (7 opponents)"` for events.

#### `opponents` (property)

```python
@property
def opponents(self) -> int:
```

Opponents in the current stage: `stage_spec.opponents`. When the curriculum is off that is the last lesson's count (the hardest stage), and a stage index past the end still reads the last lesson.

#### `finished` (property)

```python
@property
def finished(self) -> bool:
```

`True` when the curriculum is off or `stage` is the last index of `stages`. A finished curriculum never promotes again. The method comparison reads this to decide whether a variant is at the final stage before it tests the win criterion.

#### `observe(mean_score, win_rate=0.0, survival=0.0)`

```python
def observe(self, mean_score: float, win_rate: float = 0.0, survival: float = 0.0) -> bool:
```

Records one iteration's mean score, win rate and survival share (ticks survived divided by `ticks_per_game`, 0 to 1). Returns `True` when the learner is promoted. The rule:

1. If `finished`, return `False` without counting anything.
2. Add one to `iterations_in_stage`. Append the metric the current lesson names (`win_rate`, `survival` or `mean_score`) to `recent` and keep only the last `window` entries.
3. `good_enough` is true when `recent` has at least `window` entries and their mean is at least the lesson's `threshold` (for the classic ladder that is `win_threshold`, or `threshold` when judging on score).
4. `timed_out` is true when `max_iterations_per_stage` is greater than 0 and `iterations_in_stage` has reached it. With the default `0` it is never true.
5. If either holds, add one to `stage`, reset `iterations_in_stage` to 0 and `recent` to empty, and return `True`.

The metrics that are not being judged are ignored. With the default ladder, `mean_score` and `survival` play no part in promotion; in a survival lesson, wins play no part.

Worked example with `opponents=(1, 3, 7)`, `win_threshold=0.5`, `window=2`, `max_iterations_per_stage=4` (the settings `tests/test_methods.py` uses):

| Call | `recent` after | `iterations_in_stage` | Result |
| --- | --- | --- | --- |
| `observe(0.0, win_rate=0.0)` | `[0.0]` | 1 | `False` (only one entry in the window) |
| `observe(0.0, win_rate=1.0)` | `[0.0, 1.0]`, mean 0.5 | 2 | `True`, now 3 opponents |
| `observe(9.0, win_rate=0.0)` three times | `[0.0, 0.0]` | 3 | `False` each time; the high score is ignored |
| `observe(9.0, win_rate=0.0)` | `[0.0, 0.0]` | 4 | `True` by timeout, now 7 opponents, `finished` |

Two more cases from the same test. With `max_iterations_per_stage=0`, fifty calls of `observe(0.0, win_rate=0.0)` never promote. With `promote_on="score"`, `threshold=1.0` and `window=1`, one call of `observe(2.0, win_rate=0.0)` promotes at once.

Who passes what:

| Trainer | Call after each iteration |
| --- | --- |
| `ReinforceTrainer`, `PPOTrainer` | `observe(mean_score, val_win_rate, survival)` when `validation_games > 0`, else with `win_rate`; `survival` is the training games' mean ticks over `ticks_per_game` |
| `NeatTrainer` | The same rule with its own `validation_games` |
| `GeneticTrainer` | The same rule with its own `validation_games` |
| `ImitationTrainer` | Never calls `observe` |

Every caller logs a `"curriculum"` event with `describe()` when `observe` returns `True`, and rebuilds its config with `stage_config` at the start of the next iteration. Note that the thresholds are compared directly with the mean of the window; there is no per-stage scaling in the code.

### `SystemMonitor`

```python
class SystemMonitor:
```

CPU, memory and GPU readings for the dashboard.

#### `__init__()`

```python
def __init__(self) -> None:
```

Tries to import `psutil` once. If it is there, it is kept as `self.psutil` and `psutil.cpu_percent(interval=None)` is called once to prime the counter, because the first call always returns 0. If the import fails, `self.psutil` is `None`.

#### `read()`

```python
def read(self) -> dict:
```

The current readings, always with the same four keys:

| Key | With psutil | Without psutil |
| --- | --- | --- |
| `cpu_percent` | `psutil.cpu_percent(interval=None)` as a float | `0.0` |
| `memory_mb` | This process's resident memory in megabytes | `0.0` |
| `memory_percent` | System memory in use, percent | `0.0` |
| `gpu` | `"not used (numpy on the CPU)"` | the same |

The `gpu` value is a fixed string. Everything in this project runs in numpy on the CPU.

### `champion_key(stage, val_win_rate, val_score)`

```python
def champion_key(stage: int, val_win_rate: float, val_score: float) -> tuple[int, float, float]:
```

The order in which every trainer compares candidate champions: the tuple `(stage, val_win_rate, val_score)`. Python compares tuples element by element, so the curriculum stage decides first, then the validation win rate, then the validation score. A validation score against one opponent is not comparable with one against seven, which is why the stage comes first: a policy from an easier rung can never displace one that played at a harder rung. REINFORCE and PPO, the genetic algorithm and NEAT all pick their champion with this key; imitation has no curriculum and keeps its lowest-validation-loss epoch.

### `LearnerSpec`

```python
@dataclass
class LearnerSpec:
```

How to rebuild a learner brain in a worker process: its kind and its genome.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `kind` | `str` | required | `"neural"` (a `NeuralBrain` genome array) or `"neat"` (a NEAT genome dictionary); `build_learner` also accepts `"voting"` |
| `genome` | `Any` | required | The genome: a flat array for neural, a dict for NEAT, eight genes for voting |
| `neural` | `NeuralConfig | None` | `None` | The neural architecture (neural only) |

Every trainer has `learner_spec()` (the current learner) and `champion_spec()` (the best so far), both returning one of these. The method comparison keeps `champion_spec()` of every variant for its tournament.

### `build_learner(spec, chaos, rng)`

```python
def build_learner(spec: LearnerSpec, chaos: float, rng: np.random.Generator) -> Brain:
```

Builds the brain a spec describes:

- `kind == "neat"`: `NeatBrain(NeatGenome.from_dict(spec.genome), chaos=chaos)`.
- `kind == "voting"`: `VotingBrain(chaos=chaos, genome=np.asarray(spec.genome, dtype=float))`.
- anything else: `NeuralBrain(chaos=chaos, config=spec.neural, rng=rng)` with `set_genome(spec.genome)`.

The NEAT and voting imports happen inside the function to avoid an import cycle. `play_rl_episode` in [reinforce.md](reinforce.md) calls this for every learner slot, with chaos 1.0 while training and 0.0 while validating.

### `learner_ids(num_players, learners)`

```python
def learner_ids(num_players: int, learners: int) -> list[int]:
```

Evenly spreads learner slots across the roster, so learners are not all neighbours on the starting podiums. `count = max(1, min(learners, num_players))`, and slot `i` is `int(i * num_players / count)`.

| `num_players` | `learners` | Result |
| --- | --- | --- |
| 24 | 6 | `[0, 4, 8, 12, 16, 20]` |
| 7 | 6 | `[0, 1, 2, 3, 4, 5]` |
| 24 | 1 | `[0]` |
| 4 | 6 | `[0, 1, 2, 3]` |

Every trainer's `_learner_ids()` calls this with its own `learners_per_game`. The tournament in `research/comparison.py` calls it directly.

## How to use it / experiment

**Read the shared history of any trainer.**

```python
from hunger_games.config import SimulationConfig
from hunger_games.training import PPOConfig, PPOTrainer

trainer = PPOTrainer(SimulationConfig(width=60, height=60, max_days=4), PPOConfig(epochs=3, seed=0))
for _ in range(3):
    stats = trainer.step()
    print(stats.iteration, round(stats.mean_score, 2), round(stats.val_win_rate, 2), stats.opponents)
print(trainer.events.tail(5))
```

The same loop works for `ImitationTrainer`, `GeneticTrainer`, `NeatTrainer` and `ReinforceTrainer`, because `step()` returns the same `IterationStats`.

**Turn on the curriculum.**

```python
from hunger_games.training import Curriculum, CurriculumConfig, ReinforceTrainer, RLConfig

curriculum = Curriculum(CurriculumConfig(opponents=(1, 3, 7, 11, 23), win_threshold=0.5, window=5))
trainer = ReinforceTrainer(config, RLConfig(epochs=100, seed=0), curriculum=curriculum)
trainer.run()
print([s.opponents for s in trainer.learning_history])
```

Each trainer's `_apply_curriculum` resizes the roster before every iteration to `min(learners_per_game, 24) + curriculum.opponents` players. With 6 learners the stages are games of 7, 9, 13, 17 and 29 tributes. The `curriculum.png` chart in a run folder shows the ladder.

**Tune the promotion rule.** A lower `win_threshold` or a shorter `window` promotes sooner. `promote_on="score"` with `threshold` brings back the old score-based rule. `max_iterations_per_stage` is an optional safety net: set it above 0 and a stuck learner still moves on after that many iterations. Leave it at 0 when the point of the experiment is whether the learner can earn each stage.

**Rebuild a learner by hand.**

```python
import numpy as np
from hunger_games.training.common import build_learner

brain = build_learner(trainer.champion_spec(), chaos=0.0, rng=np.random.default_rng(0))
print(brain.describe())
```

**Watch the machine.** `SystemMonitor().read()` once a second gives the numbers the dashboard shows. `psutil` is a declared dependency (`psutil>=5` in `requirements.txt` and `pyproject.toml`), so after `pip install -r requirements.txt` the readings are real. The zeros in the "without psutil" column only appear in a stripped-down environment that skipped it; the dashboard keeps working either way.

## Gotchas

- `to_row()` drops `scores` and `extra` as fields but adds the extras back as `extra_*` keys. The per-episode scores are only in memory; `learning.json` has their mean and best.
- Every trainer that applies the curriculum (genetic, neat, reinforce, ppo) passes the validation win rate to `observe` when it plays validation games, so all of them can be promoted by winning; with `validation_games=0` the training-game win rate is used instead.
- With the default `max_iterations_per_stage=0` there is no timeout at all. A learner that never wins half of its validation games stays on stage 0 for the whole run, and `finished` stays `False`. The method comparison's win criterion is only tested at the final stage, so such a variant runs its full iteration budget.
- The judged win rate comes from validation games when the trainer has any (`validation_games > 0`), else from training games. With `validation_games=2` a single win moves the window's mean by 0.1 (one of two games, averaged over five iterations), so the win rate is coarse. Raise `validation_games` for a steadier signal.
- The curriculum only counts when `finished` is false. With `enabled=False`, `opponents` is the last stage and `observe` always returns `False`.
- A curriculum object holds state. Make a fresh `Curriculum` for every trainer; sharing one between two trainers would let one promote the other.
- `IterationStats.opponents` defaults to 23, but the trainers always pass a value: the curriculum's `opponents`, or `num_players - 1` without a curriculum. `ImitationTrainer` always records `stage=0` and `num_players - 1` opponents, because it does not apply the curriculum to its roster.
- `learner_ids` never returns more slots than `num_players`. With a curriculum stage of 1 opponent and 6 learners the roster is 7 tributes, and all 6 learners play.
- `build_learner` builds neural learners with `NeuralBrain(...)` directly, not `create_brain`, so `config.endgame_instinct` does not apply to learners. Voting learners are built without the endgame flag too.
- `LearnerSpec.kind` is not validated. A typo such as `"nueral"` falls through to the neural branch, and a NEAT dictionary there raises inside `np.asarray` or `set_genome`.
- `EventLog` stamps are relative to when the log was created, which is when the trainer was created. A trainer that sits idle before `run()` has a gap at the start.
- `SystemMonitor.read()` measures `cpu_percent` since the previous call. Two calls in quick succession give a near-zero reading.
