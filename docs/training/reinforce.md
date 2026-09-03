# `reinforce.py`

**Source:** [hunger_games/training/reinforce.py](../../hunger_games/training/reinforce.py)
**Depends on:** `json`, `time`, `collections.abc.Callable`, `concurrent.futures.ProcessPoolExecutor`, `dataclasses`, `pathlib` (standard library); `numpy`; [brain/__init__.py](../brain/init.md) (`Brain`, `create_brain`); [brain/mlp.py](../brain/mlp.md) (`MLP`, `Adam`); [brain/neural.py](../brain/neural.md) (`NeuralBrain`, `softmax`); [hunger_games/config.py](../config.md) (`SimulationConfig`, and through it `RewardConfig`); [hunger_games/game.py](../game.md) (`Game`); [hunger_games/perception.py](../perception.md) (`VECTOR_SIZE`); [hunger_games/player.py](../player.md) (`Player`); [hunger_games/recorder.py](../recorder.md) (`Recorder`, `Recording`); [hunger_games/research/telemetry.py](../research/telemetry.md) (`BehaviorTelemetry`); [hunger_games/scenario.py](../scenario.md) (`Scenario`); [training/common.py](common.md) (`Curriculum`, `EventLog`, `IterationStats`, `LearnerSpec`, `build_learner`, `learner_ids`)
**Used by:** [training/__init__.py](init.md) (re-exports `EpochStats`, `ReinforceTrainer`, `RLConfig`); [training/ppo.py](ppo.md) (`PPOTrainer` subclasses `ReinforceTrainer`, `PPOConfig` subclasses `RLConfig`); [training/imitation.py](imitation.md) (`play_rl_episode` plays the student's greedy validation games); [training/neat.py](neat.md) (`play_rl_episode` scores every genome); [training/genetic.py](genetic.md) (`_run_episode_job` in `"voting"` mode); [research/comparison.py](../research/comparison.md) (`_run_episode_job` for the tournament, and the `"reinforce"` method); [training/runs.py](runs.md) (`save_run`); [hunger_games/ui/session.py](../ui/session.md) (`ReinforceTrainer` with `initial_genome` and `curriculum`); [hunger_games/ui/app.py](../ui/app.md) (`RLConfig`); [experiments/run_rl.py](../experiments/run_rl.md); `tests/test_methods.py`; `tests/test_research.py`; `tests/test_feed.py`; `tests/test_imitation.py`

## Purpose

This file trains the neural brain by reinforcement learning. The genetic algorithm in [genetic.md](genetic.md) scores whole games. This trainer scores every action: after each tick a learning tribute gets a reward built from the weights in `RewardConfig`, and the policy network is nudged to make well-rewarded actions more likely. A second network, the value network, predicts how much reward is still to come from a state. Subtracting that prediction (the "baseline") makes the learning signal far less noisy. This is REINFORCE with a learned baseline, the simplest actor-critic method, written in plain numpy on top of the `MLP` class.

It also holds `play_rl_episode`, the episode player every other trainer borrows. It puts any learner described by a `LearnerSpec` (neural, NEAT or voting) into the learner slots, plays a game against the config's brain, and returns the experience, the outcomes, a game-level `learner_won` flag and the telemetry. The imitation, genetic and NEAT trainers and the method comparison's tournament all score with it.

Everything a researcher asks for is logged per epoch in `EpochStats`, and the shared `IterationStats` is appended to `learning_history` with events and the curriculum. Win rates are game-level: a game is won when any learner copy was the victor. The curriculum is promoted on validation wins. The constructor accepts an `initial_genome`, so the policy can start from a network that already has instincts. The recommended flow is to pretrain by imitation ([imitation.md](imitation.md)) and then reinforce from that champion.

## Concepts you need

**Policy.** A function from state to a probability for each action. Here the state is the 50-value perception vector (`VECTOR_SIZE`), the actions are the neural brain's 16-item menu (`MENU_SIZE`), and the policy is the `MLP` inside a `NeuralBrain` followed by a softmax.

**Episode, return, discount.** One game is an episode. Each learner collects a list of rewards, one per decision. The return at step `t` is `G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...`, with `gamma = RewardConfig.discount = 0.98`.

**Policy gradient.** To make action `a` more likely in state `s`, increase `log p(a | s)`. REINFORCE multiplies that push by how good the outcome was. The loss for one sample is `-log p(a) * A` where `A` is the advantage.

**Baseline and advantage.** `A = G - V(s)`, the return minus what the value network predicted. Actions that did better than expected get pushed up, worse than expected pushed down.

**Entropy.** `H = -sum_a p(a) log p(a)`, in nats. A uniform policy over 16 actions has `H = ln 16 = 2.77`; a certain policy has `H = 0`. An entropy bonus keeps the policy exploring.

**Shaping.** A dense reward that pays out for progress toward a goal. `RewardConfig.approach` is one: a bonus per cell moved closer to water while thirsty. It is off by default (`0.0`).

**Greedy versus sampling.** During training the learner samples from the softmax at temperature 1 (`chaos=1.0`). During validation it takes the argmax (`chaos=0.0`).

**Game-level win.** Six learner copies share one game and only one tribute can be the victor. `play_rl_episode` reports `learner_won`, true when any copy won. `win_rate` and `val_win_rate` are the fraction of games with `learner_won`, so they can reach 1.0. The per-copy `won` flags are still in `outcomes`.

**Hooks.** `Game.decision_hooks` are called right after a brain decides; `Game.tick_hooks` at the end of each tick. Most of the reward is computed in a tick hook and attached to the most recent decision.

**Learner spec.** A `LearnerSpec` is a kind plus a genome. `build_learner` turns it into a brain inside the worker. See [common.md](common.md).

## Walkthrough

### `RLConfig`

```python
@dataclass
class RLConfig:
```

| Field | Default | Meaning |
| --- | --- | --- |
| `epochs` | `30` | Rounds of collect-then-update `run()` performs |
| `episodes_per_epoch` | `4` | Games played per epoch to collect experience |
| `learners_per_game` | `6` | Tributes per game driven by the policy (the rest use the config's brain) |
| `learning_rate` | `1e-3` | Adam step size for the policy network |
| `value_learning_rate` | `3e-3` | Adam step size for the value network |
| `entropy_bonus` | `0.01` | Weight `beta` of the entropy term |
| `value_hidden` | `(32,)` | Hidden layer widths of the value network |
| `validation_games` | `2` | Games per epoch with the greedy policy on fixed seeds |
| `validation_seed` | `90000` | First validation seed (game `i` uses `validation_seed + i`) |
| `max_grad_norm` | `5.0` | Gradients with a larger combined length are scaled down |
| `workers` | `1` | CPU cores for collecting episodes |
| `seed` | `None` | The trainer's own seed |
| `record_showcase` | `True` | Whether to record the first training game of every epoch for the dashboard's training feed |

The reward weights and the discount live in `SimulationConfig.reward` (a `RewardConfig`, see [../config.md](../config.md)) because the reward is a property of the game, not of the learner. `PPOConfig` in [ppo.md](ppo.md) extends this dataclass.

### `EpochStats`

```python
@dataclass
class EpochStats:
```

| Field | Meaning |
| --- | --- |
| `epoch` | Which epoch (0 first) |
| `policy_loss` | `-(mean log p(a) * A) - beta * mean H` on the batch, before the update |
| `value_loss` | Mean squared error between value predictions and returns |
| `entropy` | Mean policy entropy at the decisions made (nats) |
| `train_return` | Mean total reward per learner episode during collection |
| `val_return` | Mean total reward per learner episode on the validation seeds, greedy |
| `train_survival` | Mean ticks survived by learners during collection |
| `val_survival` | Mean ticks survived on validation |
| `win_rate` | Fraction of training games won by a learner copy (game-level, from `_outcome_means`) |
| `val_win_rate` | Same, validation games |
| `kill_rate` | Mean kills per learner episode, training |
| `seconds` | Wall-clock seconds this epoch took |
| `cumulative_seconds` | Seconds since training started |
| `genome` | The policy genome after this epoch's update (a copy, `repr=False`) |
| `telemetry` | Merged telemetry of this epoch's training episodes (`repr=False`) |
| `showcase` | A `Recording` of this epoch's first training game, or `None` (`repr=False`) |

The field comments in the source still read "fraction of learner episodes that won". The values stored come from `_outcome_means`, which counts games since version 0.7.0.

#### `to_row()`

```python
def to_row(self) -> dict:
```

Every field except `genome`, `telemetry` and `showcase`, for `history.json` and the method's own plots.

### `play_rl_episode(...)`

```python
def play_rl_episode(config: SimulationConfig, scenario: Scenario | None, genome, learner_ids: list[int], seed: int, greedy: bool, record: bool = False) -> dict:
```

Plays one game with the learner driving `learner_ids` and returns their experience. A top-level function so worker processes can run it.

**The `genome` argument.** It may be a flat neural genome array (the common case) or a `LearnerSpec` of any kind. The function does `spec = genome if isinstance(genome, LearnerSpec) else LearnerSpec("neural", genome, game_config.neural)`. So the imitation trainer passes a plain array, while the REINFORCE, genetic and NEAT trainers and the tournament pass specs.

**Setup.** Copies the config with the episode `seed`. The brain factory gives each learner slot `build_learner(spec, 0.0 if greedy else 1.0, rng)` and keeps it in `learners` so its `last_index` can be read; every other slot gets `create_brain(config.brain_name, config.chaos, ...)`. A `BehaviorTelemetry` tracks only the learner ids. Three per-learner lists are made (`vectors`, `indices`, `rewards`), plus `previous` (bars and kills per learner), `dead_paid` and `last_distances`.

**Decision hook `on_decision(player, perception, action)`.** Ignores non-learners. For a learner: first the approach shaping (if the learner was thirsty and water was visible before and after, add `reward.approach * (water_before - water_now)` to the *previous* decision's reward; likewise grass while hungry), then update `last_distances`, then append `perception.to_vector()`, the brain's `last_index`, and a placeholder `0.0` reward.

**Tick hook `on_tick(current)`.** For every learner, with `prev = (thirst, hunger, health, kills)` from the previous tick:

| Term | When | Amount |
| --- | --- | --- |
| `survive_tick` (`0.01`) | Alive | `+ survive_tick` |
| `damage_taken` (`-2.0`) | Alive, health fell | `+ damage_taken * max(0, prev_health - health)` |
| `need_gain` (`0.5`) | Alive, previous thirst below 0.5, thirst rose | `+ need_gain * max(0, thirst - prev_thirst)` |
| `need_gain` (`0.5`) | Alive, previous hunger below 0.5, hunger rose | `+ need_gain * max(0, hunger - prev_hunger)` |
| `death` (`-3.0`) | Not alive and not yet paid | `+ death`, once |
| `kill` (`1.0`) | Any | `+ kill * (kills - prev_kills)` |
| `approach` (`0.0`) | Computed in the decision hook | `+ approach * cells closer to water (thirsty) or grass (hungry)` |

The reward is added to the latest decision, provided the learner has decided at least once.

**Play.** `Recorder(game).record_all()` when `record` is on, then `game.run()` either way (a no-op if the recorder already finished the game).

**End of game.** Each learner gets `reward.placement * (n - placement) / (n - 1)` (first place earns the full `2.0`), plus `reward.win` (`5.0`) if it is the sole survivor, added to its last decision. Its outcome records `won = 1` in that case.

**Return value.** A dict with:

| Key | Contents |
| --- | --- |
| `learner_won` | `True` when any learner copy's `won` is 1: the game-level win flag |
| `vectors`, `indices`, `rewards` | `{pid: np.ndarray}` per learner |
| `outcomes` | `{pid: {"return", "survival", "won", "kills"}}` per learner copy |
| `telemetry` | The telemetry summary |
| `recording` | The `Recording`, or `None` |

`learner_won` is what every trainer and the tournament use for win rates. The per-copy `won` flags stay in `outcomes` for anyone who wants the old per-copy count.

### `_run_episode_job(args)`

```python
def _run_episode_job(args: tuple) -> dict:
```

Unpacks a tuple for `ProcessPoolExecutor.map` and calls `play_rl_episode`. The genetic trainer and the method comparison import this directly.

### `ReinforceTrainer`

```python
class ReinforceTrainer:
```

#### `__init__(config, rl, scenario=None, initial_genome=None, curriculum=None)`

```python
def __init__(self, config: SimulationConfig, rl: RLConfig, scenario: Scenario | None = None, initial_genome: np.ndarray | None = None, curriculum: Curriculum | None = None) -> None:
```

Stores `config`, `rl`, `scenario` and `curriculum`; makes `events = EventLog()`, an empty `learning_history` and `best_mean_score = -inf`. Seeds `self.rng` from `rl.seed`. The policy is `NeuralBrain(chaos=1.0, config=config.neural, rng=self.rng).network`, an `MLP` of shape `[50, *hidden_layers, 16]`. If `initial_genome` is given it is loaded with `set_genome`. The value network is `MLP([VECTOR_SIZE, *rl.value_hidden, 1], config.neural.activation, "xavier_uniform", rng=self.rng)` and always starts fresh. Each gets its own `Adam`. Also `history`, `epoch = 0`, `_stop`, `_started`, `best_genome = None` and `best_val_return = -inf`.

#### `settings` (property)

```python
@property
def settings(self):
```

Returns `self.rl`.

#### `_learner_ids()`

```python
def _learner_ids(self) -> list[int]:
```

`learner_ids(config.num_players, rl.learners_per_game)`; with 6 of 24 that is `[0, 4, 8, 12, 16, 20]`.

#### `learner_spec()` and `champion_spec()`

```python
def learner_spec(self) -> LearnerSpec:
def champion_spec(self) -> LearnerSpec:
```

`LearnerSpec("neural", genome, config.neural)` with a copy of the current policy genome, or the champion.

#### `_apply_curriculum()`

```python
def _apply_curriculum(self) -> None:
```

With a curriculum, copies the config with `num_players = min(learners_per_game, 24) + curriculum.opponents`. Called at the start of every epoch.

#### `_record_iteration(scores, entropy, mean_length, win_rate, val_score, seconds, extra, telemetry, showcase, val_win_rate=0.0)`

```python
def _record_iteration(self, scores: list[float], entropy: float, mean_length: float, win_rate: float, val_score: float, seconds: float, extra: dict, telemetry: dict, showcase, val_win_rate: float = 0.0) -> IterationStats:
```

Appends a unified `IterationStats` with `iteration = len(learning_history)`, the scores and their mean and max, `win_rate` and `val_win_rate` as given, `cumulative_seconds` as the previous record's plus `seconds`, the curriculum's `stage` and `opponents` (or `num_players - 1`), `learner = learner_spec().genome`, and the rest as given. Logs a `"rollout"` event (`iteration I: mean score M, length L ticks, win rate W`) and a `"record"` event when the mean beats `best_mean_score`.

Then the curriculum. `judged_win` is `val_win_rate` when `rl.validation_games > 0`, else the training `win_rate`. It calls `curriculum.observe(mean_score, judged_win)` and logs a `"curriculum"` event on promotion. With the default `CurriculumConfig` that means the policy climbs a stage when it has won at least half of its greedy validation games over the last five epochs. Returns the stats. `PPOTrainer` inherits this method unchanged.

#### `step(on_progress=None)`

```python
def step(self, on_progress: Callable[[int, int], None] | None = None) -> IterationStats:
```

Runs `step_epoch` and returns `learning_history[-1]`. The dashboard and the method comparison call this.

#### `_collect(seeds, greedy, on_progress=None, record_first=False)`

```python
def _collect(self, seeds: list[int], greedy: bool, on_progress: Callable[[int, int], None] | None = None, record_first: bool = False) -> list[dict]:
```

Takes `spec = self.learner_spec()` once, builds one job `(config, scenario, spec, learners, seed, greedy, record)` per seed with `record` only for the first when asked, and runs them through a pool when `workers > 1` and there is more than one job. Returns the episode dicts in seed order.

#### `_returns(rewards)`

```python
def _returns(self, rewards: np.ndarray) -> np.ndarray:
```

Discounted returns, walking backwards with `running = rewards[t] + discount * running`. For `[0, 0, 5]` and discount 0.98: `[4.80, 4.90, 5.0]`.

#### `_update(episodes)`

```python
def _update(self, episodes: list[dict]) -> tuple[float, float, float]:
```

One gradient step on both networks. Returns `(policy_loss, value_loss, entropy)`.

1. Gather every learner trajectory: vectors, indices and `_returns(rewards)`. Return `(0.0, 0.0, 0.0)` if empty.
2. `values, value_cache = self.value.forward_cached(states)`.
3. `advantages = returns - values`, normalised to mean 0 and standard deviation 1.
4. `logits, policy_cache = self.policy.forward_cached(states)`; `probs`, `log_probs`, per-sample `entropy`.
5. The gradient at the logits: `(probs - one_hot) * advantages[:, None] + entropy_bonus * probs * (log_probs + entropy[:, None])`, divided by N. This uses `d log p(a) / d z_j = 1[j = a] - p_j`.
6. Backward, `_clip`, Adam step on the policy.
7. `value_loss = mean((values - returns) ** 2)`; gradient `2 * (values - returns) / N`; backward, clip, Adam step on the value network.
8. `policy_loss = -(log_probs[range(N), actions] * advantages).mean() - entropy_bonus * entropy.mean()`.

PPO overrides this method and adds `_advantages`; see [ppo.md](ppo.md).

#### `_clip(grads)`

```python
def _clip(self, grads: list[tuple[np.ndarray, np.ndarray]]) -> list[tuple[np.ndarray, np.ndarray]]:
```

Global-norm clipping: if the combined length of every layer's gradients exceeds `max_grad_norm`, scale them all by `max_grad_norm / norm`.

#### `_outcome_means(episodes)` (static)

```python
@staticmethod
def _outcome_means(episodes: list[dict]) -> tuple[float, float, float, float]:
```

Four means over the episodes, or zeros if there are no outcomes:

| Position | Value | Averaged over |
| --- | --- | --- |
| 0 | `return` | Every learner outcome (copies times games) |
| 1 | `survival` | Every learner outcome |
| 2 | `learner_won` | Every episode: the game-level win rate |
| 3 | `kills` | Every learner outcome |

So with 4 episodes and 6 learners the return is a mean of 24 numbers and the win rate is a mean of 4.

#### `step_epoch(on_progress=None)`

```python
def step_epoch(self, on_progress: Callable[[int, int], None] | None = None) -> EpochStats:
```

1. Start the clocks.
2. `_apply_curriculum()`.
3. Draw `episodes_per_epoch` seeds and `_collect` them with `greedy=False`, recording the first when `record_showcase` is on.
4. `_update(episodes)`.
5. Training means from `_outcome_means`: `train_return`, `train_survival`, `win_rate`, `kill_rate`.
6. Validation on seeds `validation_seed + i` with `greedy=True` and the updated policy. Never recorded. `val_return`, `val_survival` and `val_win_rate` come from `_outcome_means` of those episodes, or zeros when `validation_games` is 0.
7. If `val_return` beats `best_val_return` (or no best yet), store a copy of the policy as `best_genome`.
8. Merge the training telemetry, build `EpochStats` (with `showcase = episodes[0].get("recording")`), append, increment `epoch`.
9. `_record_iteration(scores, entropy, train_survival, win_rate, val_return, seconds, {"policy_loss", "value_loss"}, telemetry, showcase, val_win_rate)` with `scores` = every learner's return in every training episode.

The showcase is the game played by the policy *before* this epoch's update; `genome` is the policy *after* it.

#### `run(on_epoch=None, on_progress=None)`

```python
def run(self, on_epoch: Callable[[EpochStats], None] | None = None, on_progress: Callable[[int, int], None] | None = None) -> list[EpochStats]:
```

Clears `_stop`, loops `step_epoch` while `epoch < rl.epochs` and not stopped, calling `on_epoch(stats)` after each. Returns `history`.

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

`best_genome` if one has been stored, otherwise the current policy genome.

#### `champion_brain(chaos=0.0)`

```python
def champion_brain(self, chaos: float = 0.0) -> NeuralBrain:
```

A `NeuralBrain` with the champion loaded, greedy by default.

#### `save_policy(path)` and `save_champion(path)`

```python
def save_policy(self, path: str | Path) -> None:
def save_champion(self, path: str | Path) -> None:
```

Writes JSON with `brain_name` (`"neural"`), `neural`, `genome`, `value_genome`, `value_hidden`, `fitness` (`best_val_return` if finite, else `0.0`), `epochs` and `method` (`"reinforce"`). `save_champion` is an alias, so every trainer has the same method name. `PPOTrainer` inherits both and therefore also writes `"reinforce"`.

#### `history_rows()`

```python
def history_rows(self) -> list[dict]:
```

`[stats.to_row() for stats in self.history]`.

## How to use it / experiment

**A minimal run.**

```python
from hunger_games.config import SimulationConfig
from hunger_games.training import ReinforceTrainer, RLConfig

config = SimulationConfig(width=80, height=80, max_days=8)
trainer = ReinforceTrainer(config, RLConfig(epochs=10, episodes_per_epoch=4, seed=0))
trainer.run(on_epoch=lambda e: print(e.epoch, round(e.val_return, 2), round(e.val_win_rate, 2)))
print(trainer.events.tail(3))
trainer.save_champion("policy.json")
```

**Warm-start from an imitation champion, with the curriculum.**

```python
from hunger_games.training import Curriculum, CurriculumConfig, ImitationConfig, ImitationTrainer

student = ImitationTrainer(config, ImitationConfig(seed=0))
student.run()
trainer = ReinforceTrainer(
    config, RLConfig(seed=0), initial_genome=student.champion, curriculum=Curriculum(CurriculumConfig())
)
trainer.run()
```

The policy starts as the student; the value network starts fresh. The first stage is a game of 7 tributes (6 learners and 1 opponent). The policy moves to 3 opponents once it has won at least half of its validation games over five epochs, and so on up the ladder. There is no timeout.

**Score any learner with `play_rl_episode`.**

```python
from hunger_games.training.common import LearnerSpec, learner_ids
from hunger_games.training.reinforce import play_rl_episode

result = play_rl_episode(config, None, trainer.champion_spec(), learner_ids(24, 6), seed=1, greedy=True)
print(result["learner_won"], [o["return"] for o in result["outcomes"].values()])
```

A `LearnerSpec("neat", genome_dict)` or `LearnerSpec("voting", eight_genes)` works the same way.

**What each logged metric means.**

| Metric | Meaning | Healthy trend |
| --- | --- | --- |
| `policy_loss` | Batch mean of `-log p(a) * A - beta * H` with normalised `A` | Hovers near 0; watch only for blow-ups |
| `value_loss` | How wrong the baseline was | Falls over the first epochs, then settles |
| `entropy` | Spread of the policy's choices (max 2.77) | Slowly down |
| `train_return` | Mean total reward per learner episode, sampled policy | Up, but noisy |
| `val_return` | Same on fixed seeds, greedy | Up. Picks the champion |
| `win_rate`, `val_win_rate` | Fraction of games a learner copy won | Up. With 2 validation games the values are 0, 0.5 or 1 per epoch; the curriculum and the comparison average them over a window |

**Shape the reward.** `SimulationConfig(reward=RewardConfig(kill=3.0, death=-1.0))` breeds fighters; `RewardConfig(survive_tick=0.05, kill=0.0)` breeds hiders (see the zombie video's lesson in [ppo.md](ppo.md)).

**Continue training.** `run()` resumes; change `trainer.rl.epochs` first if the budget is used up.

## Gotchas

- **`validation_games=0` freezes the champion.** `val_return` is then always `0.0`, so `champion` stays the epoch-0 policy. Use `history[-1].genome` for the latest policy in that case. The curriculum then falls back to the training win rate.
- **Validation win rates are coarse.** Two games per epoch give 0, 0.5 or 1. Raise `validation_games` if the curriculum promotes on a lucky pair.
- **Showcases stay in memory.** Set `record_showcase=False` for long runs on big maps.
- The showcase is the epoch's *first* training game, played by the sampling policy before the update. It is not a greedy game and not the champion.
- Advantages are normalised per batch, so `policy_loss` cannot be compared across epochs.
- Rewards are attached to the *latest decision*, so a learner's last decision before dying carries the death penalty, the placement bonus, and possibly a win bonus.
- The approach term only counts when water (or grass) was visible both before and after the move.
- A warm start loads the policy only. The value network is always fresh, so expect `value_loss` to start high.
- The same `spawn` rules as the genetic trainer apply when `workers > 1`. Each job pickles a `LearnerSpec` holding the genome.
- `build_learner` builds neural learners with `NeuralBrain(...)` directly, not `create_brain`, so `config.endgame_instinct` does not apply to learners. It does apply to the opponents.
- Learners always use `config.neural` for their architecture. An `initial_genome` of the wrong size raises `ValueError` from `set_genome`.
- `IterationStats.iteration` is `len(learning_history)` at the time of the record, which equals `epoch - 1` after `step_epoch` increments the counter. They agree as long as every epoch records once.
- Each epoch is one gradient step. Thirty epochs is a smoke test. Real runs need hundreds of epochs, or PPO's several passes per batch, and a warm start.
