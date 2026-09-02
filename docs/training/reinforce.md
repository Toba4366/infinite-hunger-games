# `reinforce.py`

**Source:** [hunger_games/training/reinforce.py](../../hunger_games/training/reinforce.py)
**Depends on:** `json`, `time`, `collections.abc.Callable`, `concurrent.futures.ProcessPoolExecutor`, `dataclasses`, `pathlib` (standard library); `numpy`; [brain/__init__.py](../brain/init.md) (`Brain`, `create_brain`); [brain/mlp.py](../../hunger_games/brain/mlp.py) (`MLP`, `Adam`); [brain/neural.py](../brain/neural.md) (`NeuralBrain`, `softmax`); [hunger_games/config.py](../config.md) (`SimulationConfig`, and through it `RewardConfig`); [hunger_games/game.py](../game.md) (`Game`); [hunger_games/perception.py](../perception.md) (`VECTOR_SIZE`); [hunger_games/player.py](../player.md) (`Player`); [hunger_games/research/telemetry.py](../../hunger_games/research/telemetry.py) (`BehaviorTelemetry`); [hunger_games/scenario.py](../scenario.md) (`Scenario`)
**Used by:** [training/__init__.py](init.md) (re-exports `EpochStats`, `ReinforceTrainer`, `RLConfig`); [training/runs.py](runs.md) (`save_run` reads `trainer.rl`, `history`, `history_rows()`, `save_policy`); [hunger_games/ui/session.py](../ui/session.md) (`ReinforceTrainer`, `RLConfig`, `save_policy`); [hunger_games/ui/app.py](../ui/app.md) (`RLConfig`); [experiments/run_rl.py](../experiments/run_rl.md); `tests/test_research.py`

## Purpose

This file trains the neural brain by reinforcement learning. The genetic algorithm in [genetic.md](genetic.md) scores whole games. This trainer scores every action: after each tick a learning tribute gets a reward built from the weights in `RewardConfig`, and the policy network is nudged to make well-rewarded actions more likely. A second network, the value network, predicts how much reward is still to come from a state. Subtracting that prediction (the "baseline") makes the learning signal far less noisy. This is REINFORCE with a learned baseline, the simplest actor-critic method, written in plain numpy on top of the `MLP` class.

Everything a researcher asks for is logged per epoch: policy loss, value loss, policy entropy, training return, validation return on held-out seeds, survival time, win and kill rates, wall-clock time, and the behaviour telemetry from `research/telemetry.py`.

## Concepts you need

**Policy.** A function from state to a probability for each action. Here the state is the 50-value perception vector (`VECTOR_SIZE`), the actions are the neural brain's 16-item menu (`MENU_SIZE`: 6 simple actions, attack, flee, 8 moves), and the policy is the `MLP` inside a `NeuralBrain` followed by a softmax.

**Episode, return, discount.** One game is an episode. Each learner collects a list of rewards, one per decision. The return at step `t` is `G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...`, with `gamma = RewardConfig.discount = 0.98`. Discounting says a reward 50 ticks away is worth `0.98^50 = 0.36` of a reward now.

**Policy gradient.** To make action `a` more likely in state `s`, increase `log p(a | s)`. REINFORCE multiplies that push by how good the outcome was. The loss for one sample is `-log p(a) * A` where `A` is the advantage.

**Baseline and advantage.** `A = G - V(s)`, the return minus what the value network predicted. Actions that did better than expected get pushed up, worse than expected pushed down. Without the baseline every positive return pushes every action up and learning is dominated by noise.

**Entropy.** `H = -sum_a p(a) log p(a)`, in nats. A uniform policy over 16 actions has `H = ln 16 = 2.77`; a certain policy has `H = 0`. An entropy bonus keeps the policy exploring.

**Greedy versus sampling.** During training the learner samples from the softmax at temperature 1 (`chaos=1.0`). During validation it takes the argmax (`chaos=0.0`). See `NeuralBrain.probabilities` in [../brain/neural.md](../brain/neural.md).

**Hooks.** `Game.decision_hooks` are called right after a brain decides and before the action is carried out; `Game.tick_hooks` are called at the end of each tick after `tick` has advanced. The reward is computed in a tick hook and attached to the most recent decision.

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

The reward weights and the discount are not here. They live in `SimulationConfig.reward` (a `RewardConfig`, see [../config.md](../config.md)) because the reward is a property of the game, not of the learner.

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
| `win_rate` | Fraction of learner episodes that won, training |
| `val_win_rate` | Same, validation |
| `kill_rate` | Mean kills per learner episode, training |
| `seconds` | Wall-clock seconds this epoch took |
| `cumulative_seconds` | Seconds since training started |
| `genome` | The policy genome after this epoch's update (a copy, `repr=False`) |
| `telemetry` | Merged telemetry of this epoch's training episodes (`repr=False`) |

#### `to_row()`

```python
def to_row(self) -> dict:
```

Every field except `genome` and `telemetry`, for `history.json` and the plots.

### `play_rl_episode(...)`

```python
def play_rl_episode(config: SimulationConfig, scenario: Scenario | None, genome: np.ndarray, learner_ids: list[int], seed: int, greedy: bool) -> dict:
```

Plays one game with the policy driving `learner_ids` and returns their experience. A top-level function so worker processes can run it.

**Setup.** Copies the config with the episode `seed`. The brain factory gives each learner slot a `NeuralBrain(chaos=0.0 if greedy else 1.0, config=neural, rng=rng)` loaded with `genome`, and keeps it in `learners` so its `last_index` can be read. Every other slot gets `create_brain(config.brain_name, config.chaos, ...)`. A `BehaviorTelemetry` is attached that tracks only the learner ids. Three per-learner lists are made: `vectors`, `indices`, `rewards`.

**Decision hook `on_decision(player, perception, action)`.** For a learner, appends `perception.to_vector()` to `vectors`, `learners[pid].last_index` to `indices` (the menu index chosen by `decide_index`), and a placeholder `0.0` to `rewards`. All three lists stay the same length.

**Tick hook `on_tick(current)`.** Runs after every tick for every learner, whether it decided or not. With `prev = (thirst, hunger, health, kills)` from the previous tick (or the current values on the first tick, so the first deltas are zero), the tick's reward `r` is:

| Term | When | Amount |
| --- | --- | --- |
| `survive_tick` (`0.01`) | Alive | `+ survive_tick` |
| `damage_taken` (`-2.0`) | Alive, health fell | `+ damage_taken * max(0, prev_health - health)` |
| `need_gain` (`0.5`) | Alive, previous thirst below 0.5, thirst rose | `+ need_gain * max(0, thirst - prev_thirst)` |
| `need_gain` (`0.5`) | Alive, previous hunger below 0.5, hunger rose | `+ need_gain * max(0, hunger - prev_hunger)` |
| `death` (`-3.0`) | Not alive and not yet paid | `+ death`, once |
| `kill` (`1.0`) | Any | `+ kill * (kills - prev_kills)` |

Then `previous[pid]` is updated and `r` is added to `rewards[pid][-1]`, the latest decision, provided there has been at least one decision. A learner who died this tick before its turn to decide has its death penalty attached to the previous tick's decision. Health lost to thirst or hunger drain counts as damage, so `damage_taken` also punishes letting the bars run dry.

The hooks are appended after the telemetry's, so death bookkeeping is done before the reward is computed.

**End of game.** After `game.run()`, with `n = len(game.players)`:

- `bonus = placement * (n - (player.placement or n)) / max(1, n - 1)`. First place earns the full `placement` weight (`2.0`), last earns nothing. Survivors of a draw share a placement equal to how many survived.
- If the learner is alive and is the only one alive, `bonus += win` (`5.0`).
- The bonus is added to the last decision's reward.

**Return value.** A dict with `vectors`, `indices` and `rewards` (each `{pid: np.ndarray}`), `outcomes` (`{pid: {"return", "survival", "won", "kills"}}` where `survival` is `death_ticks.get(pid, game.tick)`), and `telemetry` (the summary).

### `_run_episode_job(args)`

```python
def _run_episode_job(args: tuple) -> dict:
```

Unpacks a tuple for `ProcessPoolExecutor.map` and calls `play_rl_episode`.

### `ReinforceTrainer`

```python
class ReinforceTrainer:
```

#### `__init__(config, rl, scenario=None)`

```python
def __init__(self, config: SimulationConfig, rl: RLConfig, scenario: Scenario | None = None) -> None:
```

Seeds `self.rng` from `rl.seed`. The policy is `NeuralBrain(chaos=1.0, config=config.neural, rng=self.rng).network`, an `MLP` of shape `[50, *hidden_layers, 16]` with the config's activation and initializer (1088 parameters for the default `(16,)`). The value network is `MLP([VECTOR_SIZE, *rl.value_hidden, 1], config.neural.activation, "xavier_uniform", rng=self.rng)` (1665 parameters for `(32,)`). Each gets its own `Adam`. Also sets up `history`, `epoch = 0`, `_stop`, `_started`, `best_genome = None` and `best_val_return = -inf`.

#### `_learner_ids()`

```python
def _learner_ids(self) -> list[int]:
```

`count = min(learners_per_game, num_players)` slots at `int(i * num_players / count)`. With 6 of 24 that is `[0, 4, 8, 12, 16, 20]`, evenly spaced so learners are not all neighbours on the podiums.

#### `_collect(seeds, greedy, on_progress=None)`

```python
def _collect(self, seeds: list[int], greedy: bool, on_progress: Callable[[int, int], None] | None = None) -> list[dict]:
```

Reads the policy genome once, builds one job `(config, scenario, genome, learners, seed, greedy)` per seed, and runs them through a pool when `workers > 1` and there is more than one job, else in sequence. Calls `on_progress(done, total)` after each. Returns the list of episode dicts.

#### `_returns(rewards)`

```python
def _returns(self, rewards: np.ndarray) -> np.ndarray:
```

Walks the rewards backwards with `running = rewards[t] + discount * running` and stores `running` at each `t`. That is the discounted return `G_t`. For rewards `[0, 0, 5]` and discount 0.98 the returns are `[4.80, 4.90, 5.0]`.

#### `_update(episodes)`

```python
def _update(self, episodes: list[dict]) -> tuple[float, float, float]:
```

One gradient step on both networks. Returns `(policy_loss, value_loss, entropy)`.

1. **Gather.** For every learner in every episode with at least one decision, collect its vectors, indices and `_returns(rewards)`. Concatenate into `states` (N x 50), `actions` (N), `returns` (N). Return `(0.0, 0.0, 0.0)` if nothing was collected.
2. **Baseline.** `values, value_cache = self.value.forward_cached(states)`, flattened to N.
3. **Advantages.** `advantages = returns - values`, then normalised: `(A - mean) / (std + 1e-8)`. After this the batch has mean 0 and standard deviation 1, so the size of a step does not depend on the reward scale, and roughly half the samples are pushed down.
4. **Policy forward.** `logits, policy_cache = self.policy.forward_cached(states)`; `probs = softmax(logits)`; `log_probs = log(probs + 1e-12)`; `entropy = -(probs * log_probs).sum(axis=1)` per sample.
5. **Gradient at the logits.** Let `z` be the logits, `p = softmax(z)`, `a` the chosen action and `A` its advantage. For the loss `-log p(a) * A`, the derivative with respect to `z_j` is `(p_j - 1[j = a]) * A`, because `d log p(a) / d z_j = 1[j = a] - p_j` for a softmax. For the entropy term `-beta * H`, the derivative is `beta * p_j * (log p_j + H)`. So

   `grad_logits = (probs - one_hot) * advantages[:, None] + entropy_bonus * probs * (log_probs + entropy[:, None])`

   divided by N for a batch mean.
6. **Policy step.** `self.policy.backward(policy_cache, grad_logits)` gives one `(grad_w, grad_b)` per layer, `_clip` scales them, `policy_optimizer.step` applies Adam.
7. **Value loss and step.** `value_loss = mean((values - returns) ** 2)`. Its gradient at the output is `2 * (values - returns) / N`, reshaped to a column. Backward, clip, Adam step. Note the value net is updated *after* it was used as the baseline, so the baseline in step 2 is the previous epoch's prediction.
8. **Loss for the record.** `policy_loss = -(log_probs[range(N), actions] * advantages).mean() - entropy_bonus * entropy.mean()`, computed from the pre-update probabilities.

#### `_clip(grads)`

```python
def _clip(self, grads: list[tuple[np.ndarray, np.ndarray]]) -> list[tuple[np.ndarray, np.ndarray]]:
```

Computes the global norm across every layer's weight and bias gradients. If it exceeds `max_grad_norm`, every gradient is multiplied by `max_grad_norm / (norm + 1e-12)`. This is the same "clip by global norm" rule that most deep learning libraries offer. It stops one unlucky batch from throwing the weights far away.

#### `_outcome_means(episodes)` (static)

```python
@staticmethod
def _outcome_means(episodes: list[dict]) -> tuple[float, float, float, float]:
```

Means of `return`, `survival`, `won` and `kills` over every learner outcome in every episode. `(0.0, 0.0, 0.0, 0.0)` if empty.

#### `step_epoch(on_progress=None)`

```python
def step_epoch(self, on_progress: Callable[[int, int], None] | None = None) -> EpochStats:
```

1. Start the clocks.
2. Draw `episodes_per_epoch` random seeds from `self.rng` and `_collect` them with `greedy=False`.
3. `_update(episodes)`.
4. Training means from `_outcome_means`.
5. Validation: seeds `validation_seed + i` for `i < validation_games`, collected with `greedy=True` and the *updated* policy. The validation kill rate is discarded.
6. If `val_return > best_val_return` or no best yet, store a copy of the current policy genome as `best_genome`.
7. Merge the training episodes' telemetry.
8. Build `EpochStats` (with a copy of the current genome), append, increment `epoch`.

#### `run(on_epoch=None, on_progress=None)`

```python
def run(self, on_epoch: Callable[[EpochStats], None] | None = None, on_progress: Callable[[int, int], None] | None = None) -> list[EpochStats]:
```

Clears `_stop`, loops `step_epoch` while `epoch < rl.epochs` and not stopped, calling `on_epoch(stats)` after each. Returns `history`. A second `run()` continues from the current epoch.

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

`best_genome` if one has been stored, otherwise the current policy genome. Never `None` in practice, because a fresh policy exists from `__init__`.

#### `champion_brain(chaos=0.0)`

```python
def champion_brain(self, chaos: float = 0.0) -> NeuralBrain:
```

A `NeuralBrain` with the champion loaded. The default chaos of 0 makes it greedy, the same way it was validated.

#### `save_policy(path)`

```python
def save_policy(self, path: str | Path) -> None:
```

Writes JSON in the same shape as a GA champion file, plus extras:

| Key | Value |
| --- | --- |
| `brain_name` | `"neural"` |
| `neural` | `asdict(config.neural)` |
| `genome` | The champion policy as a list |
| `value_genome` | The value network's `genome()` as a list |
| `value_hidden` | `list(rl.value_hidden)` |
| `fitness` | `best_val_return` if finite, else `0.0` |
| `epochs` | `len(history)` |
| `method` | `"reinforce"` |

`GeneticTrainer.load_champion` reads it back and the dashboard's "Load champion into all" accepts it.

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
trainer.run(on_epoch=lambda e: print(e.epoch, round(e.val_return, 2), round(e.entropy, 2)))
trainer.save_policy("policy.json")
```

**What each logged metric means, and the trend you want.**

| Metric | Meaning | Healthy trend |
| --- | --- | --- |
| `policy_loss` | Batch mean of `-log p(a) * A - beta * H` with normalised `A` | Hovers near 0 and is *not* a progress signal. Watch it only for blow-ups (huge values or `nan`) |
| `value_loss` | How wrong the baseline was, in squared reward units | Falls over the first epochs, then settles. A steady rise means returns are drifting faster than the value net can follow; lower `value_learning_rate` or raise `episodes_per_epoch` |
| `entropy` | How spread out the policy's choices are (max `ln 16 = 2.77`) | Slowly down. Stuck at 2.77 means nothing is being learned; a crash toward 0 in a few epochs is premature collapse, so raise `entropy_bonus` |
| `train_return` | Mean undiscounted total reward per learner episode, sampled policy | Up, but noisy: seeds change every epoch |
| `val_return` | Same on fixed seeds with the greedy policy | Up. This is the primary yardstick and picks the champion |
| `train_survival`, `val_survival` | Ticks alive (max `ticks_per_game`, 576 by default) | Up, if the reward favours survival |
| `win_rate`, `val_win_rate` | Fraction of learner episodes that won | Up. With 6 learners per game at most one can win, so the ceiling is about 0.17 |
| `kill_rate` | Kills per learner episode | Depends on the reward. Up if `kill` dominates, flat if survival does |
| `seconds`, `cumulative_seconds` | Wall-clock | Roughly flat. Rising seconds usually mean learners survive longer, so games run longer |

**Shape the reward.** `SimulationConfig(reward=RewardConfig(kill=3.0, death=-1.0))` breeds fighters; `RewardConfig(survive_tick=0.05, kill=0.0)` breeds hiders. Compare `deaths_by_cause.png` and `action_distribution_over_training.png` between run folders.

**Fewer learners, harder opponents.** `learners_per_game=1` gives a single learner against 23 voting brains and the cleanest signal, but 4 episodes then yield only 4 trajectories per epoch. Raise `episodes_per_epoch` to keep the batch large.

**Continue training.** `run()` resumes; change `trainer.rl.epochs` first if the budget is used up.

**Gradient check.** `tests/test_research.py` checks the MLP's backward pass against finite differences. If you change the loss, add a similar check for `grad_logits`.

## Gotchas

- **`validation_games=0` freezes the champion.** `val_return` is then always `0.0`, so the "better than best" test succeeds only on epoch 0 and `champion` stays the epoch-0 policy. Use `history[-1].genome` for the latest policy in that case.
- Advantages are normalised per batch, so `policy_loss` cannot be compared across epochs and does not go to zero as the policy improves.
- The value network sees returns on the raw reward scale while advantages are normalised. `value_loss` therefore scales with the reward weights; doubling every weight roughly quadruples it.
- Rewards are attached to the *latest decision*, and every tick adds to it, so a learner's last decision before dying carries the death penalty, the placement bonus, and possibly a win bonus. Long spans without a decision do not happen, because every living tribute decides every tick.
- The same `spawn` rules as the genetic trainer apply when `workers > 1`: a `__main__` guard, a script file, and pickle-able `config` and `scenario`. See [genetic.md](genetic.md).
- `play_rl_episode` builds learners with `NeuralBrain(...)` directly, not `create_brain`, so `config.endgame_instinct` does not apply to learners. It does apply to the opponents.
- Learners always use `config.neural` for their architecture. A policy saved with `hidden_layers=(32,)` will not load into a `ReinforceTrainer` built with the default `(16,)`.
- `greedy=True` validation uses chaos 0 and picks the argmax. A policy that looks fine when sampling can score badly greedily if its argmax action is a poor default (for example "attack" with no weapon). Compare `train_return` and `val_return`.
- Each epoch is one gradient step. Thirty epochs is thirty updates, which is a smoke test. Real runs need hundreds of epochs or many more episodes per epoch.
