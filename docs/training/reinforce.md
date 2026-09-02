# `reinforce.py`

**Source:** [hunger_games/training/reinforce.py](../../hunger_games/training/reinforce.py)
**Depends on:** `json`, `time`, `collections.abc.Callable`, `concurrent.futures.ProcessPoolExecutor`, `dataclasses`, `pathlib` (standard library); `numpy`; [brain/__init__.py](../brain/init.md) (`Brain`, `create_brain`); [brain/mlp.py](../brain/mlp.md) (`MLP`, `Adam`); [brain/neural.py](../brain/neural.md) (`NeuralBrain`, `softmax`); [hunger_games/config.py](../config.md) (`SimulationConfig`, and through it `RewardConfig`); [hunger_games/game.py](../game.md) (`Game`); [hunger_games/perception.py](../perception.md) (`VECTOR_SIZE`); [hunger_games/player.py](../player.md) (`Player`); [hunger_games/recorder.py](../recorder.md) (`Recorder`, `Recording`); [hunger_games/research/telemetry.py](../research/telemetry.md) (`BehaviorTelemetry`); [hunger_games/scenario.py](../scenario.md) (`Scenario`)
**Used by:** [training/__init__.py](init.md) (re-exports `EpochStats`, `ReinforceTrainer`, `RLConfig`); [training/imitation.py](imitation.md) (`play_rl_episode` plays the student's greedy validation games); [training/runs.py](runs.md) (`save_run` reads `settings`, `config`, `history`, `history_rows()`, `champion`, `save_champion`); [hunger_games/ui/session.py](../ui/session.md) (`ReinforceTrainer` with `initial_genome` for warm starts, `RLConfig`, `save_champion`, and `history[-1].showcase` for the training feed); [hunger_games/ui/app.py](../ui/app.md) (`RLConfig`); [experiments/run_rl.py](../experiments/run_rl.md); `tests/test_research.py`; `tests/test_feed.py`; `tests/test_imitation.py` (the warm start)

## Purpose

This file trains the neural brain by reinforcement learning. The genetic algorithm in [genetic.md](genetic.md) scores whole games. This trainer scores every action: after each tick a learning tribute gets a reward built from the weights in `RewardConfig`, and the policy network is nudged to make well-rewarded actions more likely. A second network, the value network, predicts how much reward is still to come from a state. Subtracting that prediction (the "baseline") makes the learning signal far less noisy. This is REINFORCE with a learned baseline, the simplest actor-critic method, written in plain numpy on top of the `MLP` class.

Everything a researcher asks for is logged per epoch: policy loss, value loss, policy entropy, training return, validation return on held-out seeds, survival time, win and kill rates, wall-clock time, and the behaviour telemetry from `research/telemetry.py`. With `record_showcase` on, each epoch also keeps a tick-by-tick `Recording` of its first training game, so the dashboard's training feed can replay what the learners actually did.

The constructor accepts an `initial_genome`, so the policy can start from a network that already has instincts. The recommended flow is to pretrain by imitation ([imitation.md](imitation.md)) and then reinforce from that champion. A cold-started policy usually dies of thirst before any reward reaches it.

## Concepts you need

**Policy.** A function from state to a probability for each action. Here the state is the 50-value perception vector (`VECTOR_SIZE`), the actions are the neural brain's 16-item menu (`MENU_SIZE`: 6 simple actions, attack, flee, 8 moves), and the policy is the `MLP` inside a `NeuralBrain` followed by a softmax.

**Episode, return, discount.** One game is an episode. Each learner collects a list of rewards, one per decision. The return at step `t` is `G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...`, with `gamma = RewardConfig.discount = 0.98`. Discounting says a reward 50 ticks away is worth `0.98^50 = 0.36` of a reward now.

**Policy gradient.** To make action `a` more likely in state `s`, increase `log p(a | s)`. REINFORCE multiplies that push by how good the outcome was. The loss for one sample is `-log p(a) * A` where `A` is the advantage.

**Baseline and advantage.** `A = G - V(s)`, the return minus what the value network predicted. Actions that did better than expected get pushed up, worse than expected pushed down. Without the baseline every positive return pushes every action up and learning is dominated by noise.

**Entropy.** `H = -sum_a p(a) log p(a)`, in nats. A uniform policy over 16 actions has `H = ln 16 = 2.77`; a certain policy has `H = 0`. An entropy bonus keeps the policy exploring.

**Shaping.** A dense reward that pays out for progress toward a goal, not just for reaching it. `RewardConfig.approach` is one: a bonus per cell moved closer to water while thirsty. It is off by default (`0.0`) because imitation pretraining is the preferred way to give a network instincts. Shaping rewards are easy to game and change what "optimal" means.

**Greedy versus sampling.** During training the learner samples from the softmax at temperature 1 (`chaos=1.0`). During validation it takes the argmax (`chaos=0.0`). See `NeuralBrain.probabilities` in [../brain/neural.md](../brain/neural.md).

**Hooks.** `Game.decision_hooks` are called right after a brain decides and before the action is carried out; `Game.tick_hooks` are called at the end of each tick after `tick` has advanced. Most of the reward is computed in a tick hook and attached to the most recent decision; the approach term is computed in the decision hook from the previous tick's distances.

**Recordings.** A `Recorder` wraps a `Game`, captures frame 0 on construction, and `record_all()` steps the game to the end while capturing after every tick. The hooks above still fire during a recorded game, because `Recorder.step` calls `game.step()`. See [../recorder.md](../recorder.md).

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
| `showcase` | A `Recording` of this epoch's first training game, or `None` when `record_showcase` is off (`repr=False`) |

#### `to_row()`

```python
def to_row(self) -> dict:
```

Every field except `genome`, `telemetry` and `showcase`, for `history.json` and the plots. It walks `self.__dict__` and skips those three keys. The recording holds one frame per tick and is not JSON, so it never reaches a run folder.

### `play_rl_episode(...)`

```python
def play_rl_episode(config: SimulationConfig, scenario: Scenario | None, genome: np.ndarray, learner_ids: list[int], seed: int, greedy: bool, record: bool = False) -> dict:
```

Plays one game with the policy driving `learner_ids` and returns their experience. A top-level function so worker processes can run it. The imitation trainer reuses it with `greedy=True` for its validation games.

**Setup.** Copies the config with the episode `seed`. The brain factory gives each learner slot a `NeuralBrain(chaos=0.0 if greedy else 1.0, config=neural, rng=rng)` loaded with `genome`, and keeps it in `learners` so its `last_index` can be read. Every other slot gets `create_brain(config.brain_name, config.chaos, ...)`. A `BehaviorTelemetry` is attached that tracks only the learner ids. Three per-learner lists are made: `vectors`, `indices`, `rewards`. Two more dictionaries hold state between ticks: `previous` (bars and kills per learner) and `last_distances` (water distance, grass distance, and whether the learner was thirsty and hungry at its last decision). `dead_paid` remembers who has already received the death penalty.

**Decision hook `on_decision(player, perception, action)`.** Ignores non-learners. For a learner it does two things, in this order:

1. *Approach shaping.* If the learner has a previous decision (`rewards[pid]` is not empty) and `last_distances` holds its previous tick's `(water_before, grass_before, was_thirsty, was_hungry)`, then:
   - if `was_thirsty` and both `water_before` and `perception.water_distance` are finite, add `reward.approach * (water_before - perception.water_distance)` to `rewards[pid][-1]`;
   - if `was_hungry` and both grass distances are finite, add `reward.approach * (grass_before - perception.grass_distance)` to `rewards[pid][-1]`.

   At this point `rewards[pid][-1]` is the *previous* decision's reward, because this decision's placeholder has not been appended yet. So the previous action is rewarded for closing the distance to what was needed, and penalised for opening it. Then `last_distances[pid]` is updated with this tick's `water_distance`, `grass_distance`, `thirst < 0.5 and not in_water`, and `hunger < 0.5 and food_count == 0`. With the default `approach = 0.0` the bookkeeping still runs but adds zero.
2. *Record the decision.* Append `perception.to_vector()` to `vectors`, `learners[pid].last_index` to `indices` (the menu index chosen by `decide_index`), and a placeholder `0.0` to `rewards`. All three lists stay the same length.

**Tick hook `on_tick(current)`.** Runs after every tick for every learner, whether it decided or not. With `prev = (thirst, hunger, health, kills)` from the previous tick (or the current values on the first tick, so the first deltas are zero), the tick's reward `r` is:

| Term | When | Amount |
| --- | --- | --- |
| `survive_tick` (`0.01`) | Alive | `+ survive_tick` |
| `damage_taken` (`-2.0`) | Alive, health fell | `+ damage_taken * max(0, prev_health - health)` |
| `need_gain` (`0.5`) | Alive, previous thirst below 0.5, thirst rose | `+ need_gain * max(0, thirst - prev_thirst)` |
| `need_gain` (`0.5`) | Alive, previous hunger below 0.5, hunger rose | `+ need_gain * max(0, hunger - prev_hunger)` |
| `death` (`-3.0`) | Not alive and not yet paid | `+ death`, once |
| `kill` (`1.0`) | Any | `+ kill * (kills - prev_kills)` |
| `approach` (`0.0`) | Computed in the decision hook, see above | `+ approach * cells closer to water (thirsty) or grass (hungry)` |

Then `previous[pid]` is updated and `r` is added to `rewards[pid][-1]`, the latest decision, provided there has been at least one decision. A learner who died this tick before its turn to decide has its death penalty attached to the previous tick's decision. Health lost to thirst or hunger drain counts as damage, so `damage_taken` also punishes letting the bars run dry.

The hooks are appended after the telemetry's, so death bookkeeping is done before the reward is computed.

**Play.** `recording = Recorder(game).record_all() if record else None`. When `record` is on, the recorder plays the whole game and captures every tick; the hooks fire as normal. Then `game.run()` is called either way. If the recorder already played the game to the end, `run()` loops zero times, runs the end-of-game bookkeeping (which is safe to repeat), and returns. If nothing was recorded, `run()` is what plays the game.

**End of game.** With `n = len(game.players)`:

- `bonus = placement * (n - (player.placement or n)) / max(1, n - 1)`. First place earns the full `placement` weight (`2.0`), last earns nothing. Survivors of a draw share a placement equal to how many survived.
- If the learner is alive and is the only one alive, `bonus += win` (`5.0`).
- The bonus is added to the last decision's reward.

**Return value.** A dict with `vectors`, `indices` and `rewards` (each `{pid: np.ndarray}`), `outcomes` (`{pid: {"return", "survival", "won", "kills"}}` where `survival` is `death_ticks.get(pid, game.tick)`), `telemetry` (the summary), and `recording` (the `Recording`, or `None`).

### `_run_episode_job(args)`

```python
def _run_episode_job(args: tuple) -> dict:
```

Unpacks a tuple for `ProcessPoolExecutor.map` and calls `play_rl_episode`.

### `ReinforceTrainer`

```python
class ReinforceTrainer:
```

#### `__init__(config, rl, scenario=None, initial_genome=None)`

```python
def __init__(self, config: SimulationConfig, rl: RLConfig, scenario: Scenario | None = None, initial_genome: np.ndarray | None = None) -> None:
```

Seeds `self.rng` from `rl.seed`. The policy is `NeuralBrain(chaos=1.0, config=config.neural, rng=self.rng).network`, an `MLP` of shape `[50, *hidden_layers, 16]` with the config's activation and initializer (5872 parameters for the default `(64, 32)`). If `initial_genome` is given, it is loaded into the policy with `set_genome` (a warm start from an imitation-pretrained network or an earlier champion). The value network is `MLP([VECTOR_SIZE, *rl.value_hidden, 1], config.neural.activation, "xavier_uniform", rng=self.rng)` (1665 parameters for `(32,)`) and always starts fresh. Each gets its own `Adam`. Also sets up `history`, `epoch = 0`, `_stop`, `_started`, `best_genome = None` and `best_val_return = -inf`.

#### `settings` (property)

```python
@property
def settings(self):
```

Returns `self.rl`. Every trainer exposes this name so `save_run` can write the trainer's settings without knowing which trainer it has.

#### `_learner_ids()`

```python
def _learner_ids(self) -> list[int]:
```

`count = min(learners_per_game, num_players)` slots at `int(i * num_players / count)`. With 6 of 24 that is `[0, 4, 8, 12, 16, 20]`, evenly spaced so learners are not all neighbours on the podiums. The dashboard's "live" feed mode gives the champion to these same slots.

#### `_collect(seeds, greedy, on_progress=None, record_first=False)`

```python
def _collect(self, seeds: list[int], greedy: bool, on_progress: Callable[[int, int], None] | None = None, record_first: bool = False) -> list[dict]:
```

Reads the policy genome once, builds one job `(config, scenario, genome, learners, seed, greedy, record)` per seed, where `record` is `record_first and index == 0`, and runs them through a pool when `workers > 1` and there is more than one job, else in sequence. Calls `on_progress(done, total)` after each. Returns the list of episode dicts, in seed order. At most the first episode carries a recording.

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

   divided by N for a batch mean. The imitation trainer uses the same `(p - onehot)` identity with the advantage replaced by 1.
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
2. Draw `episodes_per_epoch` random seeds from `self.rng` and `_collect` them with `greedy=False` and `record_first=self.rl.record_showcase`. So the first training game of the epoch is recorded when the switch is on.
3. `_update(episodes)`.
4. Training means from `_outcome_means`.
5. Validation: seeds `validation_seed + i` for `i < validation_games`, collected with `greedy=True` and the *updated* policy. Validation games are never recorded. The validation kill rate is discarded.
6. If `val_return > best_val_return` or no best yet, store a copy of the current policy genome as `best_genome`.
7. Merge the training episodes' telemetry.
8. Build `EpochStats` (with a copy of the current genome, and `showcase=episodes[0].get("recording")`, or `None` if there were no episodes), append, increment `epoch`.

The showcase is the game played by the policy *before* this epoch's update, while `genome` is the policy *after* it. They are one gradient step apart.

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

**A minimal run.**

```python
from hunger_games.config import SimulationConfig
from hunger_games.training import ReinforceTrainer, RLConfig

config = SimulationConfig(width=80, height=80, max_days=8)
trainer = ReinforceTrainer(config, RLConfig(epochs=10, episodes_per_epoch=4, seed=0))
trainer.run(on_epoch=lambda e: print(e.epoch, round(e.val_return, 2), round(e.entropy, 2)))
trainer.save_champion("policy.json")
```

**Warm-start from an imitation champion.** The recommended flow:

```python
from hunger_games.training import ImitationConfig, ImitationTrainer

student = ImitationTrainer(config, ImitationConfig(seed=0))
student.run()
trainer = ReinforceTrainer(config, RLConfig(seed=0), initial_genome=student.champion)
trainer.run()
```

The policy starts as the student; the value network starts fresh and needs a few epochs to become a useful baseline. In the dashboard, run imitation, then tick "start from the current champion" and switch the method to reinforce.

**Watch an epoch.** Each `EpochStats.showcase` is a full recording of that epoch's first training game, with the learners in the `_learner_ids()` slots. Save it or turn it into a GIF:

```python
from hunger_games.renderer import export_recording_gif

trainer.history[0].showcase.save("epoch_0.replay")
export_recording_gif(trainer.history[-1].showcase, "epoch_last.gif")
```

In the dashboard, set the training feed to "replay" and each epoch's game is loaded as soon as the arena is free. See [../ui/session.md](../ui/session.md).

**What each logged metric means, and the trend you want.**

| Metric | Meaning | Healthy trend |
| --- | --- | --- |
| `policy_loss` | Batch mean of `-log p(a) * A - beta * H` with normalised `A` | Hovers near 0 and is *not* a progress signal. Watch it only for blow-ups (huge values or `nan`) |
| `value_loss` | How wrong the baseline was, in squared reward units | Falls over the first epochs, then settles. A steady rise means returns are drifting faster than the value net can follow; lower `value_learning_rate` or raise `episodes_per_epoch` |
| `entropy` | How spread out the policy's choices are (max `ln 16 = 2.77`) | Slowly down. Stuck at 2.77 means nothing is being learned; a crash toward 0 in a few epochs is premature collapse, so raise `entropy_bonus`. A warm-started policy starts low |
| `train_return` | Mean undiscounted total reward per learner episode, sampled policy | Up, but noisy: seeds change every epoch |
| `val_return` | Same on fixed seeds with the greedy policy | Up. This is the primary yardstick and picks the champion |
| `train_survival`, `val_survival` | Ticks alive (max `ticks_per_game`, 576 by default) | Up, if the reward favours survival |
| `win_rate`, `val_win_rate` | Fraction of learner episodes that won | Up. With 6 learners per game at most one can win, so the ceiling is about 0.17 |
| `kill_rate` | Kills per learner episode | Depends on the reward. Up if `kill` dominates, flat if survival does |
| `seconds`, `cumulative_seconds` | Wall-clock | Roughly flat. Rising seconds usually mean learners survive longer, so games run longer |

**Shape the reward.** `SimulationConfig(reward=RewardConfig(kill=3.0, death=-1.0))` breeds fighters; `RewardConfig(survive_tick=0.05, kill=0.0)` breeds hiders. Compare `deaths_by_cause.png` and `action_distribution_over_training.png` between run folders.

**Try the approach reward.** `RewardConfig(approach=0.05)` pays a thirsty learner 0.05 per cell it closes on visible water and charges the same per cell it retreats. It is a cold-start crutch: it gives a random policy a gradient toward water before it ever drinks. Compare it against a warm start from imitation, which is the default recommendation and does not change the reward.

**Fewer learners, harder opponents.** `learners_per_game=1` gives a single learner against 23 voting brains and the cleanest signal, but 4 episodes then yield only 4 trajectories per epoch. Raise `episodes_per_epoch` to keep the batch large.

**Continue training.** `run()` resumes; change `trainer.rl.epochs` first if the budget is used up.

**Gradient check.** `tests/test_research.py` checks the MLP's backward pass against finite differences. If you change the loss, add a similar check for `grad_logits`.

## Gotchas

- **`validation_games=0` freezes the champion.** `val_return` is then always `0.0`, so the "better than best" test succeeds only on epoch 0 and `champion` stays the epoch-0 policy. Use `history[-1].genome` for the latest policy in that case.
- **Showcases stay in memory.** Every epoch's recording lives in `history` until the trainer is dropped. For long runs on big maps, set `record_showcase=False`. The recordings never reach `history.json` or the run folder either way.
- The showcase is the epoch's *first* training game, played by the sampling policy (chaos 1) before the update. It is not a greedy game and it is not the champion. To watch the champion, use `champion_brain()` in a scenario or the dashboard's "live" feed mode.
- Advantages are normalised per batch, so `policy_loss` cannot be compared across epochs and does not go to zero as the policy improves.
- The value network sees returns on the raw reward scale while advantages are normalised. `value_loss` therefore scales with the reward weights; doubling every weight roughly quadruples it.
- Rewards are attached to the *latest decision*, and every tick adds to it, so a learner's last decision before dying carries the death penalty, the placement bonus, and possibly a win bonus. Long spans without a decision do not happen, because every living tribute decides every tick.
- The approach term only counts when water (or grass) was visible both before and after the move. Walking out of sight of the lake earns nothing either way, and a learner that cannot see water gets no pull toward it. `in_water` switches the thirsty flag off, so standing in the lake without drinking is not rewarded by this term.
- A warm start loads the policy only. The value network is always fresh, so the first few epochs after a warm start have a poor baseline and noisier updates. Expect `value_loss` to start high.
- The same `spawn` rules as the genetic trainer apply when `workers > 1`: a `__main__` guard, a script file, and pickle-able `config` and `scenario`. See [genetic.md](genetic.md). The first episode's `Recording` is pickled back from its worker as well.
- `play_rl_episode` builds learners with `NeuralBrain(...)` directly, not `create_brain`, so `config.endgame_instinct` does not apply to learners. It does apply to the opponents.
- Learners always use `config.neural` for their architecture. A policy saved with `hidden_layers=(16,)` will not load into a `ReinforceTrainer` built with the default `(64, 32)`, and neither will an `initial_genome` of the wrong size; `set_genome` raises `ValueError`.
- `greedy=True` validation uses chaos 0 and picks the argmax. A policy that looks fine when sampling can score badly greedily if its argmax action is a poor default (for example "attack" with no weapon). Compare `train_return` and `val_return`.
- Each epoch is one gradient step. Thirty epochs is thirty updates, which is a smoke test. Real runs need hundreds of epochs or many more episodes per epoch, and a warm start.
