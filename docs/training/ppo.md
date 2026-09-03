# `ppo.py`

**Source:** [hunger_games/training/ppo.py](../../hunger_games/training/ppo.py)
**Depends on:** `dataclasses` (standard library); `numpy`; [brain/neural.py](../brain/neural.md) (`softmax`); [training/reinforce.py](reinforce.md) (`ReinforceTrainer`, `RLConfig`)
**Used by:** [training/__init__.py](init.md) (re-exports `PPOConfig`, `PPOTrainer`); [research/comparison.py](../research/comparison.md) (the `"ppo"` method); [ui/session.py](../ui/session.md) (`Session.start_training` builds a `PPOTrainer` for the `"ppo"` method); [ui/app.py](../ui/app.md) (`PPOConfig` backs the clip ratio and passes-per-batch controls); `tests/test_methods.py`

## Purpose

Proximal Policy Optimisation, the method the zombie video uses. It is a small file because it inherits almost everything from `ReinforceTrainer` in [reinforce.md](reinforce.md): episode collection, the value network, validation, events, the curriculum, the champion and the run folder. It replaces one thing, the update.

PPO improves on REINFORCE in two ways:

1. It reuses each batch of experience for several gradient passes instead of one.
2. It clips the probability ratio between the new and old policy, so no single update can move the policy too far. That is what makes the reuse safe.

It also uses generalised advantage estimation (GAE), which blends the one-step value error with the full return to trade bias against variance.

## Concepts you need

**Old policy and new policy.** Experience is collected with the policy as it was at the start of the epoch, the old policy. During the update the weights change, so the new policy assigns different probabilities to the same actions. REINFORCE takes one step and stops, so old and new never drift far. PPO takes many steps and needs a guard.

**The probability ratio.** `r = p_new(a | s) / p_old(a | s)`. It is 1 before the first step. Above 1 means the new policy likes the action more than the old one did.

**The clipped surrogate.** The objective for one sample is `min(r * A, clip(r, 1 - eps, 1 + eps) * A)`, with `A` the advantage and `eps = clip_ratio`. When `A > 0`, the objective stops rising once `r` passes `1 + eps`, so there is no reason to push the action's probability further. When `A < 0`, it stops once `r` drops below `1 - eps`. The minimum takes the pessimistic side, so the clip never makes the objective look better than the unclipped one.

**GAE.** The advantage estimate `A_t = delta_t + (gamma * lambda) * A_{t+1}` with `delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)`. With `lambda = 1` it is the discounted return minus the value (what REINFORCE uses, low bias, high variance). With `lambda = 0` it is the one-step error (high bias if the value network is wrong, low variance). The default 0.95 is close to the full return but a little smoother.

**Minibatches and passes.** One pass over the data is one epoch of the update (`update_epochs`, not to be confused with the trainer's `epochs`). Each pass shuffles the samples and steps once per minibatch of `minibatch_size`.

**Why this is more stable than REINFORCE.** REINFORCE's single step has to be small, because a large one can push the policy somewhere the collected advantages no longer describe. PPO can take several steps, because the clip removes the gradient from any sample whose ratio has already moved past the clip edge in the direction its advantage wanted. The policy improves as much as the data supports and no further. That is why the comparison's method notes call PPO the most stable of the reward methods.

Side by side:

| | REINFORCE (`ReinforceTrainer._update`) | PPO (`PPOTrainer._update`) |
| --- | --- | --- |
| Advantage | discounted return minus value | GAE with `gae_lambda` |
| Objective per sample | `log p(a) * A` | `min(r * A, clip(r) * A)` |
| Passes over a batch | 1 | `update_epochs` |
| Steps per pass | 1 (the whole batch) | one per minibatch of `minibatch_size` |
| Guard against a large step | gradient-norm clipping only | ratio clipping plus gradient-norm clipping |
| Value targets | the discounted returns | `advantages + values` from GAE |

**The zombie video's lesson about survival rewards.** The video's agent was first rewarded for staying alive. It learned to run from the zombies and never fight, because running was the safest way to collect the survival reward. The fix was to make the survival reward small next to the reward for the actual goal. This project's `RewardConfig` follows that: `survive_tick` is `0.01` per tick, while a kill is `1.0`, a win `5.0`, a death `-3.0` and first place `2.0`. Over a full 576-tick game the survival reward adds up to `5.76`, about the size of one win, so it matters but cannot dominate. If you raise `survive_tick`, expect the learner to hide.

## Walkthrough

### `PPOConfig`

```python
@dataclass
class PPOConfig(RLConfig):
```

Every field of `RLConfig` (see [reinforce.md](reinforce.md)) plus PPO's own knobs.

| Field | Default | Meaning |
| --- | --- | --- |
| `clip_ratio` | `0.2` | New/old probability ratios are kept within `1 +/- clip_ratio` |
| `update_epochs` | `4` | Gradient passes over each batch of experience |
| `minibatch_size` | `256` | Samples per gradient step |
| `gae_lambda` | `0.95` | GAE lambda: 1 is the full return, 0 the one-step error |

The inherited fields that matter most here are `learning_rate`, `entropy_bonus`, `max_grad_norm` and `episodes_per_epoch`. The discount comes from `SimulationConfig.reward.discount` (`0.98`), as for REINFORCE.

### `PPOTrainer`

```python
class PPOTrainer(ReinforceTrainer):
    method = "ppo"
```

REINFORCE's machinery with PPO's clipped, multi-pass update. `method` is the label the dashboard and run folders use. The constructor, `step_epoch`, `step`, `run`, `stop`, `_collect`, `_clip`, `_record_iteration`, `_apply_curriculum`, `champion`, `champion_brain`, `save_policy`, `save_champion` and `history_rows` are all inherited unchanged. The settings object is reached as `self.rl`, which is a `PPOConfig` here.

#### `_advantages(rewards, values)`

```python
def _advantages(self, rewards: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
```

GAE advantages and value targets for one trajectory. `gamma` is `config.reward.discount` and `lam` is `rl.gae_lambda`. Walking backwards from the last step:

```
next_value = values[t + 1]  if t + 1 < len(values)  else 0.0
delta      = rewards[t] + gamma * next_value - values[t]
running    = delta + gamma * lam * running
advantages[t] = running
```

The value after the last step is 0 because the episode ended. Returns `(advantages, advantages + values)`; the second array is the target the value network is trained toward. Note that `advantages + values` equals `values` plus the GAE estimate, which is a lambda-weighted mix of the one-step target and the full return.

Worked example with rewards `[0, 0, 5]`, values `[1, 1, 1]`, `gamma = 0.98`, `lam = 0.95`:

| `t` | `delta` | `running` |
| --- | --- | --- |
| 2 | `5 + 0 - 1 = 4.0` | `4.0` |
| 1 | `0 + 0.98 * 1 - 1 = -0.02` | `-0.02 + 0.931 * 4.0 = 3.704` |
| 0 | `-0.02` | `-0.02 + 0.931 * 3.704 = 3.428` |

The value targets are `[4.428, 4.704, 5.0]`.

#### `_update(episodes)`

```python
def _update(self, episodes: list[dict]) -> tuple[float, float, float]:
```

Several clipped policy passes and value passes over the collected experience. Returns `(policy_loss, value_loss, entropy)` averaged over every minibatch of every pass.

1. **Gather.** For each learner trajectory with at least one decision: `values = self.value.forward(vectors)[:, 0]`, then `advantage, target = self._advantages(rewards, values)`. Collect the vectors, the action indices, the advantages and the targets. Return `(0.0, 0.0, 0.0)` if nothing was collected.
2. **Stack** into `states` (N x 50), `actions` (N), `advantages` (N), `value_targets` (N).
3. **Normalise the advantages** to mean 0 and standard deviation 1, as REINFORCE does.
4. **Freeze the old policy.** `old_log_probs = log(softmax(policy.forward(states))[range(N), actions] + 1e-12)`. This is computed once, before any step, and stays fixed for every pass.
5. **For each of `update_epochs` passes:** shuffle the N indices with `self.rng.permutation`, then for each minibatch of `minibatch_size`:
   - Forward the policy with a cache: `logits, cache = policy.forward_cached(x)`, `probs = softmax(logits)`, `log_probs = log(probs + 1e-12)`, and `chosen = log_probs[range(n), a]`.
   - The ratio: `ratio = exp(chosen - old_lp)`.
   - The clipped objective: `clipped = clip(ratio, 1 - clip_ratio, 1 + clip_ratio)` and `surrogate = minimum(ratio * adv, clipped * adv)`.
   - The active mask: `active = (ratio * adv <= clipped * adv)` as floats. A sample is active when the unclipped term is the minimum, which is where the gradient flows.
   - Entropy per sample: `entropy = -(probs * log_probs).sum(axis=1)`.
   - The gradient at the logits (see below), divided by `n`.
   - Policy step: `policy_optimizer.step(self._clip(policy.backward(cache, grad_logits)))`.
   - Value pass on the same minibatch: `v, value_cache = value.forward_cached(x)`, gradient `2 * (v - target) / n`, backward, clip, `value_optimizer.step`.
   - Record `-surrogate.mean() - entropy_bonus * entropy.mean()`, `mean((v - target) ** 2)` and `entropy.mean()`.
6. Return the means of the three recorded lists.

**The gradient with respect to the logits.** Let `z` be the logits, `p = softmax(z)`, `a` the chosen action, `A` its advantage and `r = exp(log p_a - log p_a_old)`. Since `d log p_a / d z_j = 1[j = a] - p_j` for a softmax, `d r / d z_j = r * (1[j = a] - p_j)`. The loss is `-surrogate`. Where the sample is active the surrogate is `r * A`, so

```
d(-surrogate) / d z_j = -A * r * (onehot_j - p_j)
```

Where the sample is not active the surrogate is `clipped * A`, which does not depend on `z`, so the gradient is 0. The code writes this in one line:

```python
grad_logits = -(adv * ratio * active)[:, None] * (one_hot - probs)
```

Then the entropy bonus is added, `entropy_bonus * probs * (log_probs + entropy[:, None])`, the same term REINFORCE uses, and everything is divided by `n` for a minibatch mean.

**What `active` means in words.** With `A > 0` the sample is active while `r <= 1 + eps`: the policy may keep raising the action's probability until it is 20 percent more likely than under the old policy, then the sample stops contributing. With `A < 0` it is active while `r >= 1 - eps`. The `<=` comparison keeps a sample active on the boundary, where `ratio == clipped`. On the first minibatch of the first pass every ratio is 1, so every sample is active and the step is a plain policy-gradient step with GAE advantages.

**Worked example of one sample across passes.** Take `clip_ratio = 0.2`, a normalised advantage `A = +1.5` and an action the old policy chose with probability `0.10`.

| Pass | `p_new(a)` | `ratio` | `clipped` | `surrogate` | `active` |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.10 | 1.00 | 1.00 | 1.50 | 1 (gradient flows) |
| 2 | 0.11 | 1.10 | 1.10 | 1.65 | 1 |
| 3 | 0.12 | 1.20 | 1.20 | 1.80 | 1 (on the boundary) |
| 4 | 0.13 | 1.30 | 1.20 | 1.80 | 0 (no gradient from this sample) |

Once the action is 20 percent more likely than under the old policy, this sample stops pushing. Other samples in the minibatch may still be active, so the weights can still move, but not because of this one. With `A = -1.5` the table mirrors: the sample is active while the ratio is at least 0.8.

**Old versus new value network.** The advantages and targets are computed once with the value network as it was before the update. The value network is then stepped on every minibatch of every pass toward those fixed targets. The policy's ratios use the frozen `old_log_probs`, but the policy weights move on every minibatch, so later minibatches in a pass see a slightly different policy than earlier ones. That is normal PPO.

## How to use it / experiment

**A minimal run.**

```python
from hunger_games.config import SimulationConfig
from hunger_games.training import PPOConfig, PPOTrainer

config = SimulationConfig(width=80, height=80, max_days=8)
trainer = PPOTrainer(config, PPOConfig(epochs=20, episodes_per_epoch=4, seed=0))
trainer.run(on_epoch=lambda e: print(e.epoch, round(e.val_return, 2), round(e.entropy, 2)))
trainer.save_champion("ppo.json")
```

`run` yields `EpochStats` (the REINFORCE record); `trainer.learning_history` holds the shared `IterationStats`.

**Warm start from imitation, with the curriculum.** The flow the dashboard's help text suggests:

```python
from hunger_games.training import Curriculum, CurriculumConfig, ImitationConfig, ImitationTrainer

student = ImitationTrainer(config, ImitationConfig(seed=0))
student.run()
trainer = PPOTrainer(
    config, PPOConfig(epochs=100, seed=0), initial_genome=student.champion, curriculum=Curriculum(CurriculumConfig())
)
trainer.run()
```

**Compare with REINFORCE on the same seeds.** Build a `ReinforceTrainer` and a `PPOTrainer` with the same `config`, the same `seed` and the same episode counts, run both, and plot `val_score` from each `learning_history`. `experiments/run_comparison.py` does this for every method and overlays the curves.

**Knobs to try.**

| Change | Effect |
| --- | --- |
| `clip_ratio=0.1` | Smaller steps per epoch, safer, slower |
| `update_epochs=1` | One pass; close to REINFORCE with GAE and minibatches |
| `update_epochs=10` | More reuse of each batch; watch for entropy collapsing |
| `gae_lambda=1.0` | Full-return advantages, noisier |
| `gae_lambda=0.0` | One-step advantages; only good once `value_loss` is low |
| `minibatch_size=64` | More steps per pass, more noise per step |

**Read the losses.** `extra_policy_loss` in `learning.json` is `-surrogate - beta * H` averaged over minibatches. Because the advantages are normalised and the ratio starts at 1, it hovers near 0 like REINFORCE's. Watch `extra_value_loss` fall and `entropy` drift down slowly. A fast entropy crash means too many passes or too small a clip.

## Gotchas

- `PPOTrainer.method` is `"ppo"`, and the inherited `save_policy` writes that label into `champion.json` (`"method": "ppo"`).
- `save_run(trainer, "ppo", ...)` draws the REINFORCE chart set, because `training_run_plots` treats every method other than `"imitation"`, `"genetic"` and `"neat"` as REINFORCE. The rows have the same keys, so the charts are correct.
- `update_epochs` is the number of passes over one batch. `epochs` (inherited) is the number of collect-then-update rounds. Four passes on four episodes is still a small amount of data; PPO gets its stability from the clip, not from more data.
- The value targets are fixed before the passes, so a large `update_epochs` can overfit the value network to one batch. The recorded `value_loss` then looks better than it is on fresh episodes.
- The per-minibatch value gradient is `2 * (v - target) / n` with `n` the minibatch size, so the effective value learning rate is the same per step regardless of `minibatch_size`; more minibatches per pass means more steps.
- `minibatch_size` larger than the number of samples gives one minibatch per pass. The last minibatch of a pass is whatever is left and can be small.
- Every inherited gotcha of [reinforce.md](reinforce.md) applies: `validation_games=0` freezes the champion, showcases stay in memory, `workers > 1` needs a `__main__` guard on macOS, and learners are built without `endgame_instinct`.
- The old log-probabilities are recomputed from the current policy at the start of `_update`, not stored during collection. They are equal, because nothing changes the policy between collection and the update within an epoch.
