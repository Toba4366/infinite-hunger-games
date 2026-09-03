"""training/ppo.py - Proximal Policy Optimisation, the method the zombie video uses.

PPO improves on REINFORCE in two ways. First, it reuses each batch of
experience for several gradient passes instead of one. Second, it clips
the probability ratio between the new and old policy so no single update
can move the policy too far, which is what makes the reuse safe. It also
uses generalised advantage estimation (GAE), which blends the one-step
value error with the full return to trade bias against variance.

Everything else (episode collection, the value network, validation,
events, the curriculum) is inherited from `ReinforceTrainer`.
"""

# Settings.
from dataclasses import dataclass

# numpy for the maths.
import numpy as np

# The softmax.
from hunger_games.brain.neural import softmax

# The parent trainer and its settings.
from hunger_games.training.reinforce import ReinforceTrainer, RLConfig


@dataclass
class PPOConfig(RLConfig):
    """The RL settings plus PPO's own knobs."""

    # The ratio clip: new/old probability ratios are kept within 1 +/- this.
    clip_ratio: float = 0.2
    # Gradient passes over each batch of experience.
    update_epochs: int = 4
    # Samples per gradient step.
    minibatch_size: int = 256
    # GAE lambda: 1 is the full return, 0 the one-step error.
    gae_lambda: float = 0.95


class PPOTrainer(ReinforceTrainer):
    """REINFORCE's machinery with PPO's clipped, multi-pass update."""

    # Label for run folders.
    method = "ppo"

    def _advantages(self, rewards: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """GAE advantages and value targets for one trajectory."""
        # Discount and lambda.
        gamma = self.config.reward.discount
        lam = self.rl.gae_lambda
        # Output.
        advantages = np.zeros_like(rewards)
        # Running advantage from the end (the value after the last step is 0: the episode ended).
        running = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            next_value = values[t + 1] if t + 1 < len(values) else 0.0
            delta = rewards[t] + gamma * next_value - values[t]
            running = delta + gamma * lam * running
            advantages[t] = running
        # Targets for the value network.
        return advantages, advantages + values

    def _update(self, episodes: list[dict]) -> tuple[float, float, float]:
        """Several clipped policy passes and value passes over the collected experience."""
        # Gather trajectories.
        xs, acts, advs, targets = [], [], [], []
        for episode in episodes:
            for pid, vectors in episode["vectors"].items():
                if len(vectors) == 0:
                    continue
                values = self.value.forward(vectors)[:, 0]
                advantage, target = self._advantages(episode["rewards"][pid], values)
                xs.append(vectors)
                acts.append(episode["indices"][pid])
                advs.append(advantage)
                targets.append(target)
        # Nothing collected.
        if not xs:
            return 0.0, 0.0, 0.0
        # Stack.
        states = np.concatenate(xs)
        actions = np.concatenate(acts)
        advantages = np.concatenate(advs)
        value_targets = np.concatenate(targets)
        count = len(states)
        # Normalise the advantages.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # The old policy's log-probabilities of the actions taken (fixed for all passes).
        old_log_probs = np.log(softmax(self.policy.forward(states))[np.arange(count), actions] + 1e-12)
        # Bookkeeping for the record.
        policy_losses, value_losses, entropies = [], [], []
        # Several passes.
        for _ in range(self.rl.update_epochs):
            # Shuffle.
            order = self.rng.permutation(count)
            # Minibatches.
            for start in range(0, count, self.rl.minibatch_size):
                batch = order[start : start + self.rl.minibatch_size]
                x, a, adv, old_lp, target = (
                    states[batch],
                    actions[batch],
                    advantages[batch],
                    old_log_probs[batch],
                    value_targets[batch],
                )
                n = len(batch)
                # Policy forward.
                logits, cache = self.policy.forward_cached(x)
                probs = softmax(logits)
                log_probs = np.log(probs + 1e-12)
                chosen = log_probs[np.arange(n), a]
                # Ratio new/old.
                ratio = np.exp(chosen - old_lp)
                # The clipped objective: which samples are inside the clip region (gradient flows) and which are not.
                clipped = np.clip(ratio, 1.0 - self.rl.clip_ratio, 1.0 + self.rl.clip_ratio)
                surrogate = np.minimum(ratio * adv, clipped * adv)
                active = (ratio * adv <= clipped * adv).astype(float)
                # Entropy.
                entropy = -(probs * log_probs).sum(axis=1)
                # Gradient with respect to the logits of -surrogate: -adv * ratio * (onehot - p), only where active.
                one_hot = np.zeros_like(probs)
                one_hot[np.arange(n), a] = 1.0
                grad_logits = -(adv * ratio * active)[:, None] * (one_hot - probs)
                # Entropy bonus gradient.
                grad_logits += self.rl.entropy_bonus * probs * (log_probs + entropy[:, None])
                grad_logits /= n
                # Step.
                self.policy_optimizer.step(self._clip(self.policy.backward(cache, grad_logits)))
                # Value pass.
                v, value_cache = self.value.forward_cached(x)
                v = v[:, 0]
                self.value_optimizer.step(
                    self._clip(self.value.backward(value_cache, (2.0 * (v - target) / n)[:, None]))
                )
                # Record.
                policy_losses.append(float(-surrogate.mean() - self.rl.entropy_bonus * entropy.mean()))
                value_losses.append(float(np.mean((v - target) ** 2)))
                entropies.append(float(entropy.mean()))
        # Means over the passes.
        return float(np.mean(policy_losses)), float(np.mean(value_losses)), float(np.mean(entropies))
