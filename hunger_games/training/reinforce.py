"""training/reinforce.py - reinforcement learning by policy gradient (REINFORCE with a value baseline).

The genetic algorithm scores whole games. Reinforcement learning scores
every action: after each tick the learner gets a reward (see `RewardConfig`
in config.py), and the policy network is nudged to make well-rewarded
actions more likely. A second network, the value network, predicts how much
reward is still to come from a state; subtracting that prediction (the
"baseline") makes the learning signal far less noisy. This is the simplest
actor-critic method, written in plain numpy on top of brain/mlp.py.

Everything a researcher asks for is logged per epoch: policy loss, value
loss, policy entropy, training reward, validation reward on held-out seeds,
survival time, win and kill rates, wall-clock time, and the behaviour
telemetry from research/telemetry.py.
"""

# Parallel episode collection.
# JSON for saving policies and histories.
import json

# Timing.
import time

# Type hints.
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor

# Settings and stats.
from dataclasses import asdict, dataclass, field

# Filesystem paths.
from pathlib import Path

# numpy for everything numeric.
import numpy as np

# Brain construction.
from hunger_games.brain import Brain, create_brain

# The maths engine and optimiser.
from hunger_games.brain.mlp import MLP, Adam

# The neural brain and the softmax.
from hunger_games.brain.neural import NeuralBrain, softmax

# Settings.
from hunger_games.config import SimulationConfig

# The game.
from hunger_games.game import Game

# The perception vector size.
from hunger_games.perception import VECTOR_SIZE

# The body.
from hunger_games.player import Player

# Recordings for the training feed.
from hunger_games.recorder import Recorder, Recording

# Behaviour measurement.
from hunger_games.research.telemetry import BehaviorTelemetry

# Custom setups.
from hunger_games.scenario import Scenario

# The shared training pieces.
from hunger_games.training.common import (
    Curriculum,
    EventLog,
    IterationStats,
    LearnerSpec,
    build_learner,
    learner_ids,
)


@dataclass
class RLConfig:
    """Every knob of the policy-gradient learner."""

    # How many epochs (rounds of collect-then-update) to run.
    epochs: int = 30
    # Games played per epoch to collect experience.
    episodes_per_epoch: int = 4
    # How many tributes in each game are driven by the learning policy (the rest use the config's brain).
    learners_per_game: int = 6
    # Step size for the policy network.
    learning_rate: float = 1e-3
    # Step size for the value network.
    value_learning_rate: float = 3e-3
    # Bonus for keeping the policy varied (prevents collapsing onto one action too early).
    entropy_bonus: float = 0.01
    # Hidden layer widths of the value network.
    value_hidden: tuple[int, ...] = (32,)
    # Games played per epoch with the greedy policy on fixed seeds, for validation.
    validation_games: int = 2
    # The first validation seed (validation game i uses this + i).
    validation_seed: int = 90000
    # Gradients larger than this are scaled down (training stability).
    max_grad_norm: float = 5.0
    # CPU cores for collecting episodes.
    workers: int = 1
    # The trainer's own seed.
    seed: int | None = None
    # Whether to record the first training game of every epoch for the dashboard's training feed.
    record_showcase: bool = True


@dataclass
class EpochStats:
    """What happened in one epoch."""

    # Which epoch (0 first).
    epoch: int
    # Mean policy loss over the update.
    policy_loss: float
    # Mean value loss (squared error of the baseline).
    value_loss: float
    # Mean policy entropy at the decisions made (nats).
    entropy: float
    # Mean total reward per learner episode during collection.
    train_return: float
    # Mean total reward per learner episode on the validation seeds (greedy policy).
    val_return: float
    # Mean ticks survived by learners during collection.
    train_survival: float
    # Mean ticks survived on validation.
    val_survival: float
    # Fraction of learner episodes that won, training.
    win_rate: float
    # Fraction of learner episodes that won, validation.
    val_win_rate: float
    # Mean kills per learner episode, training.
    kill_rate: float
    # Wall-clock seconds this epoch took.
    seconds: float
    # Total seconds since training started.
    cumulative_seconds: float
    # The policy genome after this epoch's update (a copy).
    genome: np.ndarray = field(repr=False, default=None)  # type: ignore[assignment]
    # Behaviour telemetry summary from this epoch's training episodes.
    telemetry: dict = field(default_factory=dict, repr=False)
    # A recording of one real training game from this epoch (the dashboard's training feed).
    showcase: Recording | None = field(default=None, repr=False)

    def to_row(self) -> dict:
        """A JSON-friendly dictionary without the big arrays."""
        # Everything but the genome, telemetry and showcase recording.
        return {k: v for k, v in self.__dict__.items() if k not in ("genome", "telemetry", "showcase")}


def play_rl_episode(
    config: SimulationConfig,
    scenario: Scenario | None,
    genome,
    learner_ids: list[int],
    seed: int,
    greedy: bool,
    record: bool = False,
) -> dict:
    """Play one game with the policy driving `learner_ids`, and return their
    experience: per learner, the perception vectors, chosen menu indices and
    rewards, plus outcomes and behaviour telemetry. A top-level function so
    worker processes can run it.
    """
    # A config copy with this episode's seed.
    game_config = SimulationConfig(**{**config.to_dict_raw(), "seed": seed})
    # The learners' brains, kept so we can read their last choice.
    learners: dict[int, Brain] = {}
    # The learner description: a flat neural genome (the common case) or a LearnerSpec of any kind.
    spec = genome if isinstance(genome, LearnerSpec) else LearnerSpec("neural", genome, game_config.neural)

    # Brain factory: learners get the policy, everyone else the config's brain.
    def factory(index: int, rng: np.random.Generator) -> Brain:
        # A learner.
        if index in learner_ids:
            # Greedy validation uses chaos 0 (argmax); training samples at temperature 1.
            brain = build_learner(spec, 0.0 if greedy else 1.0, rng)
            # Remember it.
            learners[index] = brain
            return brain
        # An opponent.
        return create_brain(
            game_config.brain_name, game_config.chaos, rng, game_config.neural, game_config.endgame_instinct
        )

    # The game.
    game = Game(game_config, 0, brain_factory=factory, scenario=scenario)
    # Telemetry for the learners only.
    telemetry = BehaviorTelemetry(game.arena.width, game.arena.height, set(learner_ids)).attach(game)
    # Experience per learner: vectors, indices, rewards.
    vectors: dict[int, list] = {i: [] for i in learner_ids}
    # Indices.
    indices: dict[int, list] = {i: [] for i in learner_ids}
    # Rewards.
    rewards: dict[int, list] = {i: [] for i in learner_ids}
    # The reward weights.
    reward = game_config.reward
    # Previous stats per learner, for shaping rewards from changes.
    previous: dict[int, tuple] = {}
    # Whether each learner has already received a death penalty.
    dead_paid: set[int] = set()
    # The last known distances to water and grass per learner, for the approach reward.
    last_distances: dict[int, tuple[float, float, bool, bool]] = {}

    # Decision hook: record the state and the chosen index.
    def on_decision(player: Player, perception, action) -> None:
        # Only learners.
        if player.player_id not in learners:
            return
        # Approach shaping: reward the previous action for closing the distance to what was needed.
        pid = player.player_id
        previous_distances = last_distances.get(pid)
        if previous_distances is not None and rewards[pid]:
            water_before, grass_before, was_thirsty, was_hungry = previous_distances
            if was_thirsty and water_before != float("inf") and perception.water_distance != float("inf"):
                rewards[pid][-1] += reward.approach * (water_before - perception.water_distance)
            if was_hungry and grass_before != float("inf") and perception.grass_distance != float("inf"):
                rewards[pid][-1] += reward.approach * (grass_before - perception.grass_distance)
        # Remember this tick's distances and needs for the next comparison.
        last_distances[pid] = (
            perception.water_distance,
            perception.grass_distance,
            perception.thirst < 0.5 and not perception.in_water,
            perception.hunger < 0.5 and perception.food_count == 0,
        )
        # The vector and the index (decide_index stored the probabilities; the index is recovered from the action).
        vectors[player.player_id].append(perception.to_vector())
        # The chosen index (every learner brain records it).
        indices[player.player_id].append(getattr(learners[player.player_id], "last_index", 0))
        # A placeholder reward, filled in at the end of the tick.
        rewards[player.player_id].append(0.0)

    # Tick hook: compute this tick's reward for every learner that decided.
    def on_tick(current: Game) -> None:
        # Each learner.
        for pid in learner_ids:
            # The body.
            player = current.player_by_id[pid]
            # Their previous bars.
            prev = previous.get(pid, (player.thirst, player.hunger, player.health, player.kills))
            # Start from nothing.
            r = 0.0
            # Alive: survival and shaping.
            if player.alive:
                r += reward.survive_tick
                # Health lost.
                r += reward.damage_taken * max(0.0, prev[2] - player.health)
                # Needs restored while low.
                if prev[0] < 0.5:
                    r += reward.need_gain * max(0.0, player.thirst - prev[0])
                if prev[1] < 0.5:
                    r += reward.need_gain * max(0.0, player.hunger - prev[1])
            # Just died: the penalty, once.
            elif pid not in dead_paid:
                r += reward.death
                dead_paid.add(pid)
            # Kills this tick.
            r += reward.kill * (player.kills - prev[3])
            # Remember for next tick.
            previous[pid] = (player.thirst, player.hunger, player.health, player.kills)
            # Attach to the last decision, if there was one this tick.
            if rewards[pid] and len(rewards[pid]) == len(vectors[pid]):
                rewards[pid][-1] += r

    # Register.
    game.decision_hooks.append(on_decision)
    # And the tick hook (after telemetry's, so death bookkeeping is done).
    game.tick_hooks.append(on_tick)
    # Play, recording every tick if asked.
    recording = Recorder(game).record_all() if record else None
    # Finish (a no-op if the recorder already ran the game to the end).
    game.run()
    # End-of-game bonuses.
    n = len(game.players)
    # Each learner.
    outcomes = {}
    for pid in learner_ids:
        # The body.
        player = game.player_by_id[pid]
        # Placement bonus.
        bonus = reward.placement * (n - (player.placement or n)) / max(1, n - 1)
        # Win bonus.
        if player.alive and len(game.alive_players) == 1:
            bonus += reward.win
        # Attach to the last decision.
        if rewards[pid]:
            rewards[pid][-1] += bonus
        # Outcomes.
        outcomes[pid] = {
            "return": float(sum(rewards[pid])),
            "survival": game.death_ticks.get(pid, game.tick),
            "won": int(player.alive and len(game.alive_players) == 1),
            "kills": player.kills,
        }
    # Package (learner_won: did any learner copy win this game?).
    return {
        "learner_won": bool(any(o["won"] for o in outcomes.values())),
        "vectors": {pid: np.asarray(v) for pid, v in vectors.items()},
        "indices": {pid: np.asarray(v, dtype=int) for pid, v in indices.items()},
        "rewards": {pid: np.asarray(v, dtype=float) for pid, v in rewards.items()},
        "outcomes": outcomes,
        "telemetry": telemetry.summary(),
        "recording": recording,
    }


def _run_episode_job(args: tuple) -> dict:
    """Unpack a job tuple for the process pool."""
    # Forward.
    return play_rl_episode(*args)


class ReinforceTrainer:
    """Trains a NeuralBrain policy by REINFORCE with a learned value baseline."""

    # Label for run folders and champion files (PPO overrides it).
    method = "reinforce"

    def __init__(
        self,
        config: SimulationConfig,
        rl: RLConfig,
        scenario: Scenario | None = None,
        initial_genome: np.ndarray | None = None,
        curriculum: Curriculum | None = None,
    ) -> None:
        """Create a fresh policy and value network, or start the policy from a given genome (a warm start)."""
        # Settings.
        self.config = config
        # Learner settings.
        self.rl = rl
        # The curriculum (opponents grow as the learner improves), if any.
        self.curriculum = curriculum
        # The event monitor.
        self.events = EventLog()
        # The unified per-iteration history every method shares.
        self.learning_history: list[IterationStats] = []
        # The best mean score seen (for "new record" events).
        self.best_mean_score = -np.inf
        # Optional custom setup.
        self.scenario = scenario
        # The trainer's own randomness.
        self.rng = np.random.default_rng(rl.seed)
        # The policy network (same architecture the NeuralBrain uses).
        self.policy = NeuralBrain(chaos=1.0, config=config.neural, rng=self.rng).network
        # A warm start: begin from an existing policy (an imitation-pretrained one, or an earlier champion).
        if initial_genome is not None:
            self.policy.set_genome(np.asarray(initial_genome, dtype=float))
        # The value network: state in, one number out.
        self.value = MLP([VECTOR_SIZE, *rl.value_hidden, 1], config.neural.activation, "xavier_uniform", rng=self.rng)
        # Optimisers.
        self.policy_optimizer = Adam(self.policy, rl.learning_rate)
        # Value optimiser.
        self.value_optimizer = Adam(self.value, rl.value_learning_rate)
        # History.
        self.history: list[EpochStats] = []
        # Epoch counter.
        self.epoch = 0
        # Stop flag.
        self._stop = False
        # Training start time.
        self._started: float | None = None
        # The genome with the best validation return so far.
        self.best_genome: np.ndarray | None = None
        # Its validation return.
        self.best_val_return = -np.inf

    # ------------------------------------------------------------ episodes

    @property
    def settings(self):
        """The trainer's own settings dataclass (every trainer exposes this name for run folders)."""
        # The RL settings.
        return self.rl

    def _learner_ids(self) -> list[int]:
        """Which tribute slots the policy drives (spread across the roster)."""
        # Shared rule.
        return learner_ids(self.config.num_players, self.rl.learners_per_game)

    def learner_spec(self) -> LearnerSpec:
        """The current policy as something a worker can rebuild."""
        # Neural.
        return LearnerSpec("neural", self.policy.genome().copy(), self.config.neural)

    def champion_spec(self) -> LearnerSpec:
        """The best policy so far."""
        # Neural.
        return LearnerSpec("neural", np.asarray(self.champion, dtype=float), self.config.neural)

    def _apply_curriculum(self) -> None:
        """Size the roster for the current curriculum stage: learner copies plus that stage's opponents."""
        # Nothing to do without a curriculum.
        if self.curriculum is None:
            return
        # Learners plus opponents.
        players = min(self.rl.learners_per_game, 24) + self.curriculum.opponents
        # Apply.
        self.config = SimulationConfig(**{**self.config.to_dict_raw(), "num_players": players})

    def _record_iteration(
        self,
        scores: list[float],
        entropy: float,
        mean_length: float,
        win_rate: float,
        val_score: float,
        seconds: float,
        extra: dict,
        telemetry: dict,
        showcase,
        val_win_rate: float = 0.0,
    ) -> IterationStats:
        """Append a unified IterationStats, log events, and advance the curriculum."""
        # Mean.
        mean_score = float(np.mean(scores)) if scores else 0.0
        # Stage info.
        stage = self.curriculum.stage if self.curriculum is not None else 0
        opponents = self.curriculum.opponents if self.curriculum is not None else self.config.num_players - 1
        # The stats.
        stats = IterationStats(
            iteration=len(self.learning_history),
            scores=list(scores),
            mean_score=mean_score,
            best_score=float(max(scores)) if scores else 0.0,
            entropy=entropy,
            mean_length=mean_length,
            win_rate=win_rate,
            val_score=val_score,
            val_win_rate=val_win_rate,
            seconds=seconds,
            cumulative_seconds=(self.learning_history[-1].cumulative_seconds if self.learning_history else 0.0)
            + seconds,
            stage=stage,
            opponents=opponents,
            extra=extra,
            learner=self.learner_spec().genome,
            telemetry=telemetry,
            showcase=showcase,
        )
        # Keep.
        self.learning_history.append(stats)
        # Events.
        self.events.add(
            "rollout",
            f"iteration {stats.iteration}: mean score {mean_score:.2f}, length {mean_length:.0f} ticks, win rate {win_rate:.2f}",
        )
        if mean_score > self.best_mean_score:
            self.best_mean_score = mean_score
            self.events.add("record", f"new best mean score {mean_score:.2f}")
        # Curriculum (promotion is judged on the validation games when there are any, else on training games).
        judged_win = val_win_rate if self.rl.validation_games > 0 else win_rate
        if self.curriculum is not None and self.curriculum.observe(mean_score, judged_win):
            self.events.add(
                "curriculum", f"promoted to stage {self.curriculum.stage}: {self.curriculum.opponents} opponents"
            )
        # Done.
        return stats

    def step(self, on_progress: Callable[[int, int], None] | None = None) -> IterationStats:
        """One iteration in the shared shape (every trainer has this method)."""
        # Run an epoch and hand back the unified stats appended for it.
        self.step_epoch(on_progress)
        return self.learning_history[-1]

    def _collect(
        self,
        seeds: list[int],
        greedy: bool,
        on_progress: Callable[[int, int], None] | None = None,
        record_first: bool = False,
    ) -> list[dict]:
        """Play one episode per seed, in parallel if asked; optionally record the first one."""
        # The policy as a learner spec.
        spec = self.learner_spec()
        # Learners.
        learners = self._learner_ids()
        # Jobs (only the first records, when asked).
        jobs = [
            (self.config, self.scenario, spec, learners, seed, greedy, record_first and index == 0)
            for index, seed in enumerate(seeds)
        ]
        # Results.
        results = []
        # Parallel.
        if self.rl.workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=self.rl.workers) as pool:
                for index, result in enumerate(pool.map(_run_episode_job, jobs)):
                    results.append(result)
                    if on_progress is not None:
                        on_progress(index + 1, len(jobs))
        # Sequential.
        else:
            for index, job in enumerate(jobs):
                results.append(_run_episode_job(job))
                if on_progress is not None:
                    on_progress(index + 1, len(jobs))
        # Done.
        return results

    # -------------------------------------------------------------- update

    def _returns(self, rewards: np.ndarray) -> np.ndarray:
        """Discounted returns: each step's reward plus discounted future rewards."""
        # Output.
        returns = np.zeros_like(rewards)
        # Running total from the end.
        running = 0.0
        # Backward.
        for t in range(len(rewards) - 1, -1, -1):
            running = rewards[t] + self.config.reward.discount * running
            returns[t] = running
        # Done.
        return returns

    def _update(self, episodes: list[dict]) -> tuple[float, float, float]:
        """One gradient step on the policy and the value network from the collected experience."""
        # Gather every learner trajectory.
        xs, acts, rets = [], [], []
        # Each episode.
        for episode in episodes:
            # Each learner.
            for pid, vectors in episode["vectors"].items():
                # Skip empty trajectories.
                if len(vectors) == 0:
                    continue
                xs.append(vectors)
                acts.append(episode["indices"][pid])
                rets.append(self._returns(episode["rewards"][pid]))
        # Nothing collected.
        if not xs:
            return 0.0, 0.0, 0.0
        # Stack.
        states = np.concatenate(xs)
        # Actions.
        actions = np.concatenate(acts)
        # Returns.
        returns = np.concatenate(rets)
        # Number of samples.
        count = len(states)
        # Value baseline.
        values, value_cache = self.value.forward_cached(states)
        # Flatten.
        values = values[:, 0]
        # Advantages, normalised for stability.
        advantages = returns - values
        # Normalise.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # Policy forward.
        logits, policy_cache = self.policy.forward_cached(states)
        # Probabilities.
        probs = softmax(logits)
        # Log probabilities.
        log_probs = np.log(probs + 1e-12)
        # Entropy per sample.
        entropy = -(probs * log_probs).sum(axis=1)
        # One-hot of the chosen actions.
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(count), actions] = 1.0
        # Gradient of the policy loss with respect to the logits:
        # for -log p(a) * adv it is (p - onehot) * adv; for -beta * H it is beta * p * (log p + H).
        grad_logits = (probs - one_hot) * advantages[:, None] + self.rl.entropy_bonus * probs * (
            log_probs + entropy[:, None]
        )
        # Mean over the batch.
        grad_logits /= count
        # Backprop.
        policy_grads = self.policy.backward(policy_cache, grad_logits)
        # Clip.
        policy_grads = self._clip(policy_grads)
        # Step.
        self.policy_optimizer.step(policy_grads)
        # Value loss: mean squared error.
        value_loss = float(np.mean((values - returns) ** 2))
        # Its gradient at the output.
        grad_value = (2.0 * (values - returns) / count)[:, None]
        # Backprop.
        value_grads = self._clip(self.value.backward(value_cache, grad_value))
        # Step.
        self.value_optimizer.step(value_grads)
        # Policy loss for the record (before the update).
        policy_loss = float(
            -(log_probs[np.arange(count), actions] * advantages).mean() - self.rl.entropy_bonus * entropy.mean()
        )
        # Done.
        return policy_loss, value_loss, float(entropy.mean())

    def _clip(self, grads: list[tuple[np.ndarray, np.ndarray]]) -> list[tuple[np.ndarray, np.ndarray]]:
        """Scale all gradients down if their combined length exceeds max_grad_norm."""
        # Combined length.
        norm = np.sqrt(sum(float((grad_w * grad_w).sum() + (grad_b * grad_b).sum()) for grad_w, grad_b in grads))
        # Fine.
        if norm <= self.rl.max_grad_norm:
            return grads
        # Scale.
        scale = self.rl.max_grad_norm / (norm + 1e-12)
        # Apply.
        return [(grad_w * scale, grad_b * scale) for grad_w, grad_b in grads]

    # --------------------------------------------------------------- epoch

    @staticmethod
    def _outcome_means(episodes: list[dict]) -> tuple[float, float, float, float]:
        """Mean return, survival, game-level win rate and kill rate over the episodes."""
        # All outcomes.
        outcomes = [o for episode in episodes for o in episode["outcomes"].values()]
        # Empty.
        if not outcomes:
            return 0.0, 0.0, 0.0, 0.0
        # Means (a game is won when any learner copy was the victor).
        return (
            float(np.mean([o["return"] for o in outcomes])),
            float(np.mean([o["survival"] for o in outcomes])),
            float(np.mean([episode.get("learner_won", False) for episode in episodes])),
            float(np.mean([o["kills"] for o in outcomes])),
        )

    def step_epoch(self, on_progress: Callable[[int, int], None] | None = None) -> EpochStats:
        """Collect episodes, update, validate, and record."""
        # Clock.
        if self._started is None:
            self._started = time.time()
        # Start.
        started = time.time()
        # Size the roster for the curriculum stage.
        self._apply_curriculum()
        # Training seeds.
        seeds = [int(self.rng.integers(2**31 - 1)) for _ in range(self.rl.episodes_per_epoch)]
        # Collect, recording the first game for the training feed.
        episodes = self._collect(seeds, greedy=False, on_progress=on_progress, record_first=self.rl.record_showcase)
        # Update.
        policy_loss, value_loss, entropy = self._update(episodes)
        # Training outcomes.
        train_return, train_survival, win_rate, kill_rate = self._outcome_means(episodes)
        # Validation on fixed seeds with the greedy policy.
        val_seeds = [self.rl.validation_seed + i for i in range(self.rl.validation_games)]
        # Play.
        validation = self._collect(val_seeds, greedy=True) if self.rl.validation_games > 0 else []
        # Validation outcomes.
        val_return, val_survival, val_win_rate, _ = (
            self._outcome_means(validation) if validation else (0.0, 0.0, 0.0, 0.0)
        )
        # Track the best policy by validation return.
        if val_return > self.best_val_return or self.best_genome is None:
            self.best_val_return = val_return
            self.best_genome = self.policy.genome().copy()
        # Merge the telemetry of the training episodes.
        telemetry = BehaviorTelemetry.merge([episode["telemetry"] for episode in episodes])
        # Stats.
        stats = EpochStats(
            epoch=self.epoch,
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy=entropy,
            train_return=train_return,
            val_return=val_return,
            train_survival=train_survival,
            val_survival=val_survival,
            win_rate=win_rate,
            val_win_rate=val_win_rate,
            kill_rate=kill_rate,
            seconds=time.time() - started,
            cumulative_seconds=time.time() - self._started,
            genome=self.policy.genome().copy(),
            telemetry=telemetry,
            showcase=episodes[0].get("recording") if episodes else None,
        )
        # Record.
        self.history.append(stats)
        # Count.
        self.epoch += 1
        # The unified record.
        scores = [o["return"] for episode in episodes for o in episode["outcomes"].values()]
        self._record_iteration(
            scores,
            entropy,
            train_survival,
            win_rate,
            val_return,
            stats.seconds,
            {"policy_loss": policy_loss, "value_loss": value_loss},
            telemetry,
            stats.showcase,
            val_win_rate,
        )
        # Done.
        return stats

    def run(
        self,
        on_epoch: Callable[[EpochStats], None] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[EpochStats]:
        """Run every epoch (or until `stop()`)."""
        # Reset.
        self._stop = False
        # Loop.
        while self.epoch < self.rl.epochs and not self._stop:
            stats = self.step_epoch(on_progress)
            if on_epoch is not None:
                on_epoch(stats)
        # History.
        return self.history

    def stop(self) -> None:
        """Ask a running `run()` to finish after the current epoch."""
        # Flag.
        self._stop = True

    # ------------------------------------------------------------- outputs

    @property
    def champion(self) -> np.ndarray | None:
        """The policy genome with the best validation return so far (the current policy if none validated)."""
        # Best, or current.
        return self.best_genome if self.best_genome is not None else self.policy.genome()

    def champion_brain(self, chaos: float = 0.0) -> NeuralBrain:
        """A NeuralBrain loaded with the champion policy."""
        # Build.
        brain = NeuralBrain(chaos=chaos, config=self.config.neural, rng=self.rng)
        # Load.
        brain.set_genome(self.champion)
        # Done.
        return brain

    def save_policy(self, path: str | Path) -> None:
        """Write the champion policy (and the value network) to JSON, in the same shape as a GA champion file."""
        # Data.
        data = {
            "brain_name": "neural",
            "neural": asdict(self.config.neural),
            "genome": self.champion.tolist(),
            "value_genome": self.value.genome().tolist(),
            "value_hidden": list(self.rl.value_hidden),
            "fitness": float(self.best_val_return) if np.isfinite(self.best_val_return) else 0.0,
            "epochs": len(self.history),
            "method": self.method,
        }
        # Write.
        Path(path).write_text(json.dumps(data))

    def save_champion(self, path: str | Path) -> None:
        """Alias of save_policy, so every trainer has the same method name."""
        # Delegate.
        self.save_policy(path)

    def history_rows(self) -> list[dict]:
        """The history as JSON-friendly rows."""
        # Rows.
        return [stats.to_row() for stats in self.history]
