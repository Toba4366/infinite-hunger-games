"""training/imitation.py - pretraining a network by copying the voting brain.

A fresh network chooses actions at random, so it usually dies of thirst on
day three before any reward or fitness signal can teach it otherwise.
Imitation learning (also called behaviour cloning) fixes that: play games
with a competent teacher (the voting brain), record every (perception,
action) pair, and train the network to predict the teacher's action. That
is ordinary supervised learning with a cross-entropy loss and Adam, using
the MLP's backpropagation. The result is a network with working instincts
that the genetic or policy-gradient trainer can then improve from (a "warm
start").

Every epoch logs training and validation loss and accuracy, and plays a
greedy validation game so survival, win rate, behaviour telemetry and a
showcase recording are available exactly as for the other trainers.
"""

# Parallel demonstration collection.
# JSON for saving.
import json

# Timing.
import time

# Type hints.
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor

# Settings and stats.
from dataclasses import asdict, dataclass, field

# Paths.
from pathlib import Path

# numpy for the data and maths.
import numpy as np

# Brains.
from hunger_games.brain import Brain, create_brain

# The optimiser.
from hunger_games.brain.mlp import Adam

# The student network and the menu.
from hunger_games.brain.neural import NeuralBrain, softmax

# Settings.
from hunger_games.config import SimulationConfig

# The game.
from hunger_games.game import Game

# The perception vector size.
# Recordings for the training feed.
from hunger_games.recorder import Recording

# Custom setups.
from hunger_games.scenario import Scenario

# Greedy validation games reuse the RL episode player.
from hunger_games.training.reinforce import play_rl_episode


@dataclass
class ImitationConfig:
    """Every knob of the imitation learner."""

    # Which brain to copy.
    teacher: str = "voting"
    # The chaos dial while the teacher demonstrates: 0 makes it pick its favourite action every time,
    # which gives clean labels (a teacher that sometimes acts at random is hard to copy).
    teacher_chaos: float = 0.0
    # How many teacher games to record for demonstrations (12 games give about 40,000 decisions).
    demonstration_games: int = 12
    # Passes over the demonstration data.
    epochs: int = 30
    # Samples per gradient step.
    batch_size: int = 256
    # Adam step size.
    learning_rate: float = 1e-3
    # Fraction of the demonstrations held out to measure validation loss and accuracy.
    validation_fraction: float = 0.2
    # Greedy games the student plays per epoch on fixed seeds (survival, win rate, telemetry, showcase).
    validation_games: int = 1
    # The first validation seed.
    validation_seed: int = 90000
    # Tributes driven by the student in validation games (the rest use the config's brain).
    learners_per_game: int = 6
    # CPU cores for collecting demonstrations and validation games.
    workers: int = 1
    # The trainer's own seed (data shuffling, network init).
    seed: int | None = None
    # Whether to record the validation game for the dashboard's training feed.
    record_showcase: bool = True


@dataclass
class ImitationStats:
    """What happened in one epoch."""

    # Which epoch (0 first).
    epoch: int
    # Mean cross-entropy on the training demonstrations.
    train_loss: float
    # Mean cross-entropy on the held-out demonstrations.
    val_loss: float
    # Fraction of training demonstrations the student gets right.
    train_accuracy: float
    # Fraction of held-out demonstrations the student gets right.
    val_accuracy: float
    # Mean ticks the student survived in the validation games.
    val_survival: float
    # Fraction of validation games the student won.
    val_win_rate: float
    # Seconds this epoch took.
    seconds: float
    # Seconds since training started.
    cumulative_seconds: float
    # The student's genome after this epoch (a copy).
    genome: np.ndarray = field(repr=False, default=None)  # type: ignore[assignment]
    # Behaviour telemetry from the validation games.
    telemetry: dict = field(default_factory=dict, repr=False)
    # A recording of one validation game.
    showcase: Recording | None = field(default=None, repr=False)

    def to_row(self) -> dict:
        """A JSON-friendly dictionary without the big arrays."""
        # Everything but the arrays.
        return {k: v for k, v in self.__dict__.items() if k not in ("genome", "telemetry", "showcase")}


def collect_demonstration_game(
    config: SimulationConfig, scenario: Scenario | None, teacher: str, seed: int, teacher_chaos: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Play one game with the teacher brain and return every (perception vector, menu index) pair."""
    # A config copy with this game's seed and the teacher's chaos.
    game_config = SimulationConfig(**{**config.to_dict_raw(), "seed": seed, "chaos": teacher_chaos})

    # Every tribute is the teacher.
    def factory(index: int, rng: np.random.Generator) -> Brain:
        return create_brain(teacher, game_config.chaos, rng, game_config.neural, game_config.endgame_instinct)

    # The game.
    game = Game(game_config, 0, brain_factory=factory, scenario=scenario)
    # The samples.
    vectors: list[np.ndarray] = []
    labels: list[int] = []

    # Record each decision.
    def on_decision(player, perception, action) -> None:
        vectors.append(perception.to_vector())
        labels.append(NeuralBrain.action_to_menu_index(action))

    # Hook and play.
    game.decision_hooks.append(on_decision)
    game.run()
    # Arrays.
    return np.asarray(vectors, dtype=float), np.asarray(labels, dtype=int)


def _demo_job(args: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Unpack a job tuple for the process pool."""
    # Forward.
    return collect_demonstration_game(*args)


def _validation_job(args: tuple) -> dict:
    """Unpack a validation job tuple for the process pool."""
    # Forward.
    return play_rl_episode(*args)


class ImitationTrainer:
    """Trains a NeuralBrain to copy a teacher brain's decisions."""

    def __init__(
        self,
        config: SimulationConfig,
        imitation: ImitationConfig,
        scenario: Scenario | None = None,
        initial_genome: np.ndarray | None = None,
    ) -> None:
        """Create the student (fresh, or from a genome) and an empty history."""
        # Settings.
        self.config = config
        # Learner settings.
        self.imitation = imitation
        # Optional custom setup.
        self.scenario = scenario
        # The trainer's own randomness.
        self.rng = np.random.default_rng(imitation.seed)
        # The student network.
        self.policy = NeuralBrain(chaos=0.0, config=config.neural, rng=self.rng).network
        # A warm start.
        if initial_genome is not None:
            self.policy.set_genome(np.asarray(initial_genome, dtype=float))
        # The optimiser.
        self.optimizer = Adam(self.policy, imitation.learning_rate)
        # Demonstrations, filled by collect().
        self.train_x: np.ndarray | None = None
        self.train_y: np.ndarray | None = None
        self.val_x: np.ndarray | None = None
        self.val_y: np.ndarray | None = None
        # History.
        self.history: list[ImitationStats] = []
        # Epoch counter.
        self.epoch = 0
        # Stop flag.
        self._stop = False
        # Start time.
        self._started: float | None = None
        # The genome with the best validation loss so far.
        self.best_genome: np.ndarray | None = None
        # Its validation loss.
        self.best_val_loss = np.inf

    @property
    def settings(self):
        """The trainer's own settings dataclass."""
        # The imitation settings.
        return self.imitation

    def _learner_ids(self) -> list[int]:
        """Which tribute slots the student drives in validation games (spread across the roster)."""
        # How many.
        count = min(self.imitation.learners_per_game, self.config.num_players)
        # Spread.
        return [int(i * self.config.num_players / count) for i in range(count)]

    # ---------------------------------------------------------- data

    def collect(self, on_progress: Callable[[int, int], None] | None = None) -> int:
        """Play the teacher games and split the demonstrations into training and validation sets."""
        # Seeds.
        seeds = [int(self.rng.integers(2**31 - 1)) for _ in range(self.imitation.demonstration_games)]
        # Jobs.
        jobs = [
            (self.config, self.scenario, self.imitation.teacher, seed, self.imitation.teacher_chaos) for seed in seeds
        ]
        # Results.
        results = []
        # Parallel or sequential.
        if self.imitation.workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=self.imitation.workers) as pool:
                for index, result in enumerate(pool.map(_demo_job, jobs)):
                    results.append(result)
                    if on_progress is not None:
                        on_progress(index + 1, len(jobs))
        else:
            for index, job in enumerate(jobs):
                results.append(_demo_job(job))
                if on_progress is not None:
                    on_progress(index + 1, len(jobs))
        # Stack.
        x = np.concatenate([r[0] for r in results])
        y = np.concatenate([r[1] for r in results])
        # Shuffle.
        order = self.rng.permutation(len(x))
        x, y = x[order], y[order]
        # Split.
        split = int(len(x) * (1.0 - self.imitation.validation_fraction))
        self.train_x, self.train_y = x[:split], y[:split]
        self.val_x, self.val_y = x[split:], y[split:]
        # How many samples.
        return len(x)

    # ------------------------------------------------------- learning

    def _loss_and_accuracy(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Mean cross-entropy and accuracy of the student on a set."""
        # Empty.
        if len(x) == 0:
            return 0.0, 0.0
        # Probabilities.
        probabilities = softmax(self.policy.forward(x))
        # Cross-entropy of the right answers.
        loss = float(-np.log(probabilities[np.arange(len(y)), y] + 1e-12).mean())
        # Accuracy.
        accuracy = float((probabilities.argmax(axis=1) == y).mean())
        # Done.
        return loss, accuracy

    def step_epoch(self, on_progress: Callable[[int, int], None] | None = None) -> ImitationStats:
        """One pass over the demonstrations, then validation."""
        # Collect on the first epoch.
        if self.train_x is None:
            self.collect(on_progress)
        # Clocks.
        if self._started is None:
            self._started = time.time()
        started = time.time()
        # A fresh shuffle each epoch.
        order = self.rng.permutation(len(self.train_x))
        # Mini-batches.
        for start in range(0, len(order), self.imitation.batch_size):
            batch = order[start : start + self.imitation.batch_size]
            x, y = self.train_x[batch], self.train_y[batch]
            # Forward with cache.
            logits, cache = self.policy.forward_cached(x)
            probabilities = softmax(logits)
            # Gradient of the mean cross-entropy with respect to the logits: (p - onehot) / batch.
            grad = probabilities.copy()
            grad[np.arange(len(y)), y] -= 1.0
            grad /= len(y)
            # Backprop and step.
            self.optimizer.step(self.policy.backward(cache, grad))
        # Scores.
        train_loss, train_accuracy = self._loss_and_accuracy(self.train_x, self.train_y)
        val_loss, val_accuracy = self._loss_and_accuracy(self.val_x, self.val_y)
        # Track the best student by validation loss.
        if val_loss < self.best_val_loss or self.best_genome is None:
            self.best_val_loss = val_loss
            self.best_genome = self.policy.genome().copy()
        # Greedy validation games.
        val_survival, val_win_rate, telemetry, showcase = self._validate()
        # Stats.
        stats = ImitationStats(
            epoch=self.epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy,
            val_survival=val_survival,
            val_win_rate=val_win_rate,
            seconds=time.time() - started,
            cumulative_seconds=time.time() - self._started,
            genome=self.policy.genome().copy(),
            telemetry=telemetry,
            showcase=showcase,
        )
        # Record.
        self.history.append(stats)
        self.epoch += 1
        # Done.
        return stats

    def _validate(self) -> tuple[float, float, dict, Recording | None]:
        """Play the student greedily on the fixed validation seeds."""
        # None asked for.
        if self.imitation.validation_games <= 0:
            return 0.0, 0.0, {}, None
        # Jobs (the first records the showcase).
        genome = self.policy.genome()
        learners = self._learner_ids()
        jobs = [
            (
                self.config,
                self.scenario,
                genome,
                learners,
                self.imitation.validation_seed + i,
                True,
                i == 0 and self.imitation.record_showcase,
            )
            for i in range(self.imitation.validation_games)
        ]
        # Play.
        if self.imitation.workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=self.imitation.workers) as pool:
                episodes = list(pool.map(_validation_job, jobs))
        else:
            episodes = [_validation_job(job) for job in jobs]
        # Outcomes.
        outcomes = [o for episode in episodes for o in episode["outcomes"].values()]
        survival = float(np.mean([o["survival"] for o in outcomes])) if outcomes else 0.0
        win_rate = float(np.mean([o["won"] for o in outcomes])) if outcomes else 0.0
        # Telemetry merged.
        from hunger_games.research.telemetry import BehaviorTelemetry

        telemetry = BehaviorTelemetry.merge([episode["telemetry"] for episode in episodes])
        # Done.
        return survival, win_rate, telemetry, episodes[0].get("recording")

    def run(
        self,
        on_epoch: Callable[[ImitationStats], None] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[ImitationStats]:
        """Run every epoch (or until `stop()`)."""
        # Reset.
        self._stop = False
        # Loop.
        while self.epoch < self.imitation.epochs and not self._stop:
            stats = self.step_epoch(on_progress)
            if on_epoch is not None:
                on_epoch(stats)
        # History.
        return self.history

    def stop(self) -> None:
        """Ask a running `run()` to finish after the current epoch."""
        # Flag.
        self._stop = True

    # ------------------------------------------------------------ outputs

    @property
    def champion(self) -> np.ndarray | None:
        """The student with the best validation loss so far (the current one if none validated)."""
        # Best, or current.
        return self.best_genome if self.best_genome is not None else self.policy.genome()

    def champion_brain(self, chaos: float = 0.0) -> NeuralBrain:
        """A NeuralBrain loaded with the champion."""
        # Build.
        brain = NeuralBrain(chaos=chaos, config=self.config.neural, rng=self.rng)
        brain.set_genome(self.champion)
        return brain

    def save_policy(self, path: str | Path) -> None:
        """Write the champion in the same JSON shape as the other trainers' champion files."""
        # Data.
        data = {
            "brain_name": "neural",
            "neural": asdict(self.config.neural),
            "genome": self.champion.tolist(),
            "fitness": float(-self.best_val_loss) if np.isfinite(self.best_val_loss) else 0.0,
            "epochs": len(self.history),
            "method": "imitation",
            "teacher": self.imitation.teacher,
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
