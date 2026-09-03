"""training/common.py - what every training method has in common.

Every method here trains one learner network against opponents that use
the voting brain from the video. They differ in how the network changes
between iterations (copying a teacher, evolving, or following rewards),
but not in how an iteration is scored or reported. This module holds the
shared pieces: the per-iteration statistics every trainer fills in, the
event log that feeds the dashboard's event monitor, the curriculum that
grows the number of opponents as the learner improves, a system monitor
for CPU and memory, and the description of a learner brain that worker
processes can rebuild.
"""

# Settings and stats.
# Timestamps for events.
import time
from dataclasses import dataclass, field

# Type hints.
from typing import Any

# numpy for genomes.
import numpy as np

# Brains.
from hunger_games.brain.base import Brain
from hunger_games.brain.neural import NeuralBrain

# Settings.
from hunger_games.config import NeuralConfig, SimulationConfig

# Recordings for the training feed.
from hunger_games.recorder import Recording


@dataclass
class IterationStats:
    """One iteration of any method, in the same shape for every method."""

    # Which iteration (0 first).
    iteration: int
    # The score of every learner episode this iteration (a return under the reward function).
    scores: list[float]
    # Their mean.
    mean_score: float
    # The best of them.
    best_score: float
    # Policy entropy (nats): high = exploring, low = confident.
    entropy: float
    # Mean ticks the learner survived this iteration.
    mean_length: float
    # Fraction of training games a learner copy won (game-level: one victor per game).
    win_rate: float
    # Mean score in the greedy validation games on fixed seeds.
    val_score: float
    # Seconds this iteration took.
    seconds: float
    # Seconds since training started.
    cumulative_seconds: float
    # Fraction of validation games a learner copy won (game-level: one victor per game).
    val_win_rate: float = 0.0
    # Curriculum stage index and the number of opponents in it.
    stage: int = 0
    # Opponents faced.
    opponents: int = 23
    # Method-specific numbers (losses, species counts, accuracy, ...).
    extra: dict = field(default_factory=dict)
    # The learner after this iteration: a genome array for neural brains, a NEAT genome dict for NEAT.
    learner: Any = field(default=None, repr=False)
    # Behaviour telemetry summary of the learner's episodes.
    telemetry: dict = field(default_factory=dict, repr=False)
    # A recording of one real episode from this iteration (the dashboard's training feed).
    showcase: Recording | None = field(default=None, repr=False)

    def to_row(self) -> dict:
        """A JSON-friendly dictionary without the big arrays."""
        # The plain fields.
        row = {
            k: v for k, v in self.__dict__.items() if k not in ("learner", "telemetry", "showcase", "scores", "extra")
        }
        # Flatten the extras in.
        row.update({f"extra_{k}": v for k, v in self.extra.items()})
        # Done.
        return row


def champion_key(stage: int, val_win_rate: float, val_score: float) -> tuple[int, float, float]:
    """The order champions are compared in: highest curriculum stage first, then validation wins, then score.

    Validation scores from different stages are not comparable (one opponent is easier than seven), so the stage
    comes first; a policy from an easier rung can never displace one that played at a harder rung.
    """
    # Lexicographic tuple: Python compares element by element.
    return (int(stage), float(val_win_rate), float(val_score))


class EventLog:
    """Timestamped one-line messages about what training is doing (the dashboard's event monitor)."""

    def __init__(self, capacity: int = 500) -> None:
        """Keep at most `capacity` recent events."""
        # The events, newest last.
        self.events: list[str] = []
        # How many to keep.
        self.capacity = capacity
        # When the log started, so timestamps are relative.
        self.started = time.time()

    def add(self, kind: str, message: str) -> None:
        """Record an event of a kind ("rollout", "evolution", "curriculum", "record", "info")."""
        # Seconds since the start.
        stamp = time.time() - self.started
        # Append.
        self.events.append(f"[{stamp:7.1f}s] {kind:<10} {message}")
        # Trim.
        if len(self.events) > self.capacity:
            self.events = self.events[-self.capacity :]

    def tail(self, count: int = 20) -> list[str]:
        """The most recent events."""
        # The last ones.
        return self.events[-count:]


@dataclass
class Stage:
    """One lesson of a curriculum: who the learner faces, which rules apply, and what earns promotion."""

    # A label for events and charts ("survive", "beat 3", "generalise").
    name: str
    # Voting opponents in every game of this stage (the learner's own copies are added on top).
    opponents: int
    # Dotted `SimulationConfig` overrides applied to every game of the stage, e.g. {"gamemaker_enabled": False}.
    overrides: dict = field(default_factory=dict)
    # Override sets one of which is picked at random per episode (the generalisation lesson); empty means none.
    variants: tuple[dict, ...] = ()
    # What promotion is judged on: "win_rate" (games won), "survival" (share of the game survived) or "score".
    metric: str = "win_rate"
    # Promote when the last `window` iterations of that metric average at least this.
    threshold: float = 0.5


def apply_overrides(config: SimulationConfig, overrides: dict) -> SimulationConfig:
    """A copy of `config` with dotted overrides applied ("neural.hidden_layers", "layout", "gamemaker_enabled")."""
    # Nothing to change.
    if not overrides:
        return config
    # Work on the dictionary form, so enums (shape, layout) can be given as their string values.
    data = config.to_dict()
    for dotted, value in overrides.items():
        target = data
        parts = dotted.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    # Back to a config (from_dict restores the enums and nested dataclasses).
    return SimulationConfig.from_dict(data)


def stage_config(base: SimulationConfig, stage: Stage, learners: int) -> SimulationConfig:
    """The config every game of `stage` is played on: the learner copies plus the stage's opponents and rules."""
    # Roster size.
    players = min(learners, 24) + stage.opponents
    # Rules.
    return apply_overrides(SimulationConfig(**{**base.to_dict_raw(), "num_players": players}), stage.overrides)


def episode_config(config: SimulationConfig, stage: Stage | None, seed: int) -> SimulationConfig:
    """The config for one episode: the stage config, with one of the stage's variants chosen by the seed."""
    # No variants: the stage config as it is.
    if stage is None or not stage.variants:
        return config
    # Seeded choice, so the same seed always gives the same arena rules.
    pick = int(np.random.default_rng(seed).integers(len(stage.variants)))
    return apply_overrides(config, stage.variants[pick])


@dataclass
class CurriculumConfig:
    """Grow the difficulty as the learner improves, like the zombie video's one-to-sixteen ladder."""

    # Whether the curriculum is on.
    enabled: bool = True
    # Opponents per stage (the learner's own copies are added on top).
    opponents: tuple[int, ...] = (1, 3, 7, 11, 23)
    # Explicit lessons; when given they replace the `opponents` ladder (see `lessons()`).
    stages: tuple[Stage, ...] | None = None
    # What promotion is judged on: "win_rate" (the learner must actually win games) or "score".
    promote_on: str = "win_rate"
    # Promote when the last `window` iterations average at least this win rate (a majority of games by default).
    win_threshold: float = 0.5
    # Or, when judging on score, at least this mean score.
    threshold: float = 3.0
    # Iterations averaged for the promotion test.
    window: int = 5
    # Promote anyway after this many iterations in a stage; 0 means never (the learner must earn it).
    max_iterations_per_stage: int = 0

    @classmethod
    def lessons(
        cls, win_threshold: float = 0.5, survival_threshold: float = 0.9, window: int = 5
    ) -> "CurriculumConfig":
        """The lesson curriculum: survive first, then the arena rules, then win against a growing field, then generalise.

        Every cold start of the first full experiment died of thirst long before it met an opponent, so the first
        two lessons have no voting opponents at all and are passed on survival, not wins. The last lesson keeps the
        full field and varies the spawn layout, the arena shape and the rules from game to game.
        """
        # The rules every "win" lesson is played under.
        rules = {"gamemaker_enabled": True, "sponsors_enabled": True}
        stages = (
            # Learn to drink and eat: no opponents, no circle, no gifts; pass by surviving most of the game.
            Stage(
                "survive",
                0,
                {"gamemaker_enabled": False, "sponsors_enabled": False},
                metric="survival",
                threshold=survival_threshold,
            ),
            # Learn the arena: the circle shrinks and parachutes land.
            Stage("survive the rules", 0, dict(rules), metric="survival", threshold=survival_threshold),
            # Learn to win against more and more voting tributes.
            Stage("beat 1", 1, dict(rules), threshold=win_threshold),
            Stage("beat 3", 3, dict(rules), threshold=win_threshold),
            Stage("beat 7", 7, dict(rules), threshold=win_threshold),
            Stage("beat 11", 11, dict(rules), threshold=win_threshold),
            Stage("beat 23", 23, dict(rules), threshold=win_threshold),
            # Generalise: the full field, with the layout, the shape and the rules changing from game to game.
            Stage(
                "generalise",
                23,
                dict(rules),
                variants=(
                    {"layout": "cornucopia"},
                    {"layout": "ring"},
                    {"shape": "round"},
                    {"gamemaker_enabled": False},
                    {"sponsors_enabled": False},
                ),
                threshold=win_threshold,
            ),
        )
        return cls(stages=stages, win_threshold=win_threshold, window=window)


class Curriculum:
    """Tracks the current stage and decides when to promote."""

    def __init__(self, config: CurriculumConfig) -> None:
        """Start at the first stage."""
        # Settings.
        self.config = config
        # Current stage index.
        self.stage = 0
        # Iterations spent in the current stage.
        self.iterations_in_stage = 0
        # Recent mean scores in this stage.
        self.recent: list[float] = []

    @property
    def stages(self) -> tuple[Stage, ...]:
        """The lessons: the explicit ones, or one "beat N" stage per entry of the opponents ladder."""
        # Explicit lessons.
        if self.config.stages:
            return self.config.stages
        # The classic ladder, judged on wins or on score as configured.
        metric = "win_rate" if self.config.promote_on == "win_rate" else "score"
        threshold = self.config.win_threshold if metric == "win_rate" else self.config.threshold
        return tuple(Stage(f"beat {n}", n, metric=metric, threshold=threshold) for n in self.config.opponents)

    @property
    def stage_spec(self) -> Stage:
        """The current lesson (or the last one when the curriculum is off or finished)."""
        # Off: the hardest stage.
        if not self.config.enabled:
            return self.stages[-1]
        # Current, clamped.
        return self.stages[min(self.stage, len(self.stages) - 1)]

    @property
    def opponents(self) -> int:
        """Opponents in the current stage (or the last stage when the curriculum is off)."""
        # From the lesson.
        return self.stage_spec.opponents

    def describe(self) -> str:
        """A short label of the current lesson for events: "beat 7 (7 opponents)"."""
        # Name and count.
        return f"{self.stage_spec.name} ({self.stage_spec.opponents} opponents)"

    @property
    def finished(self) -> bool:
        """Is the learner in the final stage?"""
        # Last stage reached.
        return not self.config.enabled or self.stage >= len(self.stages) - 1

    def observe(self, mean_score: float, win_rate: float = 0.0, survival: float = 0.0) -> bool:
        """Record an iteration's score, win rate and survival share; returns True when the learner is promoted."""
        # Off or done.
        if self.finished:
            return False
        # Count.
        self.iterations_in_stage += 1
        # Remember the metric this lesson is judged on.
        stage = self.stage_spec
        judged = {"win_rate": win_rate, "survival": survival, "score": mean_score}[stage.metric]
        self.recent.append(judged)
        self.recent = self.recent[-self.config.window :]
        # Ready?
        good_enough = len(self.recent) >= self.config.window and float(np.mean(self.recent)) >= stage.threshold
        # Or stuck long enough (never, when the limit is 0).
        timed_out = 0 < self.config.max_iterations_per_stage <= self.iterations_in_stage
        # Promote.
        if good_enough or timed_out:
            self.stage += 1
            self.iterations_in_stage = 0
            self.recent = []
            return True
        # Stay.
        return False


class SystemMonitor:
    """CPU, memory and GPU readings for the dashboard (psutil when available)."""

    def __init__(self) -> None:
        """Try to import psutil once."""
        # Optional dependency.
        try:
            import psutil  # noqa: PLC0415 - optional

            self.psutil = psutil
            # Prime the CPU counter (the first call always returns 0).
            psutil.cpu_percent(interval=None)
        except ImportError:
            self.psutil = None

    def read(self) -> dict:
        """The current readings."""
        # Without psutil, report what we can.
        if self.psutil is None:
            return {"cpu_percent": 0.0, "memory_mb": 0.0, "memory_percent": 0.0, "gpu": "not used (numpy on the CPU)"}
        # Process memory.
        process = self.psutil.Process()
        # Readings.
        return {
            "cpu_percent": float(self.psutil.cpu_percent(interval=None)),
            "memory_mb": process.memory_info().rss / (1024 * 1024),
            "memory_percent": float(self.psutil.virtual_memory().percent),
            "gpu": "not used (numpy on the CPU)",
        }


@dataclass
class LearnerSpec:
    """How to rebuild a learner brain in a worker process: its kind and its genome."""

    # "neural" (a NeuralBrain genome array) or "neat" (a NEAT genome dictionary).
    kind: str
    # The genome: a flat array for neural, a dict for NEAT.
    genome: Any
    # The neural architecture (neural only).
    neural: NeuralConfig | None = None


def build_learner(spec: LearnerSpec, chaos: float, rng: np.random.Generator) -> Brain:
    """Build the learner brain a spec describes."""
    # A NEAT genome.
    if spec.kind == "neat":
        from hunger_games.brain.neat import NeatBrain, NeatGenome  # noqa: PLC0415 - avoid an import cycle

        return NeatBrain(NeatGenome.from_dict(spec.genome), chaos=chaos)
    # A voting brain's eight genes.
    if spec.kind == "voting":
        from hunger_games.brain.voting import VotingBrain  # noqa: PLC0415 - avoid an import cycle

        return VotingBrain(chaos=chaos, genome=np.asarray(spec.genome, dtype=float))
    # A neural network.
    brain = NeuralBrain(chaos=chaos, config=spec.neural, rng=rng)
    brain.set_genome(np.asarray(spec.genome, dtype=float))
    return brain


def learner_ids(num_players: int, learners: int) -> list[int]:
    """Evenly spread learner slots across the roster (so learners are not all neighbours on the podiums)."""
    # How many fit.
    count = max(1, min(learners, num_players))
    # Spread.
    return [int(i * num_players / count) for i in range(count)]
