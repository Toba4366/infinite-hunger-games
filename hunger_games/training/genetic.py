"""training/genetic.py - a genetic algorithm that evolves brains by playing games.

The idea: keep a population of genomes (weight vectors). Each generation,
put them into games as the tributes, score each one by how it placed
(plus a little for kills and days survived), keep the best, and breed the
rest by mixing two parents and adding small random mutations. Repeat.
Nothing here knows what a genome means: it works for the voting brain's
eight genes and for a neural network's thousands of weights alike.
"""

# Parallel evaluation across CPU cores.
# JSON for saving champions.
import json

# Timing each generation.
import time

# Type hints.
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor

# Dataclasses for settings and per-generation stats.
from dataclasses import asdict, dataclass, field

# Filesystem paths.
from pathlib import Path

# numpy for genomes.
import numpy as np

# Brain construction.
from hunger_games.brain import Brain, create_brain

# The settings.
from hunger_games.config import NeuralConfig, SimulationConfig

# The game.
from hunger_games.game import Game

# Recordings for the training feed.
from hunger_games.recorder import Recorder, Recording

# Behaviour measurement.
from hunger_games.research.telemetry import BehaviorTelemetry

# Custom setups.
from hunger_games.scenario import Scenario


@dataclass
class TrainingConfig:
    """Every knob of the genetic algorithm."""

    # Which brain kind is being evolved ("neural" or "voting").
    brain_name: str = "neural"
    # How many genomes are alive at once (a multiple of the player count avoids padding).
    population_size: int = 48
    # How many generations to run.
    generations: int = 20
    # How many games each genome plays per generation (more = steadier scores, slower).
    rounds_per_generation: int = 2
    # The fraction of the population copied unchanged into the next generation.
    elite_fraction: float = 0.1
    # How many random genomes compete to become a parent (bigger = stronger selection pressure).
    tournament_size: int = 3
    # The chance that a child mixes two parents rather than cloning one.
    crossover_rate: float = 0.5
    # The fraction of a child's genes that get a random nudge.
    mutation_rate: float = 0.1
    # The size of that nudge (standard deviation of the Gaussian added).
    mutation_scale: float = 0.1
    # CPU cores to evaluate games on.
    workers: int = 1
    # Seed for the trainer's own randomness (population init, breeding, game seeds).
    seed: int | None = None
    # Fitness bonus per kill.
    kills_weight: float = 0.05
    # Fitness bonus per day survived.
    days_weight: float = 0.01
    # Games played per generation by the champion against the config's brain on fixed seeds (validation).
    validation_games: int = 2
    # The first validation seed (validation game i uses this + i).
    validation_seed: int = 90000
    # Whether to measure behaviour (slower, but gives the behaviour charts).
    collect_telemetry: bool = True
    # Whether to record one real evaluation game per generation so the dashboard can replay training.
    record_showcase: bool = True


@dataclass
class GenerationStats:
    """What happened in one generation."""

    # Which generation (0 is the first).
    generation: int
    # The best fitness in the population.
    best_fitness: float
    # The average fitness.
    mean_fitness: float
    # The worst fitness.
    worst_fitness: float
    # The best genome of this generation (a copy).
    champion: np.ndarray
    # Wall-clock seconds the generation took.
    seconds: float
    # Mean fitness of this generation's champion in the validation games (against the config's brain).
    val_fitness: float = 0.0
    # Total seconds since training started.
    cumulative_seconds: float = 0.0
    # Behaviour telemetry summary from this generation's evaluation games.
    telemetry: dict = field(default_factory=dict, repr=False)
    # A recording of one real evaluation game from this generation (the dashboard's training feed).
    showcase: Recording | None = field(default=None, repr=False)

    def to_row(self) -> dict:
        """A JSON-friendly dictionary without the big arrays."""
        # Everything but the champion, telemetry and showcase recording.
        return {k: v for k, v in self.__dict__.items() if k not in ("champion", "telemetry", "showcase")}


def fitness_of(
    placement: int, kills: int, days: float, num_players: int, kills_weight: float, days_weight: float
) -> float:
    """Score one game: 1.0 for winning, 0.0 for first out, plus small bonuses."""
    # Placement 1 scores 1.0, placement num_players scores 0.0.
    placing = (num_players - placement) / max(1, num_players - 1)
    # Add the bonuses.
    return placing + kills_weight * kills + days_weight * days


def play_evaluation_game(
    config: SimulationConfig,
    scenario: Scenario | None,
    brain_name: str,
    genomes: list[np.ndarray],
    seed: int,
    collect_telemetry: bool = False,
    record: bool = False,
) -> tuple[list[tuple[int, int, float]], dict | None, Recording | None]:
    """Play one game where tribute i is driven by genomes[i].

    Returns (placement, kills, days_survived) per tribute, an optional
    behaviour telemetry summary, and an optional tick-by-tick recording of
    the game. A top-level function so worker processes can run it.
    """
    # A copy of the config with the right player count and seed.
    game_config = SimulationConfig(**{**config.to_dict_raw(), "num_players": len(genomes), "seed": seed})

    # Build each tribute's brain from its genome.
    def factory(index: int, rng: np.random.Generator) -> Brain:
        # A fresh brain of the right kind and architecture.
        brain = create_brain(brain_name, game_config.chaos, rng, game_config.neural, game_config.endgame_instinct)
        # Load the genome.
        brain.set_genome(genomes[index])
        # Done.
        return brain

    # The game.
    game = Game(game_config, 0, brain_factory=factory, scenario=scenario)
    # Optional behaviour measurement.
    telemetry = BehaviorTelemetry(game.arena.width, game.arena.height).attach(game) if collect_telemetry else None
    # Play, recording every tick if asked.
    if record:
        recording = Recorder(game).record_all()
        result = recording.result
    else:
        recording = None
        result = game.run()
    # Pull out the numbers, in tribute order.
    by_id = {row.player_id: row for row in result.players}
    # One tuple per genome, plus the telemetry and the recording.
    return (
        [(by_id[i].placement, by_id[i].kills, by_id[i].days_survived) for i in range(len(genomes))],
        telemetry.summary() if telemetry else None,
        recording,
    )


def play_validation_game(
    config: SimulationConfig,
    scenario: Scenario | None,
    brain_name: str,
    genome: np.ndarray,
    learner_ids: list[int],
    seed: int,
) -> list[tuple[int, int, float]]:
    """Play one game where `learner_ids` carry the champion genome and everyone
    else uses the config's brain. Returns (placement, kills, days) for the
    learners only. Used for validation on fixed seeds.
    """
    # A copy of the config with the seed.
    game_config = SimulationConfig(**{**config.to_dict_raw(), "seed": seed})

    # Learners get the champion, the rest the config's default brain.
    def factory(index: int, rng: np.random.Generator) -> Brain:
        # A learner.
        if index in learner_ids:
            brain = create_brain(brain_name, game_config.chaos, rng, game_config.neural, game_config.endgame_instinct)
            brain.set_genome(genome)
            return brain
        # An opponent.
        return create_brain(
            game_config.brain_name, game_config.chaos, rng, game_config.neural, game_config.endgame_instinct
        )

    # Play.
    result = Game(game_config, 0, brain_factory=factory, scenario=scenario).run()
    # Learners' rows.
    by_id = {row.player_id: row for row in result.players}
    # Done.
    return [(by_id[i].placement, by_id[i].kills, by_id[i].days_survived) for i in learner_ids]


def _run_job(args: tuple) -> tuple[list[tuple[int, int, float]], dict | None, Recording | None]:
    """Unpack a job tuple for the process pool."""
    # Forward to the real function.
    return play_evaluation_game(*args)


def _run_validation_job(args: tuple) -> list[tuple[int, int, float]]:
    """Unpack a validation job tuple for the process pool."""
    # Forward.
    return play_validation_game(*args)


class GeneticTrainer:
    """Evolves a population of genomes by playing them against each other."""

    def __init__(
        self,
        config: SimulationConfig,
        training: TrainingConfig,
        scenario: Scenario | None = None,
        initial_genome: np.ndarray | None = None,
    ) -> None:
        """Create a random starting population, or one seeded from a genome (a warm start)."""
        # The game settings.
        self.config = config
        # The algorithm settings.
        self.training = training
        # The custom setup games are played on, if any.
        self.scenario = scenario
        # The trainer's own random generator.
        self.rng = np.random.default_rng(training.seed)
        # A template brain tells us how long a genome is.
        template = create_brain(training.brain_name, config.chaos, self.rng, config.neural)
        # Genome length.
        self.genome_size = template.genome().size
        # A brain with nothing to tune cannot be trained.
        if self.genome_size == 0:
            raise ValueError(f"The '{training.brain_name}' brain has no genome to train")
        # Fresh random genomes, each from a freshly initialised brain...
        if initial_genome is None:
            self.population = [
                create_brain(training.brain_name, config.chaos, self.rng, config.neural).genome()
                for _ in range(training.population_size)
            ]
        # ...or a warm start: the given genome plus close relatives of it. The spread is a quarter of the
        # mutation scale because a trained network's weights are small and full-size noise erases its instincts.
        else:
            seed_genome = np.asarray(initial_genome, dtype=float)
            self.population = [seed_genome.copy()] + [
                seed_genome + self.rng.normal(0.0, 0.25 * training.mutation_scale, seed_genome.size)
                for _ in range(training.population_size - 1)
            ]
        # Fitness of the current population (filled by evaluate).
        self.fitness = np.zeros(training.population_size)
        # Stats per generation.
        self.history: list[GenerationStats] = []
        # Which generation we are on.
        self.generation = 0
        # Set by stop() to end run() early.
        self._stop = False
        # When training started (for cumulative time).
        self._started: float | None = None
        # Telemetry summaries collected by the last evaluate().
        self._last_telemetry: list[dict] = []
        # The recording made by the last evaluate(), if any.
        self._last_showcase: Recording | None = None

    @property
    def settings(self):
        """The trainer's own settings dataclass (every trainer exposes this name for run folders)."""
        # The GA settings.
        return self.training

    # ------------------------------------------------------------ evaluate

    def _make_jobs(self) -> list[tuple[list[int], int]]:
        """Split the population into games: lists of genome indices plus a seed each."""
        # Players per game.
        per_game = self.config.num_players
        # Jobs to run.
        jobs = []
        # Each round shuffles the population into fresh games.
        for _ in range(self.training.rounds_per_generation):
            # A random order.
            order = list(self.rng.permutation(len(self.population)))
            # Pad with random extra entrants so the last game is full.
            while len(order) % per_game:
                order.append(int(self.rng.integers(len(self.population))))
            # Cut into games.
            for start in range(0, len(order), per_game):
                # This game's genome indices and a seed.
                jobs.append((order[start : start + per_game], int(self.rng.integers(2**31 - 1))))
        # All jobs.
        return jobs

    def evaluate(self, on_progress: Callable[[int, int], None] | None = None) -> np.ndarray:
        """Play every job and return the mean fitness of each genome."""
        # The jobs.
        jobs = self._make_jobs()
        # Arguments for each job.
        arguments = [
            (
                self.config,
                self.scenario,
                self.training.brain_name,
                [self.population[i] for i in indices],
                seed,
                self.training.collect_telemetry,
                job_index == 0 and self.training.record_showcase,
            )
            for job_index, (indices, seed) in enumerate(jobs)
        ]
        # Telemetry from each job.
        self._last_telemetry = []
        # No recording yet this generation.
        self._last_showcase = None
        # Fitness totals and counts.
        totals = np.zeros(len(self.population))
        # Counts.
        counts = np.zeros(len(self.population))

        # Fold one job's results into the totals.
        def absorb(
            job_index: int, job_result: tuple[list[tuple[int, int, float]], dict | None, Recording | None]
        ) -> None:
            # Unpack.
            outcome, telemetry, recording = job_result
            # Keep the telemetry.
            if telemetry is not None:
                self._last_telemetry.append(telemetry)
            # Keep the recording (only the first job records).
            if recording is not None:
                self._last_showcase = recording
            # The genome indices that played.
            indices = jobs[job_index][0]
            # Score each.
            for genome_index, (placement, kills, days) in zip(indices, outcome, strict=False):
                # Add the score.
                totals[genome_index] += fitness_of(
                    placement,
                    kills,
                    days,
                    self.config.num_players,
                    self.training.kills_weight,
                    self.training.days_weight,
                )
                # Count the game.
                counts[genome_index] += 1
            # Report progress.
            if on_progress is not None:
                on_progress(job_index + 1, len(jobs))

        # Multi-core path.
        if self.training.workers > 1:
            # A pool of workers.
            with ProcessPoolExecutor(max_workers=self.training.workers) as pool:
                # Results come back in job order.
                for job_index, outcome in enumerate(pool.map(_run_job, arguments)):
                    absorb(job_index, outcome)
        # Single-core path.
        else:
            # One at a time.
            for job_index, args in enumerate(arguments):
                absorb(job_index, _run_job(args))
        # Mean fitness per genome.
        self.fitness = totals / np.maximum(counts, 1)
        # Done.
        return self.fitness

    # -------------------------------------------------------------- breed

    def _tournament(self) -> np.ndarray:
        """Pick a parent: the fittest of a few random genomes."""
        # Random contestants.
        contestants = self.rng.integers(len(self.population), size=self.training.tournament_size)
        # The fittest contestant.
        winner = contestants[np.argmax(self.fitness[contestants])]
        # Its genome.
        return self.population[winner]

    def _child(self) -> np.ndarray:
        """Breed one child: maybe crossover two parents, then mutate."""
        # First parent.
        parent = self._tournament()
        # Sometimes mix in a second parent gene by gene.
        if self.rng.random() < self.training.crossover_rate:
            # Second parent.
            other = self._tournament()
            # A random mask picks which genes come from which parent.
            mask = self.rng.random(self.genome_size) < 0.5
            # Combine.
            child = np.where(mask, parent, other)
        # Otherwise clone.
        else:
            child = parent.copy()
        # Which genes to nudge.
        mutate = self.rng.random(self.genome_size) < self.training.mutation_rate
        # Nudge them.
        child = child + mutate * self.rng.normal(0.0, self.training.mutation_scale, self.genome_size)
        # Done.
        return child

    def _learner_ids(self) -> list[int]:
        """Which tribute slots the champion takes in validation games (a quarter of the roster, spread out)."""
        # How many.
        count = max(1, self.config.num_players // 4)
        # Spread.
        return [int(i * self.config.num_players / count) for i in range(count)]

    def validate(self, genome: np.ndarray) -> float:
        """Mean fitness of a genome against the config's brain on the fixed validation seeds."""
        # None asked for.
        if self.training.validation_games <= 0:
            return 0.0
        # Jobs.
        learners = self._learner_ids()
        arguments = [
            (self.config, self.scenario, self.training.brain_name, genome, learners, self.training.validation_seed + i)
            for i in range(self.training.validation_games)
        ]
        # Play.
        if self.training.workers > 1 and len(arguments) > 1:
            with ProcessPoolExecutor(max_workers=self.training.workers) as pool:
                outcomes = list(pool.map(_run_validation_job, arguments))
        else:
            outcomes = [_run_validation_job(args) for args in arguments]
        # Score.
        scores = [
            fitness_of(p, k, d, self.config.num_players, self.training.kills_weight, self.training.days_weight)
            for outcome in outcomes
            for p, k, d in outcome
        ]
        # Mean.
        return float(np.mean(scores)) if scores else 0.0

    def step_generation(self, on_progress: Callable[[int, int], None] | None = None) -> GenerationStats:
        """Evaluate, validate, record, and breed the next population."""
        # Start the clocks.
        if self._started is None:
            self._started = time.time()
        started = time.time()
        # Score everyone.
        fitness = self.evaluate(on_progress)
        # Best first.
        ranking = np.argsort(fitness)[::-1]
        # The champion of this generation.
        champion = self.population[ranking[0]].copy()
        # Validate it on the held-out seeds.
        val_fitness = self.validate(champion)
        # Merge this generation's behaviour telemetry.
        telemetry = BehaviorTelemetry.merge(self._last_telemetry) if self._last_telemetry else {}
        # Record the stats.
        stats = GenerationStats(
            self.generation,
            float(fitness[ranking[0]]),
            float(fitness.mean()),
            float(fitness[ranking[-1]]),
            champion,
            time.time() - started,
            val_fitness,
            time.time() - self._started,
            telemetry,
            self._last_showcase,
        )
        # Keep them.
        self.history.append(stats)
        # How many elites survive unchanged.
        elite_count = max(1, int(self.training.elite_fraction * len(self.population)))
        # The elites.
        next_population = [self.population[i].copy() for i in ranking[:elite_count]]
        # Fill the rest with children.
        while len(next_population) < len(self.population):
            next_population.append(self._child())
        # Replace the population.
        self.population = next_population
        # Count the generation.
        self.generation += 1
        # Hand back the stats.
        return stats

    def run(
        self,
        on_generation: Callable[[GenerationStats], None] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[GenerationStats]:
        """Run every generation (or until `stop()` is called)."""
        # Reset the stop flag.
        self._stop = False
        # One generation at a time.
        while self.generation < self.training.generations and not self._stop:
            # Evolve.
            stats = self.step_generation(on_progress)
            # Report.
            if on_generation is not None:
                on_generation(stats)
        # The full history.
        return self.history

    def stop(self) -> None:
        """Ask a running `run()` to finish after the current generation."""
        # Set the flag; run() checks it between generations.
        self._stop = True

    def history_rows(self) -> list[dict]:
        """The history as JSON-friendly rows (no genomes or telemetry)."""
        # Rows.
        return [stats.to_row() for stats in self.history]

    def previous_champion(self) -> np.ndarray | None:
        """The champion of the generation before the latest, for highlighting which genes changed."""
        # Need two.
        return self.history[-2].champion if len(self.history) >= 2 else None

    # ------------------------------------------------------------ champion

    @property
    def champion(self) -> np.ndarray | None:
        """The best genome seen in any generation so far."""
        # Nothing evaluated yet.
        if not self.history:
            return None
        # The generation with the best fitness.
        best = max(self.history, key=lambda stats: stats.best_fitness)
        # Its champion.
        return best.champion

    def champion_brain(self, chaos: float | None = None) -> Brain:
        """A brain loaded with the champion genome."""
        # Chaos from the config unless overridden.
        chaos = self.config.chaos if chaos is None else chaos
        # A fresh brain of the trained kind.
        brain = create_brain(self.training.brain_name, chaos, self.rng, self.config.neural)
        # Load the champion.
        if self.champion is not None:
            brain.set_genome(self.champion)
        # Done.
        return brain

    def save_champion(self, path: str | Path) -> None:
        """Write the champion genome and its architecture to a JSON file."""
        # Nothing to save yet.
        if self.champion is None:
            raise ValueError("No generations have been run yet")
        # The data.
        data = {
            "brain_name": self.training.brain_name,
            "neural": asdict(self.config.neural),
            "genome": self.champion.tolist(),
            "fitness": max(stats.best_fitness for stats in self.history),
            "generations": len(self.history),
        }
        # Write.
        Path(path).write_text(json.dumps(data))

    @staticmethod
    def load_champion(path: str | Path) -> dict:
        """Read a champion file back: a dict with brain_name, neural, genome (as numpy), fitness."""
        # Parse.
        data = json.loads(Path(path).read_text())
        # Restore the genome as an array.
        data["genome"] = np.asarray(data["genome"], dtype=float)
        # Restore the neural config.
        data["neural"] = NeuralConfig(**{**data["neural"], "hidden_layers": tuple(data["neural"]["hidden_layers"])})
        # Done.
        return data
