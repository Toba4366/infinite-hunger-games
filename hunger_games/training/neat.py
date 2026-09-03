"""training/neat.py - NEAT: evolving the shape of the network as well as its weights.

The population starts as minimal genomes (inputs wired straight to the
outputs). Each generation every genome plays as the learner against voting
opponents and is scored by its episode return. Genomes are grouped into
species by structural similarity, fitness is shared within a species so a
big species cannot crowd everyone out, each species is given offspring in
proportion to its share of the total adjusted fitness, and offspring come
from crossover and mutation (weight nudges, new connections, new nodes).
Species that stop improving for too long are removed. This follows the
original NEAT paper and the Monopoly video's use of it.
"""

# Parallel evaluation.
# JSON for saving.
import json

# Timing.
import time

# Type hints.
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor

# Settings and stats.
from dataclasses import dataclass, field

# Paths.
from pathlib import Path

# numpy.
import numpy as np

# The genome and brain.
from hunger_games.brain.neat import InnovationTracker, NeatConfig, NeatGenome

# The menu size and perception size.
from hunger_games.brain.neural import MENU_SIZE

# Settings.
from hunger_games.config import SimulationConfig
from hunger_games.perception import VECTOR_SIZE

# Recordings.
from hunger_games.recorder import Recording

# Telemetry.
from hunger_games.research.telemetry import BehaviorTelemetry

# Custom setups.
from hunger_games.scenario import Scenario

# Shared pieces.
from hunger_games.training.common import Curriculum, EventLog, IterationStats, LearnerSpec, learner_ids

# The episode player.
from hunger_games.training.reinforce import play_rl_episode


@dataclass
class NeatTrainerConfig:
    """Every knob of the NEAT trainer."""

    # Genomes per generation.
    population_size: int = 48
    # Generations to run.
    generations: int = 30
    # Games each genome plays per generation.
    rounds_per_generation: int = 1
    # Learner copies per game (the same genome in several slots gives a steadier score).
    learners_per_game: int = 6
    # Fraction of each species allowed to reproduce.
    survival_threshold: float = 0.3
    # Generations without improvement before a species is removed.
    stagnation: int = 15
    # The number of species aimed for; the compatibility threshold adjusts to reach it.
    target_species: int = 8
    # Chance a child comes from crossover rather than a mutated clone.
    crossover_rate: float = 0.75
    # Champions of species this big are copied unchanged.
    elite_species_size: int = 5
    # Greedy validation games on fixed seeds for the champion.
    validation_games: int = 2
    # The first validation seed.
    validation_seed: int = 90000
    # CPU cores.
    workers: int = 1
    # Seed.
    seed: int | None = None
    # Record one game per generation for the training feed.
    record_showcase: bool = True
    # Mutation and speciation settings.
    neat: NeatConfig = field(default_factory=NeatConfig)


@dataclass
class Species:
    """A group of similar genomes."""

    # Id.
    id: int
    # The genome new members are compared against.
    representative: NeatGenome
    # Members this generation.
    members: list[NeatGenome] = field(default_factory=list)
    # Best fitness ever seen in this species.
    best_fitness: float = -np.inf
    # Generations since the best improved.
    stale: int = 0


def _neat_job(args: tuple) -> dict:
    """Unpack a job tuple for the process pool."""
    # Forward.
    return play_rl_episode(*args)


class NeatTrainer:
    """Evolves NEAT genomes against voting opponents."""

    # Label for run folders.
    method = "neat"

    def __init__(
        self,
        config: SimulationConfig,
        neat: NeatTrainerConfig,
        scenario: Scenario | None = None,
        initial_genome: dict | None = None,
        curriculum: Curriculum | None = None,
    ) -> None:
        """Create a minimal population (or clones of a given genome)."""
        # Settings.
        self.config = config
        self.neat = neat
        self.scenario = scenario
        self.curriculum = curriculum
        # Randomness.
        self.rng = np.random.default_rng(neat.seed)
        # Innovation numbers.
        self.tracker = InnovationTracker()
        # The population.
        if initial_genome is None:
            self.population = [
                NeatGenome.minimal(VECTOR_SIZE, MENU_SIZE, self.rng, neat.neat, self.tracker)
                for _ in range(neat.population_size)
            ]
        else:
            base = NeatGenome.from_dict(initial_genome)
            self.tracker.next_innovation = max((c.innovation for c in base.connections), default=0) + 1
            self.tracker.next_node = max(n.id for n in base.nodes) + 1
            self.population = [base.copy() for _ in range(neat.population_size)]
            for genome in self.population[1:]:
                genome.mutate(self.rng, neat.neat, self.tracker)
        # Species.
        self.species: list[Species] = []
        self.next_species_id = 0
        self.compatibility_threshold = neat.neat.compatibility_threshold
        # History.
        self.history: list[IterationStats] = []
        self.learning_history = self.history
        self.events = EventLog()
        self.generation = 0
        self._stop = False
        self._started: float | None = None
        # The best genome so far, and the id of the species it came from (never removed for stagnation).
        self.best: NeatGenome | None = None
        self.best_species_id: int | None = None
        # Game-level wins of the last evaluation.
        self._last_wins: list[int] = []
        self.best_mean_score = -np.inf

    @property
    def settings(self):
        """The trainer's own settings dataclass."""
        # NEAT settings.
        return self.neat

    def _learner_ids(self) -> list[int]:
        """Learner slots."""
        # Shared rule.
        return learner_ids(self.config.num_players, self.neat.learners_per_game)

    def _apply_curriculum(self) -> None:
        """Size the roster for the curriculum stage."""
        # Nothing without a curriculum.
        if self.curriculum is None:
            return
        players = min(self.neat.learners_per_game, 24) + self.curriculum.opponents
        self.config = SimulationConfig(**{**self.config.to_dict_raw(), "num_players": players})

    # ------------------------------------------------------------ evaluate

    def _play(self, jobs: list[tuple]) -> list[dict]:
        """Run episode jobs, in parallel if asked."""
        # Parallel.
        if self.neat.workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=self.neat.workers) as pool:
                return list(pool.map(_neat_job, jobs))
        # Sequential.
        return [_neat_job(job) for job in jobs]

    def evaluate(self, on_progress: Callable[[int, int], None] | None = None) -> tuple[list[dict], Recording | None]:
        """Score every genome by its mean episode return against voting opponents."""
        # Learners.
        learners = self._learner_ids()
        # Jobs: each genome, each round; the first job records.
        jobs = []
        owners = []
        for index, genome in enumerate(self.population):
            for r in range(self.neat.rounds_per_generation):
                seed = int(self.rng.integers(2**31 - 1))
                jobs.append(
                    (
                        self.config,
                        self.scenario,
                        LearnerSpec("neat", genome.to_dict()),
                        learners,
                        seed,
                        True,
                        index == 0 and r == 0 and self.neat.record_showcase,
                    )
                )
                owners.append(index)
        # Play.
        results = self._play(jobs)
        # Fitness = mean return.
        totals = np.zeros(len(self.population))
        counts = np.zeros(len(self.population))
        telemetry = []
        showcase = None
        self._last_wins = []
        for job_index, result in enumerate(results):
            returns = [o["return"] for o in result["outcomes"].values()]
            totals[owners[job_index]] += float(np.mean(returns)) if returns else 0.0
            counts[owners[job_index]] += 1
            telemetry.append(result["telemetry"])
            self._last_wins.append(int(result.get("learner_won", False)))
            if result.get("recording") is not None:
                showcase = result["recording"]
            if on_progress is not None:
                on_progress(job_index + 1, len(jobs))
        # Assign.
        for genome, total, count in zip(self.population, totals, counts, strict=False):
            genome.fitness = float(total / max(1, count))
        # Done.
        return telemetry, showcase

    def validate(self, genome: NeatGenome) -> tuple[float, float]:
        """Mean return and game-level win rate of a genome on the fixed validation seeds."""
        # None asked for.
        if self.neat.validation_games <= 0:
            return 0.0, 0.0
        learners = self._learner_ids()
        jobs = [
            (
                self.config,
                self.scenario,
                LearnerSpec("neat", genome.to_dict()),
                learners,
                self.neat.validation_seed + i,
                True,
                False,
            )
            for i in range(self.neat.validation_games)
        ]
        results = self._play(jobs)
        returns = [o["return"] for result in results for o in result["outcomes"].values()]
        wins = [int(result.get("learner_won", False)) for result in results]
        return (float(np.mean(returns)) if returns else 0.0), (float(np.mean(wins)) if wins else 0.0)

    # ------------------------------------------------------------ speciate

    def speciate(self) -> None:
        """Assign every genome to a species by distance to the species' representative."""
        # Clear members.
        for s in self.species:
            s.members = []
        # Assign.
        for genome in self.population:
            for s in self.species:
                if genome.distance(s.representative, self.neat.neat) < self.compatibility_threshold:
                    s.members.append(genome)
                    genome.species = s.id
                    break
            else:
                s = Species(self.next_species_id, genome.copy(), [genome])
                genome.species = s.id
                self.next_species_id += 1
                self.species.append(s)
                self.events.add("evolution", f"new species {s.id} founded ({genome.hidden_count} hidden nodes)")
        # Drop empty species.
        before = len(self.species)
        self.species = [s for s in self.species if s.members]
        if len(self.species) < before:
            self.events.add("evolution", f"{before - len(self.species)} species died out")
        # Nudge the threshold toward the target species count.
        if len(self.species) < self.neat.target_species:
            self.compatibility_threshold = max(0.5, self.compatibility_threshold - 0.3)
        elif len(self.species) > self.neat.target_species:
            self.compatibility_threshold += 0.3

    def reproduce(self) -> None:
        """Build the next population from the species."""
        # Stagnation and bests.
        for s in self.species:
            top = max(m.fitness for m in s.members)
            if top > s.best_fitness:
                s.best_fitness = top
                s.stale = 0
            else:
                s.stale += 1
        # Remove stagnant species (but never the one the champion came from).
        alive = [s for s in self.species if s.stale < self.neat.stagnation or s.id == self.best_species_id]
        if len(alive) < len(self.species):
            self.events.add("evolution", f"{len(self.species) - len(alive)} stagnant species removed")
        self.species = alive or self.species[:1]
        # Adjusted fitness (shared within the species), shifted so it is positive.
        floor = min(m.fitness for s in self.species for m in s.members)
        for s in self.species:
            for m in s.members:
                m.adjusted = (m.fitness - floor + 1e-3) / len(s.members)  # type: ignore[attr-defined]
        totals = {s.id: sum(m.adjusted for m in s.members) for s in self.species}  # type: ignore[attr-defined]
        grand = sum(totals.values()) or 1.0
        # Offspring per species.
        children: list[NeatGenome] = []
        self.tracker.reset_generation()
        for s in self.species:
            share = int(round(self.neat.population_size * totals[s.id] / grand))
            members = sorted(s.members, key=lambda m: m.fitness, reverse=True)
            # Elite.
            if len(members) >= self.neat.elite_species_size and share > 0:
                children.append(members[0].copy())
                share -= 1
            # Parents.
            parents = members[: max(1, int(np.ceil(len(members) * self.neat.survival_threshold)))]
            for _ in range(share):
                if len(parents) > 1 and self.rng.random() < self.neat.crossover_rate:
                    a, b = parents[int(self.rng.integers(len(parents)))], parents[int(self.rng.integers(len(parents)))]
                    fitter, other = (a, b) if a.fitness >= b.fitness else (b, a)
                    child = fitter.crossover(other, self.rng)
                else:
                    child = parents[int(self.rng.integers(len(parents)))].copy()
                child.mutate(self.rng, self.neat.neat, self.tracker)
                children.append(child)
            # New representative.
            s.representative = members[0].copy()
        # Fill or trim to the population size.
        while len(children) < self.neat.population_size:
            parent = max(self.population, key=lambda g: g.fitness)
            child = parent.copy()
            child.mutate(self.rng, self.neat.neat, self.tracker)
            children.append(child)
        self.population = children[: self.neat.population_size]

    # ------------------------------------------------------------ iterate

    def step(self, on_progress: Callable[[int, int], None] | None = None) -> IterationStats:
        """Evaluate, record, speciate and reproduce."""
        # Clocks.
        if self._started is None:
            self._started = time.time()
        started = time.time()
        # Curriculum.
        self._apply_curriculum()
        # Evaluate.
        telemetry, showcase = self.evaluate(on_progress)
        # Champion.
        champion = max(self.population, key=lambda g: g.fitness)
        if self.best is None or champion.fitness > self.best.fitness:
            self.best = champion.copy()
            self.events.add(
                "record",
                f"new champion: fitness {champion.fitness:.2f}, {champion.hidden_count} hidden nodes, {champion.enabled_count} connections",
            )
        # Validation.
        val_score, val_win_rate = self.validate(champion)
        # Merge telemetry.
        merged = BehaviorTelemetry.merge(telemetry) if telemetry else {}
        # Scores.
        scores = [g.fitness for g in self.population]
        mean_score = float(np.mean(scores))
        # Speciate and reproduce.
        self.speciate()
        # Remember which species the champion belongs to, so stagnation never removes it.
        if champion.fitness >= (self.best.fitness if self.best is not None else -np.inf):
            self.best_species_id = champion.species
        species_count = len(self.species)
        self.events.add(
            "evolution",
            f"generation {self.generation}: {species_count} species, best {champion.fitness:.2f}, mean {mean_score:.2f}",
        )
        self.reproduce()
        # Stats.
        stats = IterationStats(
            iteration=self.generation,
            scores=scores,
            mean_score=mean_score,
            best_score=float(champion.fitness),
            entropy=float(merged.get("entropy", 0.0)) if merged else 0.0,
            mean_length=float(merged.get("mean_survival_ticks", 0.0)) if merged else 0.0,
            win_rate=float(np.mean(self._last_wins)) if self._last_wins else 0.0,
            val_score=val_score,
            val_win_rate=val_win_rate,
            seconds=time.time() - started,
            cumulative_seconds=time.time() - self._started,
            stage=self.curriculum.stage if self.curriculum else 0,
            opponents=self.curriculum.opponents if self.curriculum else self.config.num_players - 1,
            extra={
                "species": species_count,
                "hidden_nodes": champion.hidden_count,
                "connections": champion.enabled_count,
                "threshold": self.compatibility_threshold,
            },
            learner=champion.to_dict(),
            telemetry=merged,
            showcase=showcase,
        )
        self.history.append(stats)
        if mean_score > self.best_mean_score:
            self.best_mean_score = mean_score
        # Curriculum (judged on validation wins when there are validation games).
        judged_win = val_win_rate if self.neat.validation_games > 0 else stats.win_rate
        if self.curriculum is not None and self.curriculum.observe(mean_score, judged_win):
            self.events.add(
                "curriculum", f"promoted to stage {self.curriculum.stage}: {self.curriculum.opponents} opponents"
            )
        # Count.
        self.generation += 1
        # Done.
        return stats

    def run(
        self,
        on_iteration: Callable[[IterationStats], None] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[IterationStats]:
        """Run every generation (or until `stop()`)."""
        # Reset.
        self._stop = False
        while self.generation < self.neat.generations and not self._stop:
            stats = self.step(on_progress)
            if on_iteration is not None:
                on_iteration(stats)
        return self.history

    def stop(self) -> None:
        """Finish after the current generation."""
        # Flag.
        self._stop = True

    # ------------------------------------------------------------ outputs

    @property
    def champion(self) -> dict | None:
        """The best genome so far as a dictionary."""
        # Best.
        return self.best.to_dict() if self.best is not None else None

    def champion_spec(self) -> LearnerSpec:
        """The best genome as a learner spec."""
        # NEAT.
        return LearnerSpec("neat", self.champion)

    def learner_spec(self) -> LearnerSpec:
        """The current best genome as a learner spec."""
        # Same as champion.
        return self.champion_spec()

    def champion_brain(self, chaos: float = 0.0):
        """A NeatBrain loaded with the champion."""
        # Build.
        from hunger_games.brain.neat import NeatBrain  # noqa: PLC0415

        return NeatBrain(self.best.copy() if self.best else self.population[0].copy(), chaos=chaos)

    def save_champion(self, path: str | Path) -> None:
        """Write the champion genome to JSON in the shared champion file shape."""
        # Data.
        data = {
            "brain_name": "neat",
            "genome": self.champion,
            "fitness": float(self.best.fitness) if self.best else 0.0,
            "generations": len(self.history),
            "method": "neat",
        }
        Path(path).write_text(json.dumps(data))

    def history_rows(self) -> list[dict]:
        """The history as rows."""
        # Rows.
        return [stats.to_row() for stats in self.history]
