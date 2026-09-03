# `genetic.py`

**Source:** [hunger_games/training/genetic.py](../../hunger_games/training/genetic.py)
**Depends on:** `json`, `time`, `collections.abc.Callable`, `concurrent.futures.ProcessPoolExecutor`, `dataclasses`, `pathlib` (standard library); `numpy`; [brain/__init__.py](../brain/init.md) (`Brain`, `create_brain`); [hunger_games/config.py](../config.md) (`NeuralConfig`, `SimulationConfig`); [hunger_games/game.py](../game.md) (`Game`); [hunger_games/recorder.py](../recorder.md) (`Recorder`, `Recording`); [hunger_games/research/telemetry.py](../research/telemetry.md) (`BehaviorTelemetry`); [hunger_games/scenario.py](../scenario.md) (`Scenario`); [training/common.py](common.md) (`Curriculum`, `EventLog`, `IterationStats`, `LearnerSpec`, `learner_ids`); [training/reinforce.py](reinforce.md) (`_run_episode_job`, imported inside `evaluate_against_voting` and `validate_with_wins` to avoid an import cycle)
**Used by:** [training/__init__.py](init.md) (re-exports `GenerationStats`, `GeneticTrainer`, `TrainingConfig`); [training/runs.py](runs.md) (`save_run` reads `settings`, `config`, `history`, `history_rows()`, `learning_history`, `events`, `champion`, `save_champion`); [research/comparison.py](../research/comparison.md) (the `"genetic"` method); [hunger_games/ui/session.py](../ui/session.md) (`GeneticTrainer` with `initial_genome` and `curriculum`, `TrainingConfig`, `previous_champion`, `load_champion`, `save_champion`, `learning_history`, `events`); [hunger_games/ui/app.py](../ui/app.md) (`TrainingConfig`, including the `opponents` combo); [experiments/run_ga.py](../experiments/run_ga.md); `tests/test_methods.py`; `tests/test_recorder_training.py`; `tests/test_research.py`; `tests/test_ui_session.py`; `tests/test_feed.py`; `tests/test_imitation.py` (the warm start)

## Purpose

This file is a genetic algorithm that evolves brains by playing games. The idea, in the words of the module docstring: keep a population of genomes (weight vectors). Each generation, put them into games, score each one, keep the best, and breed the rest by mixing two parents and adding small random mutations. Repeat.

Nothing here knows what a genome means. It works for the voting brain's eight genes and for a neural network's 5872 weights (the default `(64, 32)` architecture) alike, because it only ever calls `genome()` and `set_genome()`.

There are two ways to score a genome, chosen by `TrainingConfig.opponents`:

- **`"voting"` (the default).** Each genome plays as the learner against the voting brain, in the same games and with the same episode return that REINFORCE, PPO and NEAT use. Fitness is an absolute number, comparable across generations and across methods.
- **`"self"`.** The population plays itself, 24 genomes to a game, and fitness is placement plus small bonuses. This is the original tournament style from the first version of the trainer.

Either way the trainer also does what every trainer in this package does: validates the champion on fixed seeds and reports its score and its game-level win rate, records behaviour telemetry, keeps a showcase recording per generation, fills the shared `IterationStats`, logs events, applies a curriculum, and warm-starts from an `initial_genome`.

## Concepts you need

**Genome.** A flat numpy vector of floats that fully describes one brain. `Brain.genome()` reads it, `Brain.set_genome()` writes it. See [../brain/base.md](../brain/base.md).

**Fitness.** One number per genome saying how well it did. In `"voting"` mode it is the mean episode return. In `"self"` mode it is mostly placement (1.0 for the victor, 0.0 for first out), with small bonuses per kill and per day survived.

**Coevolution.** In `"self"` mode genomes are scored by playing *against each other*, so fitness is relative: a genome that scored 0.8 in generation 3 faced a different, weaker crowd than one scoring 0.8 in generation 30. Only `val_fitness`, measured against a fixed opponent on fixed seeds, is comparable across generations. In `"voting"` mode the opponents never change, so this problem goes away.

**Game-level win.** In `"voting"` mode a genome drives several copies of itself in one game, and only one tribute can be the victor. `play_rl_episode` reports `learner_won`, true when any copy won. Win rates here are fractions of games with that flag, so they can reach 1.0.

**Elitism, tournament selection, crossover, mutation.** The four moving parts of a GA. Elites are copied unchanged so the best is never lost. A tournament picks a parent by drawing a few random genomes and keeping the fittest. Crossover mixes two parents gene by gene. Mutation adds Gaussian noise to a fraction of genes.

**Warm start.** Starting the population from a known good genome instead of random ones. The population is that genome plus close relatives of it.

**Learner slots.** In `"voting"` mode a genome drives `learners_per_game` tributes spread across the roster (see `learner_ids` in [common.md](common.md)); everyone else is the config's brain. The score is the mean over those copies.

**Recordings.** A `Recorder` wraps a `Game` and copies the state after every tick. Only one game per generation is recorded, and it is left out of `to_row()`. See [../recorder.md](../recorder.md).

**Process pools.** With `workers > 1`, games run in separate Python processes via `ProcessPoolExecutor`. On macOS (and Windows) new processes start with `spawn`, which re-imports your main script. That is why the job functions here are top-level functions and why scripts need a `__main__` guard (see Gotchas).

## Walkthrough

### `TrainingConfig`

```python
@dataclass
class TrainingConfig:
```

Every knob of the algorithm. A plain dataclass, so `TrainingConfig(generations=5)` overrides one field and keeps the rest.

| Field | Default | Meaning |
| --- | --- | --- |
| `brain_name` | `"neural"` | Which brain kind is evolved (`"neural"` or `"voting"`) |
| `population_size` | `48` | How many genomes are alive at once (in `"self"` mode a multiple of `num_players` avoids padding) |
| `generations` | `20` | How many generations `run()` plays |
| `rounds_per_generation` | `2` | Games each genome plays per generation (more = steadier scores, slower) |
| `elite_fraction` | `0.1` | Fraction of the population copied unchanged into the next generation |
| `tournament_size` | `3` | Random genomes that compete to become a parent (bigger = stronger selection) |
| `crossover_rate` | `0.5` | Chance a child mixes two parents rather than cloning one |
| `mutation_rate` | `0.1` | Fraction of a child's genes that get a random nudge |
| `mutation_scale` | `0.1` | Standard deviation of that Gaussian nudge (also sets the spread of a warm start's relatives) |
| `workers` | `1` | CPU cores to evaluate games on |
| `seed` | `None` | Seed for the trainer's own randomness (population, breeding, game seeds) |
| `kills_weight` | `0.05` | Fitness bonus per kill (`"self"` mode) |
| `days_weight` | `0.01` | Fitness bonus per day survived (`"self"` mode) |
| `validation_games` | `2` | Games the champion plays per generation against the config's brain on fixed seeds |
| `validation_seed` | `90000` | The first validation seed (game `i` uses `validation_seed + i`) |
| `collect_telemetry` | `True` | Whether `"self"` evaluation games record behaviour (`"voting"` games always do) |
| `record_showcase` | `True` | Whether to record one real evaluation game per generation so the dashboard can replay training |
| `opponents` | `"voting"` | Who each genome plays against: `"voting"` (the learner against the video's brain, scored by episode return) or `"self"` (the population plays itself, scored by placement) |
| `learners_per_game` | `6` | Learner copies per game in `"voting"` mode |

Design reasoning: with the defaults in `"voting"` mode a generation is 96 episodes (48 genomes, 2 rounds), each with 6 copies of one genome against 18 voting opponents. In `"self"` mode it is 4 games of 24 (48 genomes, 2 rounds), the smallest run that gives each genome two independent placements.

### `GenerationStats`

```python
@dataclass
class GenerationStats:
```

What happened in one generation. Built by `step_generation` and kept in `trainer.history`.

| Field | Type | Meaning |
| --- | --- | --- |
| `generation` | `int` | Which generation (0 is the first) |
| `best_fitness` | `float` | Best mean fitness in the population |
| `mean_fitness` | `float` | Average fitness |
| `worst_fitness` | `float` | Worst fitness |
| `champion` | `np.ndarray` | A copy of the best genome of this generation |
| `seconds` | `float` | Wall-clock seconds the generation took (evaluation plus validation plus breeding) |
| `val_fitness` | `float` (default `0.0`) | Mean score of this generation's champion in the validation games |
| `cumulative_seconds` | `float` (default `0.0`) | Seconds since training started |
| `telemetry` | `dict` (default `{}`, `repr=False`) | Merged `BehaviorTelemetry.summary()` of this generation's evaluation games |
| `showcase` | `Recording | None` (default `None`, `repr=False`) | A recording of one real evaluation game from this generation, or `None` when `record_showcase` is off |
| `val_win_rate` | `float` (default `0.0`) | Share of validation games this generation's champion won (game-level) |
| `stage` | `int` (default `0`) | Curriculum stage the generation was played at (0 without a curriculum). Both new fields sit after `showcase` because `step` builds the stats positionally |

There is no validation win rate on `GenerationStats`. It goes to the shared `IterationStats.val_win_rate` through `_last_val_win_rate` (see `step_generation`).

#### `to_row()`

```python
def to_row(self) -> dict:
```

Returns every field except `champion`, `telemetry` and `showcase` as a plain dictionary. `history.json` uses these rows; the shared `learning.json` comes from `learning_history` instead.

### `fitness_of(...)`

```python
def fitness_of(placement: int, kills: int, days: float, num_players: int, kills_weight: float, days_weight: float) -> float:
```

Scores one game for one tribute in `"self"` mode. `placing = (num_players - placement) / max(1, num_players - 1)` maps placement 1 to 1.0 and placement `num_players` to 0.0. Then `kills_weight * kills + days_weight * days` is added.

Example with 24 players and the default weights: the victor with 2 kills after 10 days scores `1.0 + 0.1 + 0.1 = 1.2`. A tribute placed 12th with no kills after 5 days scores `12/23 + 0.05 = 0.57`.

### `play_evaluation_game(...)`

```python
def play_evaluation_game(config: SimulationConfig, scenario: Scenario | None, brain_name: str, genomes: list[np.ndarray], seed: int, collect_telemetry: bool = False, record: bool = False) -> tuple[list[tuple[int, int, float]], dict | None, Recording | None]:
```

Plays one `"self"` mode game where tribute `i` is driven by `genomes[i]`. Copies the config with `num_players = len(genomes)` and the seed, builds each tribute's brain with `create_brain(brain_name, ...)` plus `set_genome`, optionally attaches a `BehaviorTelemetry` that tracks every tribute, and plays with a `Recorder` when `record` is on. Returns `(placement, kills, days_survived)` per genome, the telemetry summary or `None`, and the `Recording` or `None`. A top-level function so worker processes can run it.

### `play_validation_game(...)`

```python
def play_validation_game(config: SimulationConfig, scenario: Scenario | None, brain_name: str, genome: np.ndarray, learner_ids: list[int], seed: int) -> list[tuple[int, int, float]]:
```

Plays one game where the slots in `learner_ids` carry `genome` and everyone else uses `create_brain(config.brain_name, ...)`. Returns `(placement, kills, days)` for the learners only. Used by `validate_with_wins` in `"self"` mode.

### `_run_job(args)` and `_run_validation_job(args)`

```python
def _run_job(args: tuple) -> tuple[list[tuple[int, int, float]], dict | None, Recording | None]:
def _run_validation_job(args: tuple) -> list[tuple[int, int, float]]:
```

Tuple-unpacking wrappers for `ProcessPoolExecutor.map`. `"voting"` mode uses `_run_episode_job` from [reinforce.md](reinforce.md) instead.

### `GeneticTrainer`

```python
class GeneticTrainer:
```

#### `__init__(config, training, scenario=None, initial_genome=None, curriculum=None)`

```python
def __init__(self, config: SimulationConfig, training: TrainingConfig, scenario: Scenario | None = None, initial_genome: np.ndarray | None = None, curriculum: Curriculum | None = None) -> None:
```

Stores `config`, `training`, `scenario` and `curriculum`, makes an `EventLog` as `events`, an empty `learning_history` and `best_mean_score = -inf`. Seeds `self.rng = np.random.default_rng(training.seed)`, builds one template brain to learn the genome length, and raises `ValueError` if that length is 0.

The starting population is built one of two ways:

- **`initial_genome is None` (a cold start).** `population_size` fresh brains are created and their genomes kept.
- **`initial_genome` given (a warm start).** `population[0]` is an exact copy. The other entries are the genome plus Gaussian noise with standard deviation `0.25 * training.mutation_scale`. The spread is a quarter of the mutation scale because a trained network's weights are small and full-size noise erases its instincts. Only the constructor uses this quarter scale.

Also sets up `fitness` (zeros), `history`, `generation = 0`, `_stop`, `_started`, `_last_telemetry` and `_last_showcase`. `_last_val_win_rate` is not created here; `step_generation` sets it and `_record_iteration` reads it with a `getattr` default of `0.0`.

#### `settings` (property)

```python
@property
def settings(self):
```

Returns `self.training`, for `save_run`.

#### `_make_jobs()`

```python
def _make_jobs(self) -> list[tuple[list[int], int]]:
```

`"self"` mode only. For each round it shuffles the genome indices, pads with random extra entrants until the count divides by `num_players`, cuts the list into games, and pairs each game with a random seed. Returns `[(indices, seed), ...]`.

#### `evaluate(on_progress=None)`

```python
def evaluate(self, on_progress: Callable[[int, int], None] | None = None) -> np.ndarray:
```

`"self"` mode scoring. Plays every job from `_make_jobs`, records only the first job when `record_showcase` is on, folds each tribute's `fitness_of(...)` into per-genome totals, keeps the telemetry summaries in `_last_telemetry` and the recording in `_last_showcase`, and returns the mean fitness per genome (also stored in `self.fitness`).

#### `_tournament()` and `_child()`

```python
def _tournament(self) -> np.ndarray:
def _child(self) -> np.ndarray:
```

`_tournament` draws `tournament_size` random indices and returns the fittest genome among them. `_child` picks a parent by tournament, with probability `crossover_rate` mixes in a second parent with a random 50/50 mask, then adds `normal(0, mutation_scale)` to a `mutation_rate` fraction of genes. Parents are never modified.

#### `_learner_ids()`

```python
def _learner_ids(self) -> list[int]:
```

`learner_ids(config.num_players, training.learners_per_game)`. With 6 of 24 that is `[0, 4, 8, 12, 16, 20]`. Used for the learner slots in `"voting"` evaluation and in validation in both modes.

#### `_kind()`

```python
def _kind(self) -> str:
```

`"neural"` when `brain_name` is `"neural"`, else `"voting"`. This is the `LearnerSpec.kind` the episode player needs to rebuild the brain.

#### `learner_spec()` and `champion_spec()`

```python
def learner_spec(self) -> LearnerSpec:
def champion_spec(self) -> LearnerSpec:
```

`LearnerSpec(self._kind(), genome, config.neural)` where `genome` is the champion, or `population[0]` before any generation. `champion_spec` is an alias.

#### `_apply_curriculum()`

```python
def _apply_curriculum(self) -> None:
```

Only in `"voting"` mode with a curriculum: copies the config with `num_players = min(learners_per_game, 24) + curriculum.opponents`. In `"self"` mode a curriculum is ignored, because the population sets the roster.

#### `evaluate_against_voting(on_progress=None)`

```python
def evaluate_against_voting(self, on_progress: Callable[[int, int], None] | None = None) -> tuple[np.ndarray, list[dict], Recording | None, list[float], list[int]]:
```

`"voting"` mode scoring. For every genome and every round it builds a `play_rl_episode` job `(config, scenario, LearnerSpec(kind, genome, neural), learner_ids, seed, True, record)`, with `record` only for the first genome's first round when `record_showcase` is on. `True` is `greedy`, so genomes play with chaos 0. Jobs run through a pool when `workers > 1`. For each job: the outcomes (one per learner slot) are averaged into the owning genome's total, the telemetry summary is kept, every learner's `survival` is added to `lengths`, and one entry `int(result["learner_won"])` is added to `wins`. So `wins` has one flag per game (genome times round), not per learner copy. Sets `self.fitness`, `_last_telemetry` and `_last_showcase`. Returns `(fitness, telemetry, showcase, lengths, wins)`.

#### `validate(genome)`

```python
def validate(self, genome: np.ndarray) -> float:
```

`self.validate_with_wins(genome)[0]`: the validation score only, kept for callers that want a single number.

#### `validate_with_wins(genome)`

```python
def validate_with_wins(self, genome: np.ndarray) -> tuple[float, float]:
```

Mean score and win rate of a genome on the fixed validation seeds. Returns `(0.0, 0.0)` if `validation_games <= 0`.

| Mode | Games | Score | Win rate |
| --- | --- | --- | --- |
| `"voting"` | `validation_games` `play_rl_episode` jobs on seeds `validation_seed + i`, greedy, never recorded | Mean episode return over every learner slot, the same number the fitness uses | Game-level: the mean of `learner_won` over the games |
| `"self"` | `validation_games` `play_validation_game` jobs on the same seeds | Mean `fitness_of` over every learner row | Per learner row: the fraction of rows with placement 1 |

Note the two modes count wins differently. `"self"` mode has no `learner_won` flag, so it counts first placings per copy, which caps the rate at one over `learners_per_game`.

#### `step_generation(on_progress=None)`

```python
def step_generation(self, on_progress: Callable[[int, int], None] | None = None) -> GenerationStats:
```

One full generation:

1. Start the clocks.
2. `_apply_curriculum()`.
3. Score everyone: `evaluate_against_voting` in `"voting"` mode (which also gives `lengths` and `wins`), else `evaluate`.
4. `ranking = np.argsort(fitness)[::-1]`, best first. The champion is a copy of `population[ranking[0]]`.
5. `val_fitness, val_win_rate = self.validate_with_wins(champion)`, and `self._last_val_win_rate = val_win_rate`. The stats then get `val_win_rate` and the curriculum's `stage`, which the `champion` property ranks generations by.
6. Merge this generation's telemetry.
7. Build `GenerationStats` and append it to `history`.
8. `_record_iteration(stats, list(fitness), lengths, wins)`, the shared record.
9. Keep `elite_count = max(1, int(elite_fraction * population_size))` elites as copies, fill the rest with `_child()`, replace the population, and increment `generation`.

Returns the stats. The population is replaced *after* the stats are recorded, so `history[-1].champion` is the best genome of the population that was just scored, and it survives into the next population as elite number one.

#### `run(on_generation=None, on_progress=None)`

```python
def run(self, on_generation: Callable[[GenerationStats], None] | None = None, on_progress: Callable[[int, int], None] | None = None) -> list[GenerationStats]:
```

Clears `_stop`, loops `step_generation` while `generation < training.generations` and not stopped, calling `on_generation(stats)` after each. Returns `self.history`. Calling `run()` again continues.

#### `stop()`

```python
def stop(self) -> None:
```

Sets `_stop`; a running `run()` finishes the current generation and returns.

#### `_record_iteration(stats, scores, lengths, wins)`

```python
def _record_iteration(self, stats: "GenerationStats", scores: list[float], lengths: list[float], wins: list[int]) -> None:
```

Appends the unified `IterationStats`, logs events, and advances the curriculum:

- `iteration = stats.generation`, `scores` = every genome's fitness, `mean_score` their mean, `best_score = stats.best_fitness`.
- `entropy` from the telemetry (`0.0` without).
- `mean_length` is the mean of `lengths` when there are any (`"voting"` mode), else the telemetry's `mean_survival_ticks`.
- `win_rate` is the mean of `wins` when there are any (`"voting"` mode, game-level), else the telemetry's `win_rate`.
- `val_score = stats.val_fitness`; `val_win_rate = getattr(self, "_last_val_win_rate", 0.0)`.
- The timings, `stage` and `opponents` from the curriculum (or `num_players - 1`), `extra={"worst_fitness": ...}`, `learner = stats.champion`, the telemetry and the showcase.

Then an `"evolution"` event (`generation G: best B, mean M, validation V`), a `"record"` event when `best_fitness` beats the best seen so far (kept in `best_mean_score`), and `curriculum.observe(mean_score)` with a `"curriculum"` event on promotion.

**The curriculum call passes the win rate.** `observe` is called with the mean score and `judged_win`, which is `record.val_win_rate` when `validation_games > 0` and `record.win_rate` otherwise, so the genetic learner is promoted only by winning games, like the other trainers.

#### `step(on_progress=None)`

```python
def step(self, on_progress: Callable[[int, int], None] | None = None) -> IterationStats:
```

Runs `step_generation` and returns `learning_history[-1]`. This is the method the dashboard and the method comparison call.

#### `history_rows()`

```python
def history_rows(self) -> list[dict]:
```

`[stats.to_row() for stats in self.history]`. No genomes, telemetry or recordings.

#### `previous_champion()`

```python
def previous_champion(self) -> np.ndarray | None:
```

`history[-2].champion` if there are at least two generations, else `None`. The dashboard compares it with the latest champion to highlight which genes changed.

#### `champion` (property)

```python
@property
def champion(self) -> np.ndarray | None:
```

The champion of the generation that ranks first by `champion_key(stage, val_win_rate, val_fitness)` ([common.md](common.md)): highest curriculum stage, then validation win rate, then validation fitness. `None` before any generation has run. It used to be the generation with the highest training fitness, which could hand the tournament a genome from an easy rung of the curriculum.

#### `champion_brain(chaos=None)`

```python
def champion_brain(self, chaos: float | None = None) -> Brain:
```

A fresh brain of `training.brain_name` with the champion loaded. `chaos` defaults to `config.chaos`.

#### `save_champion(path)`

```python
def save_champion(self, path: str | Path) -> None:
```

Raises `ValueError("No generations have been run yet")` if there is no champion. Otherwise writes JSON with `brain_name`, `neural` (the `NeuralConfig` as a dict), `genome` (a list), `fitness` (the best `best_fitness` in history) and `generations`. No `method` key.

#### `load_champion(path)` (static)

```python
@staticmethod
def load_champion(path: str | Path) -> dict:
```

Reads any trainer's champion file back. A list `genome` becomes a float numpy array; a dictionary `genome` (a NEAT champion) is left as it is. `neural`, when present, becomes a `NeuralConfig` with `hidden_layers` as a tuple. Extra keys pass through untouched.

## How to use it / experiment

**A minimal run (voting opponents).**

```python
from hunger_games.config import SimulationConfig
from hunger_games.training import GeneticTrainer, TrainingConfig

config = SimulationConfig(width=80, height=80, max_days=8)
trainer = GeneticTrainer(config, TrainingConfig(population_size=24, generations=10, seed=0))
trainer.run()
for s in trainer.learning_history:
    print(f"gen {s.iteration:2d} best {s.best_score:.2f} mean {s.mean_score:.2f} val {s.val_score:.2f} val win {s.val_win_rate:.2f}")
trainer.save_champion("champion.json")
```

**The original tournament.** `TrainingConfig(brain_name="voting", opponents="self", population_size=48)` evolves the voting brain's eight genes by playing the population against itself. Read `val_fitness`, not `best_fitness`, for progress.

**Warm-start from an imitation champion.**

```python
from hunger_games.training import ImitationConfig, ImitationTrainer

student = ImitationTrainer(config, ImitationConfig(seed=0))
student.run()
trainer = GeneticTrainer(config, TrainingConfig(mutation_scale=0.02, seed=0), initial_genome=student.champion)
trainer.run()
```

The relatives in the first population sit at `0.25 * 0.02 = 0.005` from the champion, and children get nudges of scale 0.02.

**With the curriculum.**

```python
from hunger_games.training import Curriculum, CurriculumConfig
curriculum = Curriculum(CurriculumConfig(promote_on="score", threshold=3.0, window=5))
trainer = GeneticTrainer(config, TrainingConfig(seed=0), curriculum=curriculum)
```

Every generation's episodes are sized to the stage; `learning_history[i].opponents` shows the ladder. Because promotion is judged on validation wins, the ladder is climbed only when the champion wins a majority of its validation games at each stage.

**Use the cores.** Put the code in a function, guard it, and set `workers`:

```python
def main() -> None:
    trainer = GeneticTrainer(config, TrainingConfig(workers=4, seed=0))
    trainer.run(on_generation=print)

if __name__ == "__main__":
    main()
```

**Watch a generation.** Every `GenerationStats.showcase` is a recording of the generation's first evaluation game. In `"voting"` mode that is the first genome's first round. Export it with `export_recording_gif(trainer.history[-1].showcase, "gen_last.gif")`.

**Selection pressure.** `tournament_size=2` is gentle and keeps diversity; `tournament_size=6` converges fast and may get stuck. `elite_fraction=0.0` still keeps one elite because of the `max(1, ...)`.

## Gotchas

- Promotion is judged on validation wins (game-level), so with `validation_games=2` a single win moves the five-iteration window by 0.1; raise `validation_games` for a steadier signal.
- **The macOS spawn rule.** On macOS, `multiprocessing` starts workers with `spawn`, which imports your script afresh in each worker. Without `if __name__ == "__main__":` the worker re-runs the training code at import time. The dashboard avoids the issue by running the trainer in a thread with `workers=1` by default.
- Everything sent to a worker is pickled: `config`, `scenario`, the genomes (inside a `LearnerSpec` in `"voting"` mode). A `Scenario` with a large painted map is copied once per job.
- **Warm starts with the default mutation scale lose the instincts.** Every child after generation 0 gets full `mutation_scale` noise on 10 percent of its genes. Use a scale around 0.02 for a warm-started neural population.
- **Showcases stay in memory.** Every generation's recording lives in `history` until the trainer is dropped. Set `record_showcase=False` for long runs.
- In `"voting"` mode genomes play greedily (chaos 0), so a genome's score on a given seed is deterministic. `rounds_per_generation` sets how many seeds each genome sees.
- `win_rate` in `"voting"` mode is over every evaluation game of the whole population (96 with the defaults), not the champion's games. It says how often any genome's copies won, which is a population-wide number. `val_win_rate` is the champion's.
- `"self"` mode validation counts wins per learner row, not per game, so its `val_win_rate` is capped at one over `learners_per_game`. `"voting"` mode counts games.
- `champion` is picked by training `best_fitness`. The generation with the highest `val_fitness` may be a different one; `max(trainer.history, key=lambda s: s.val_fitness).champion` gives that genome.
- The attribute that tracks the `"record"` events is called `best_mean_score` but holds the best `best_fitness`, not a mean.
- In `"self"` mode the telemetry counts every tribute (they are all population members); in `"voting"` mode it counts only the learner slots.
- `collect_telemetry=False` only affects `"self"` mode. `play_rl_episode` always measures the learners.
- `validation_games=0` makes `val_fitness` and `val_win_rate` always `0.0` and the validation line in `fitness.png` flat at zero.
- `GeneticTrainer.__init__` and `champion_brain` build brains without the `endgame_instinct` argument, and `build_learner` builds neural learners without it too. Only opponents get `config.endgame_instinct`.
- A `champion.json` from one `NeuralConfig` will not load into a brain with a different architecture: `set_genome` raises `ValueError` about the size. The same applies to `initial_genome`.
- The curriculum is ignored in `"self"` mode, and `IterationStats.stage` then stays 0 while `opponents` is `num_players - 1`.
