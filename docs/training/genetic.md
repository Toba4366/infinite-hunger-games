# `genetic.py`

**Source:** [hunger_games/training/genetic.py](../../hunger_games/training/genetic.py)
**Depends on:** `json`, `time`, `collections.abc.Callable`, `concurrent.futures.ProcessPoolExecutor`, `dataclasses`, `pathlib` (standard library); `numpy`; [brain/__init__.py](../brain/init.md) (`Brain`, `create_brain`); [hunger_games/config.py](../config.md) (`NeuralConfig`, `SimulationConfig`); [hunger_games/game.py](../game.md) (`Game`); [hunger_games/recorder.py](../recorder.md) (`Recorder`, `Recording`); [hunger_games/research/telemetry.py](../research/telemetry.md) (`BehaviorTelemetry`); [hunger_games/scenario.py](../scenario.md) (`Scenario`)
**Used by:** [training/__init__.py](init.md) (re-exports `GenerationStats`, `GeneticTrainer`, `TrainingConfig`); [training/runs.py](runs.md) (`save_run` reads `settings`, `config`, `history`, `history_rows()`, `champion`, `save_champion`); [hunger_games/ui/session.py](../ui/session.md) (`GeneticTrainer` with `initial_genome` for warm starts, `TrainingConfig`, `previous_champion`, `load_champion`, `save_champion`, and `history[-1].showcase` for the training feed); [hunger_games/ui/app.py](../ui/app.md) (`TrainingConfig`); [experiments/run_ga.py](../experiments/run_ga.md); `tests/test_recorder_training.py`; `tests/test_research.py`; `tests/test_ui_session.py`; `tests/test_feed.py`; `tests/test_imitation.py` (the warm start)

## Purpose

This file is a genetic algorithm that evolves brains by playing games. The idea, in the words of the module docstring: keep a population of genomes (weight vectors). Each generation, put them into games as the tributes, score each one by how it placed (plus a little for kills and days survived), keep the best, and breed the rest by mixing two parents and adding small random mutations. Repeat.

Nothing here knows what a genome means. It works for the voting brain's eight genes and for a neural network's 5872 weights (the default `(64, 32)` architecture) alike, because it only ever calls `genome()` and `set_genome()`.

Five things were added since the first version of this trainer, and a researcher will care about all of them:

- **Validation.** Each generation's champion also plays a few games on *fixed* seeds against the config's default brain. That gives a yardstick that does not move as the population changes.
- **Telemetry.** The evaluation games can record behaviour (which actions, at what thirst, where on the map) through `BehaviorTelemetry`, so the charts in a run folder show *how* play changed, not just the score.
- **Timing and rows.** Every generation records its own seconds and the cumulative seconds, and `to_row()` / `history_rows()` give JSON-friendly rows for `history.json` and the plots.
- **Showcase recordings.** With `record_showcase` on, the first evaluation game of every generation is recorded tick by tick with a `Recorder`. The `Recording` is stored on `GenerationStats.showcase`, and the dashboard's training feed replays it so you can watch the population play for real while training runs.
- **Warm starts.** The constructor's `initial_genome` seeds the population from an existing genome, usually the champion of an imitation run ([imitation.md](imitation.md)). A fresh random neural population dies of thirst before selection has anything to select; a warm-started one begins with working instincts.

## Concepts you need

**Genome.** A flat numpy vector of floats that fully describes one brain. `Brain.genome()` reads it, `Brain.set_genome()` writes it. See [../brain/base.md](../brain/base.md).

**Fitness.** One number per genome saying how well it did. Here it is mostly placement (1.0 for the victor, 0.0 for first out), with small bonuses per kill and per day survived.

**Coevolution.** Genomes are scored by playing *against each other*. That means fitness is relative: a genome that scored 0.8 in generation 3 faced a different, weaker crowd than one scoring 0.8 in generation 30. Only `val_fitness`, measured against a fixed opponent on fixed seeds, is comparable across generations.

**Elitism, tournament selection, crossover, mutation.** The four moving parts of a GA. Elites are copied unchanged so the best is never lost. A tournament picks a parent by drawing a few random genomes and keeping the fittest. Crossover mixes two parents gene by gene. Mutation adds Gaussian noise to a fraction of genes.

**Warm start.** Starting the population from a known good genome instead of random ones. The population is that genome plus close relatives of it, so the first generation already has variety to select from without losing the instincts.

**Recordings.** A `Recorder` wraps a `Game`, steps it, and copies the state after every tick into a `Frame`. `Recorder(game).record_all()` plays the game to the end and returns the finished `Recording`, with `result` filled in. See [../recorder.md](../recorder.md). A recording holds one frame per tick, so it is much bigger than a fitness number; that is why only one game per generation is recorded and why it is left out of `to_row()`.

**Process pools.** With `workers > 1`, games run in separate Python processes via `ProcessPoolExecutor`. Every argument and result is pickled and sent across. On macOS (and Windows) new processes are started with the `spawn` method, which re-imports your main script. That is why the job functions here are top-level functions and why scripts need a `__main__` guard (see Gotchas).

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
| `population_size` | `48` | How many genomes are alive at once (a multiple of `num_players` avoids padding) |
| `generations` | `20` | How many generations `run()` plays |
| `rounds_per_generation` | `2` | Games each genome plays per generation (more = steadier scores, slower) |
| `elite_fraction` | `0.1` | Fraction of the population copied unchanged into the next generation |
| `tournament_size` | `3` | Random genomes that compete to become a parent (bigger = stronger selection) |
| `crossover_rate` | `0.5` | Chance a child mixes two parents rather than cloning one |
| `mutation_rate` | `0.1` | Fraction of a child's genes that get a random nudge |
| `mutation_scale` | `0.1` | Standard deviation of that Gaussian nudge (also sets the spread of a warm start's relatives) |
| `workers` | `1` | CPU cores to evaluate games on |
| `seed` | `None` | Seed for the trainer's own randomness (population, breeding, game seeds) |
| `kills_weight` | `0.05` | Fitness bonus per kill |
| `days_weight` | `0.01` | Fitness bonus per day survived |
| `validation_games` | `2` | Games the champion plays per generation against the config's brain on fixed seeds |
| `validation_seed` | `90000` | The first validation seed (game `i` uses `validation_seed + i`) |
| `collect_telemetry` | `True` | Whether evaluation games record behaviour (slower, but gives the behaviour charts) |
| `record_showcase` | `True` | Whether to record one real evaluation game per generation so the dashboard can replay training |

Design reasoning: the defaults make 4 games per generation with 24 players (48 genomes, 2 rounds), which is the smallest run that gives each genome two independent placements.

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
| `val_fitness` | `float` (default `0.0`) | Mean fitness of this generation's champion in the validation games |
| `cumulative_seconds` | `float` (default `0.0`) | Seconds since training started |
| `telemetry` | `dict` (default `{}`, `repr=False`) | Merged `BehaviorTelemetry.summary()` of this generation's evaluation games |
| `showcase` | `Recording | None` (default `None`, `repr=False`) | A recording of one real evaluation game from this generation, or `None` when `record_showcase` is off |

#### `to_row()`

```python
def to_row(self) -> dict:
```

Returns every field except `champion`, `telemetry` and `showcase` as a plain dictionary. It walks `self.__dict__` and skips those three keys, so a field added to the dataclass later shows up in rows automatically. The three are left out because they are large (a default neural genome is 5872 floats, a telemetry summary holds heatmaps, a recording holds one frame per tick) and none of them is JSON. `history.json` and the dashboard's table use these rows. `save_run` collects the telemetry separately; the showcase stays in memory only.

### `fitness_of(...)`

```python
def fitness_of(placement: int, kills: int, days: float, num_players: int, kills_weight: float, days_weight: float) -> float:
```

Scores one game for one tribute. `placing = (num_players - placement) / max(1, num_players - 1)` maps placement 1 to 1.0 and placement `num_players` to 0.0. Then `kills_weight * kills + days_weight * days` is added.

Example with 24 players and the default weights: the victor with 2 kills after 10 days scores `1.0 + 0.1 + 0.1 = 1.2`. A tribute placed 12th with no kills after 5 days scores `12/23 + 0.05 = 0.57`. A draw where 3 remain gives each survivor placement 3, so `21/23 = 0.913` before bonuses.

Design reasoning: placement dominates, so the GA optimises survival to the end. The small kill bonus breaks ties toward tributes that fight; the days bonus rewards lasting longer even among early deaths. Both are deliberately tiny so they never outweigh one extra place.

### `play_evaluation_game(...)`

```python
def play_evaluation_game(config: SimulationConfig, scenario: Scenario | None, brain_name: str, genomes: list[np.ndarray], seed: int, collect_telemetry: bool = False, record: bool = False) -> tuple[list[tuple[int, int, float]], dict | None, Recording | None]:
```

Plays one game where tribute `i` is driven by `genomes[i]`. Steps:

1. Copy the config with `num_players = len(genomes)` and `seed = seed` (via `to_dict_raw()`).
2. Define a `factory(index, rng)` that calls `create_brain(brain_name, chaos, rng, neural, endgame_instinct)` and loads `genomes[index]` into it.
3. Build `Game(game_config, 0, brain_factory=factory, scenario=scenario)`. Game id is always 0.
4. If `collect_telemetry`, attach a `BehaviorTelemetry(width, height)` that tracks every tribute.
5. Play. If `record`, `Recorder(game).record_all()` plays the game while capturing every tick, and the result is read from `recording.result`. Otherwise `game.run()` plays it with no recording.
6. Look each tribute up in `result.players` by `player_id`.

Returns a 3-tuple: a list of `(placement, kills, days_survived)` in genome order, then `telemetry.summary()` or `None`, then the `Recording` or `None`.

It is a top-level function (not a method) so worker processes can import and run it. The telemetry here counts *all* tributes, because every tribute in an evaluation game is a population member. The recording is of the same game the scores come from, so what the feed shows is exactly what was scored.

### `play_validation_game(...)`

```python
def play_validation_game(config: SimulationConfig, scenario: Scenario | None, brain_name: str, genome: np.ndarray, learner_ids: list[int], seed: int) -> list[tuple[int, int, float]]:
```

Plays one game where the tribute slots in `learner_ids` carry the champion `genome` and everyone else gets `create_brain(config.brain_name, ...)`, the config's default brain (`"voting"` unless changed). The player count is the config's own `num_players`. Returns `(placement, kills, days)` for the learners only. No telemetry, no recording.

Design reasoning: the opponents never change and the seed is fixed, so the same champion always gets the same score. This is the held-out test set of the GA.

### `_run_job(args)` and `_run_validation_job(args)`

```python
def _run_job(args: tuple) -> tuple[list[tuple[int, int, float]], dict | None, Recording | None]:
def _run_validation_job(args: tuple) -> list[tuple[int, int, float]]:
```

Each unpacks a tuple and forwards to the matching play function. `ProcessPoolExecutor.map` sends one argument per call, so a tuple-unpacking wrapper is the simplest way to pass seven arguments. `_run_job` returns the same 3-tuple as `play_evaluation_game`.

### `GeneticTrainer`

```python
class GeneticTrainer:
```

Evolves a population of genomes by playing them against each other.

#### `__init__(config, training, scenario=None, initial_genome=None)`

```python
def __init__(self, config: SimulationConfig, training: TrainingConfig, scenario: Scenario | None = None, initial_genome: np.ndarray | None = None) -> None:
```

Stores `config`, `training` and `scenario`, seeds `self.rng = np.random.default_rng(training.seed)`, builds one template brain to learn the genome length, and raises `ValueError` if that length is 0.

Then it builds the starting population, one of two ways:

- **`initial_genome is None` (a cold start).** `population_size` fresh brains are created and their genomes kept, so every starting genome is drawn by the brain's own initializer (see [../brain/initializers.md](../brain/initializers.md)).
- **`initial_genome` given (a warm start).** `population[0]` is an exact copy of the genome. The other `population_size - 1` entries are close relatives: the genome plus Gaussian noise with standard deviation `0.25 * training.mutation_scale`, drawn from `self.rng`. The spread is a quarter of the mutation scale because a trained network's weights are small and full-size noise erases its instincts. Only the constructor uses this quarter scale; breeding uses the full `mutation_scale` as usual.

Also sets up `fitness` (zeros), `history`, `generation = 0`, `_stop`, `_started` (the training start time, `None` until the first generation), `_last_telemetry` (the telemetry summaries from the last `evaluate()`) and `_last_showcase` (the recording made by the last `evaluate()`, or `None`).

The genome must fit `training.brain_name` and `config.neural`. The dashboard drops the warm start when the GA is set to evolve the voting brain, because an imitation champion is a neural genome.

#### `settings` (property)

```python
@property
def settings(self):
```

Returns `self.training`. Every trainer exposes this name so `save_run` can write the trainer's settings to `config.json` without knowing which trainer it has.

#### `_make_jobs()`

```python
def _make_jobs(self) -> list[tuple[list[int], int]]:
```

Splits the population into games. For each round it shuffles the genome indices, pads with random extra entrants until the count divides by `num_players`, cuts the list into games, and pairs each game with a random seed below `2**31 - 1`. Returns `[(indices, seed), ...]`.

Example: 48 genomes, 24 players, 2 rounds gives 4 jobs. With 50 genomes the padding adds 22 random duplicates per round, so some genomes play more than once per round. That is why the config comment recommends a multiple of the player count.

#### `evaluate(on_progress=None)`

```python
def evaluate(self, on_progress: Callable[[int, int], None] | None = None) -> np.ndarray:
```

Plays every job and returns the mean fitness of each genome (also stored in `self.fitness`). Each job's arguments are `(config, scenario, brain_name, genomes, seed, collect_telemetry, record)`, where `record` is `job_index == 0 and self.training.record_showcase`. So at most one job per generation records, and it is always the first one. Before playing, `_last_telemetry` is emptied and `_last_showcase` is set to `None`.

An inner `absorb(job_index, job_result)` unpacks the 3-tuple `(outcome, telemetry, recording)`, keeps the telemetry summary if there is one, keeps the recording in `_last_showcase` if there is one, adds each tribute's `fitness_of(...)` into `totals`, counts the game, and calls `on_progress(done, total)`. With `workers > 1` the jobs go through `pool.map(_run_job, arguments)`, which returns results in job order; otherwise they run one at a time. The mean is `totals / max(counts, 1)`.

#### `_tournament()`

```python
def _tournament(self) -> np.ndarray:
```

Draws `tournament_size` random indices and returns the genome with the highest `self.fitness` among them. With size 3, the best genome in the population wins any tournament it enters, and the worst genome can only be picked if all three draws land on it.

#### `_child()`

```python
def _child(self) -> np.ndarray:
```

Picks a parent by tournament. With probability `crossover_rate`, picks a second parent and builds the child with `np.where(mask, parent, other)` where `mask` is a random 50/50 per gene. Otherwise copies the parent. Then draws a Boolean `mutate` mask with probability `mutation_rate` per gene and adds `mutate * normal(0, mutation_scale)`. Returns a new array; parents are never modified.

#### `_learner_ids()`

```python
def _learner_ids(self) -> list[int]:
```

Which tribute slots the champion takes in validation games: `count = max(1, num_players // 4)` slots, spread as `int(i * num_players / count)`. With 24 players that is `[0, 4, 8, 12, 16, 20]`. Spreading them means the learners are not all neighbours on the starting podiums. The dashboard's "live" feed mode uses the same slots when it lets the newest champion play.

#### `validate(genome)`

```python
def validate(self, genome: np.ndarray) -> float:
```

Returns `0.0` immediately if `validation_games <= 0`. Otherwise builds one job per validation game with seed `validation_seed + i`, runs them (through a pool when `workers > 1` and there is more than one game), scores every learner row with `fitness_of`, and returns the mean.

#### `step_generation(on_progress=None)`

```python
def step_generation(self, on_progress: Callable[[int, int], None] | None = None) -> GenerationStats:
```

One full generation:

1. Start the clocks (`_started` on the first call, `started` every call).
2. `fitness = self.evaluate(on_progress)`.
3. `ranking = np.argsort(fitness)[::-1]`, best first. The champion is a copy of `population[ranking[0]]`.
4. `val_fitness = self.validate(champion)`.
5. Merge this generation's telemetry with `BehaviorTelemetry.merge(self._last_telemetry)`, or `{}` if none.
6. Build `GenerationStats` (best, mean, worst, champion, seconds, val_fitness, cumulative seconds, telemetry, and `self._last_showcase` as the showcase) and append it to `history`.
7. Keep `elite_count = max(1, int(elite_fraction * population_size))` elites as copies, fill the rest with `_child()`, replace the population, and increment `generation`.

Returns the stats. Note the order: the population is replaced *after* the stats are recorded, so `history[-1].champion` is the best genome of the population that was just scored, and it survives into the next population as elite number one.

#### `run(on_generation=None, on_progress=None)`

```python
def run(self, on_generation: Callable[[GenerationStats], None] | None = None, on_progress: Callable[[int, int], None] | None = None) -> list[GenerationStats]:
```

Clears `_stop`, then loops `step_generation` while `generation < training.generations` and not stopped, calling `on_generation(stats)` after each. Returns `self.history`. Calling `run()` again continues from where it left off, because `generation` is not reset.

#### `stop()`

```python
def stop(self) -> None:
```

Sets `_stop`. A running `run()` finishes the current generation and returns. The dashboard calls this from the UI thread while `run()` executes in a background thread.

#### `history_rows()`

```python
def history_rows(self) -> list[dict]:
```

`[stats.to_row() for stats in self.history]`. This is what `history.json` and the dashboard's training table contain. No genomes, telemetry or recordings.

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

The champion of the generation with the highest `best_fitness`, or `None` before any generation has run. It is chosen by *training* fitness, not validation fitness.

#### `champion_brain(chaos=None)`

```python
def champion_brain(self, chaos: float | None = None) -> Brain:
```

A fresh brain of `training.brain_name` with the champion genome loaded. `chaos` defaults to `config.chaos`.

#### `save_champion(path)`

```python
def save_champion(self, path: str | Path) -> None:
```

Raises `ValueError("No generations have been run yet")` if there is no champion. Otherwise writes JSON with keys `brain_name`, `neural` (the `NeuralConfig` as a dict), `genome` (a list), `fitness` (the best `best_fitness` in history) and `generations` (how many were run). The other two trainers write the same keys plus a `method` and use `save_champion` as the shared method name.

#### `load_champion(path)` (static)

```python
@staticmethod
def load_champion(path: str | Path) -> dict:
```

Reads a champion file back. Converts `genome` to a float numpy array and `neural` back into a `NeuralConfig` (turning `hidden_layers` from a list to a tuple). Extra keys, such as the `method`, `teacher`, `value_genome` and `epochs` the other trainers write, pass through untouched.

## How to use it / experiment

**A minimal run.**

```python
from hunger_games.config import SimulationConfig
from hunger_games.training import GeneticTrainer, TrainingConfig

config = SimulationConfig(width=80, height=80, max_days=8)
training = TrainingConfig(brain_name="voting", population_size=48, generations=10, seed=0)
trainer = GeneticTrainer(config, training)
for stats in trainer.run():
    pass
for s in trainer.history:
    print(f"gen {s.generation:2d} best {s.best_fitness:.3f} mean {s.mean_fitness:.3f} val {s.val_fitness:.3f} {s.seconds:.1f}s")
trainer.save_champion("champion.json")
```

**Warm-start from an imitation champion.** The recommended way to evolve a neural brain:

```python
from hunger_games.training import ImitationConfig, ImitationTrainer

student = ImitationTrainer(config, ImitationConfig(seed=0))
student.run()
trainer = GeneticTrainer(
    config,
    TrainingConfig(brain_name="neural", mutation_scale=0.02, seed=0),
    initial_genome=student.champion,
)
trainer.run()
```

The relatives in the first population sit at `0.25 * 0.02 = 0.005` from the champion, and children get nudges of scale 0.02. The dashboard's tick box "start from the current champion" does the same and its tooltip suggests the 0.02 scale. Any genome of the right size works, including one from `load_champion("champion.json")["genome"]`.

**Use the cores.** Put the code in a function, guard it, and set `workers`:

```python
def main() -> None:
    trainer = GeneticTrainer(config, TrainingConfig(workers=4, seed=0))
    trainer.run(on_generation=print)

if __name__ == "__main__":
    main()
```

**Watch a generation.** Every `GenerationStats` carries a full recording of its first evaluation game. Save one as a replay, or export it as a GIF:

```python
from hunger_games.renderer import export_recording_gif

trainer.history[-1].showcase.save("gen_last.replay")
export_recording_gif(trainer.history[-1].showcase, "gen_last.gif")
```

The dashboard does this for you: set the training feed to "replay" and it loads each new generation's showcase as soon as the arena is free. See [../ui/session.md](../ui/session.md).

**Read the validation curve, not the training curve.** Plot `val_fitness` per generation. If `best_fitness` climbs but `val_fitness` is flat, the population is only getting better at beating itself.

**Change what is optimised.** Set `kills_weight=0.5` to breed hunters, or `days_weight=0.1, kills_weight=0.0` to breed survivors. Then compare the `deaths_by_cause.png` and `action_distribution_over_training.png` charts in the run folders (see [runs.md](runs.md)).

**Selection pressure.** `tournament_size=2` is gentle and keeps diversity; `tournament_size=6` converges fast and may get stuck. `elite_fraction=0.0` still keeps one elite because of the `max(1, ...)`.

**Play a champion.** `trainer.champion_brain()` gives a brain to hand to a scenario roster, or load `champion.json` in the dashboard with "Load champion into all".

## Gotchas

- **The macOS spawn rule.** On macOS, `multiprocessing` starts workers with `spawn`, which imports your script afresh in each worker. Without `if __name__ == "__main__":` the worker re-runs the training code at import time and you get a recursion of processes or a `RuntimeError` about the bootstrapping phase. It also means the code must live in a file: the interactive prompt and notebooks cannot be re-imported. The dashboard avoids the issue by running the trainer in a thread with `workers=1` by default.
- Everything sent to a worker is pickled: `config`, `scenario`, the genomes. A `Scenario` with a large painted map is copied once per job. Keep `rounds_per_generation` modest if the map is big. With `workers > 1` the first job's `Recording` is pickled back from the worker too, which is one frame per tick of extra transfer per generation.
- **Warm starts with the default mutation scale lose the instincts.** The relatives are gentle (`0.025` with the default scale), but every child after generation 0 gets full `mutation_scale` noise on 10 percent of its genes. A pretrained network's weights are small, so 0.1 noise on 587 weights is a big change. Use a scale around 0.02 for a warm-started neural population.
- **Showcases stay in memory.** Every generation's recording lives in `history` until the trainer is dropped. On a long run with a big map that adds up. Set `record_showcase=False` for runs of hundreds of generations, or for anything where nobody is watching.
- The showcase is always the *first* job of the generation, the first `num_players` genomes of the first round's shuffle. Padding is appended at the end of the order, so it only reaches the first game when the population is smaller than the player count. It is one real evaluation game, not a game of champions; with 48 genomes and 24 players, the champion of the generation is in it only half the time.
- `champion` is picked by training `best_fitness`, which is a coevolution score. The generation with the highest `val_fitness` may be a different one; `max(trainer.history, key=lambda s: s.val_fitness).champion` gives that genome.
- Fitness for a cold-started neural brain starts near 0.5 and often stays there for many generations. The default 20 generations is a smoke test, not a serious run. Expect to need hundreds of generations and a larger population, or start warm.
- `collect_telemetry=True` roughly doubles evaluation time on small maps. Turn it off for fast sweeps of GA settings, but then `save_run` writes no behaviour charts.
- `validation_games=0` makes `val_fitness` always `0.0` and the validation line in `fitness.png` flat at zero.
- `GeneticTrainer.__init__` and `champion_brain` build brains without the `endgame_instinct` argument, so a neural template brain uses the brain's default there. Only the games themselves pass `config.endgame_instinct`.
- A `champion.json` from one `NeuralConfig` will not load into a brain with a different architecture: `set_genome` raises `ValueError` about the size. `load_champion` returns the matching `neural` config so you can build the right brain. The same applies to `initial_genome`: it must fit the template brain, or the constructor raises.
