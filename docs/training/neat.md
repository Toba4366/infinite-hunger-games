# `neat.py`

**Source:** [hunger_games/training/neat.py](../../hunger_games/training/neat.py)
**Depends on:** `json`, `time`, `collections.abc.Callable`, `concurrent.futures.ProcessPoolExecutor`, `dataclasses`, `pathlib` (standard library); `numpy`; [brain/neat.py](../brain/neat.md) (`InnovationTracker`, `NeatConfig`, `NeatGenome`, and `NeatBrain` inside `champion_brain`); [brain/neural.py](../brain/neural.md) (`MENU_SIZE`); [hunger_games/config.py](../config.md) (`SimulationConfig`); [hunger_games/perception.py](../perception.md) (`VECTOR_SIZE`); [hunger_games/recorder.py](../recorder.md) (`Recording`); [research/telemetry.py](../research/telemetry.md) (`BehaviorTelemetry`); [hunger_games/scenario.py](../scenario.md) (`Scenario`); [training/common.py](common.md) (`Curriculum`, `EventLog`, `IterationStats`, `LearnerSpec`, `learner_ids`); [training/reinforce.py](reinforce.md) (`play_rl_episode`)
**Used by:** [training/__init__.py](init.md) (re-exports `NeatTrainer`, `NeatTrainerConfig`); [training/runs.py](runs.md) (`save_run` reads `settings`, `config`, `history`, `history_rows()`, `learning_history`, `events`, `champion`, `save_champion`); [research/comparison.py](../research/comparison.md) (the `"neat"` method); [ui/session.py](../ui/session.md) (`Session.start_training` builds a `NeatTrainer` for the `"neat"` method; NEAT champions are given to tributes as dictionaries); [ui/app.py](../ui/app.md) (`NeatTrainerConfig` backs the NEAT controls); `tests/test_methods.py`

## Purpose

This file evolves NEAT genomes: the shape of the network as well as its weights. The genome, its mutations and its crossover are in [../brain/neat.md](../brain/neat.md). This file is the population loop around them.

The population starts as minimal genomes, inputs wired straight to the outputs. Each generation every genome plays as the learner against voting opponents and is scored by its episode return, the same score every other trainer uses. Genomes are grouped into species by structural similarity. Fitness is shared within a species, so a big species cannot crowd everyone out. Each species is given offspring in proportion to its share of the total adjusted fitness, and offspring come from crossover and mutation. Species that stop improving for too long are removed. This follows the original NEAT paper and the Monopoly video's use of it.

The trainer also reports game-level win rates. A game is won when any of the genome's copies was the victor. The champion's validation returns its score and its win rate, and the curriculum is promoted on validation wins.

**How it relates to the Monopoly video.** The video scored each NEAT genome by playing it in games against other members of the population, a tournament fitness. Here fitness is the episode return against fixed voting opponents. That makes fitness absolute rather than relative: a score of 4.0 in generation 2 means the same as 4.0 in generation 20, and the curve can be compared with the other four trainers. The genetic trainer's `opponents="self"` mode is the closest thing to the video's tournament in this project.

## Concepts you need

**Species.** A group of genomes whose compatibility distance to the species' representative is below a threshold. Species protect new structure: a genome that just grew a node is usually worse for a while, and inside its own small species it only competes with similar genomes.

**Fitness sharing.** Each genome's fitness is divided by the size of its species. A species of 20 has to be very good to earn as many offspring as a species of 2.

**Stagnation.** A species whose best fitness has not improved for a number of generations is removed, so a dead end does not hold population slots forever.

**Elitism.** The best genome of a large enough species is copied unchanged into the next generation.

**Adaptive threshold.** The compatibility threshold moves each generation: down when there are fewer species than the target, up when there are more. This keeps the species count near `target_species` without hand tuning.

**Episode return.** The learner's total reward in one game under `RewardConfig`, computed by `play_rl_episode` in [reinforce.md](reinforce.md). The same game and the same reward the RL trainers use, with the NEAT genome in the learner slots.

**Game-level win.** `play_rl_episode` returns `learner_won`, true when any learner copy won the game. Every win rate in this file is a fraction of games with that flag.

## Walkthrough

### `NeatTrainerConfig`

```python
@dataclass
class NeatTrainerConfig:
```

| Field | Default | Meaning |
| --- | --- | --- |
| `population_size` | `48` | Genomes per generation |
| `generations` | `30` | Generations `run()` plays |
| `rounds_per_generation` | `1` | Games each genome plays per generation |
| `learners_per_game` | `6` | Learner copies per game (the same genome in several slots gives a steadier score) |
| `survival_threshold` | `0.3` | Fraction of each species allowed to reproduce |
| `stagnation` | `15` | Generations without improvement before a species is removed |
| `target_species` | `8` | The number of species aimed for; the compatibility threshold adjusts to reach it |
| `crossover_rate` | `0.75` | Chance a child comes from crossover rather than a mutated clone |
| `elite_species_size` | `5` | Champions of species at least this big are copied unchanged |
| `validation_games` | `2` | Greedy validation games on fixed seeds for the champion |
| `validation_seed` | `90000` | The first validation seed (game `i` uses `validation_seed + i`) |
| `workers` | `1` | CPU cores for the episode jobs |
| `seed` | `None` | Seed for the trainer's own randomness |
| `record_showcase` | `True` | Record one game per generation for the training feed |
| `neat` | `NeatConfig()` | Mutation and speciation settings (see [../brain/neat.md](../brain/neat.md)) |

Design reasoning: with 48 genomes and 1 round, a generation is 48 games, each with 6 copies of one genome against 18 voting opponents. That is twelve times the games of a REINFORCE epoch, which is why NEAT is the slowest method per iteration even now that each decision is a compiled forward pass.

### `Species`

```python
@dataclass
class Species:
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `id` | `int` | required | Id, from the trainer's `next_species_id` counter |
| `representative` | `NeatGenome` | required | The genome new members are compared against |
| `members` | `list[NeatGenome]` | `[]` | Members this generation |
| `best_fitness` | `float` | `-inf` | Best fitness ever seen in this species |
| `stale` | `int` | `0` | Generations since the best improved |

### `_neat_job(args)`

```python
def _neat_job(args: tuple) -> dict:
```

Unpacks a job tuple for the process pool and calls `play_rl_episode(*args)`. The tuple is `(config, scenario, LearnerSpec("neat", genome_dict), learner_ids, seed, greedy, record)`.

### `NeatTrainer`

```python
class NeatTrainer:
    method = "neat"
```

Evolves NEAT genomes against voting opponents.

#### `__init__(config, neat, scenario=None, initial_genome=None, curriculum=None)`

```python
def __init__(self, config: SimulationConfig, neat: NeatTrainerConfig, scenario: Scenario | None = None, initial_genome: dict | None = None, curriculum: Curriculum | None = None) -> None:
```

Stores the settings, seeds `self.rng` from `neat.seed`, and makes an `InnovationTracker`. The population is built one of two ways:

- **`initial_genome is None`:** `population_size` minimal genomes from `NeatGenome.minimal(VECTOR_SIZE, MENU_SIZE, ...)`, all sharing the tracker, so they have identical structure and innovation numbers and differ only in weights.
- **`initial_genome` given (a warm start):** the dictionary is rebuilt with `from_dict`, the tracker's counters are set past the genome's highest innovation number and node id, and the population is `population_size` copies. The first copy is exact; every other copy is mutated once.

Then `species = []`, `next_species_id = 0`, `compatibility_threshold = neat.neat.compatibility_threshold`, an empty `history` (and `learning_history`, which is the same list object), an `EventLog`, `generation = 0`, the stop flag, the start time, `best = None`, `best_species_id = None`, `_last_wins = []` (the game-level win flags of the last evaluation) and `best_mean_score = -inf`.

Only a NEAT dictionary fits `initial_genome`. The dashboard drops a neural warm start when the method is NEAT, and the method comparison only passes a champion of the same kind.

#### `settings` (property)

```python
@property
def settings(self):
```

Returns `self.neat`, for `save_run`.

#### `_learner_ids()`

```python
def _learner_ids(self) -> list[int]:
```

`learner_ids(config.num_players, neat.learners_per_game)`; with 6 of 24 that is `[0, 4, 8, 12, 16, 20]`.

#### `_apply_curriculum()`

```python
def _apply_curriculum(self) -> None:
```

Without a curriculum, nothing. With one, `self.config = stage_config(base, curriculum.stage_spec, learners_per_game)`: the learner copies plus the lesson's opponents, with the lesson's rule overrides applied to the config the curriculum started from (`_stage_base`).

#### `_episode_config(seed)`

`episode_config(self.config, curriculum.stage_spec, seed)`: evaluation games of a lesson with `variants` each get one randomly chosen rule set; validation keeps `self.config`.

#### `_play(jobs)`

```python
def _play(self, jobs: list[tuple]) -> list[dict]:
```

Runs episode jobs through a `ProcessPoolExecutor` when `workers > 1` and there is more than one job, else in sequence. Results come back in job order.

#### `evaluate(on_progress=None)`

```python
def evaluate(self, on_progress: Callable[[int, int], None] | None = None) -> tuple[list[dict], Recording | None]:
```

Scores every genome by its mean episode return against voting opponents. For every genome and every round it builds a job with a random seed, `greedy=True`, and `record` only for the first genome's first round when `record_showcase` is on. Then `_last_wins` is emptied and the jobs are played. For each result: the outcomes (one per learner slot) are averaged into the owning genome's total, the telemetry summary is collected, `int(result["learner_won"])` is appended to `_last_wins`, the recording is kept as the showcase, and `on_progress(done, total)` is called. Finally `genome.fitness = total / count` for every genome. Returns `(telemetry_list, showcase)`.

`_last_wins` has one entry per job, so with the defaults it is 48 flags, one per game of the generation.

Note `greedy=True`: NEAT genomes play with chaos 0 and take the argmax. There is no exploration noise in the episodes; the variety comes from mutation.

#### `validate(genome)`

```python
def validate(self, genome: NeatGenome) -> tuple[float, float]:
```

Mean return and game-level win rate of a genome on the fixed validation seeds `validation_seed + i`, greedy, never recorded. The return is averaged over every learner slot of every game; the win rate is the mean of `learner_won` over the games. Returns `(0.0, 0.0)` when `validation_games <= 0`.

#### `speciate()`

```python
def speciate(self) -> None:
```

Assigns every genome to a species by distance to the species' representative:

1. Empty every existing species' `members` (the species objects and their representatives survive from the previous generation).
2. For each genome, walk the species in order and join the first whose representative is closer than `compatibility_threshold`. If none fits, found a new `Species` with a copy of the genome as representative, log an `"evolution"` event (`new species N founded (H hidden nodes)`), and advance `next_species_id`.
3. Drop species with no members and log how many died out.
4. Nudge the threshold toward the target: if there are fewer species than `target_species`, subtract `0.3` (never below `0.5`); if more, add `0.3`.

With the default threshold of 3.0 and minimal genomes at distance about 0.13, the first generation is one species. The threshold then falls by 0.3 per generation, reaching 0.5 after nine generations, and species appear as weights and structure diverge.

#### `reproduce()`

```python
def reproduce(self) -> None:
```

Builds the next population from the species:

1. **Stagnation.** For each species, if its best member's fitness beats `best_fitness`, update it and reset `stale`; otherwise add one to `stale`.
2. **Removal.** Keep species with `stale < stagnation`, or whose id is `best_species_id` (the champion's species). Log how many were removed. If nothing survives, keep the first species.
3. **Adjusted fitness.** `floor` is the lowest fitness in any species. Each member's `adjusted = (fitness - floor + 1e-3) / len(species.members)`. The shift makes every value positive so shares work with negative returns; the division is fitness sharing. Species totals are summed into `grand`.
4. **Offspring per species.** `tracker.reset_generation()`, then for each species: `share = round(population_size * total / grand)`; members sorted best first. If the species has at least `elite_species_size` members and `share > 0`, the best member is copied unchanged and `share` drops by one. Parents are the top `ceil(len(members) * survival_threshold)`, at least one. For each remaining share: with more than one parent and probability `crossover_rate`, pick two parents at random, order them fitter first, and `fitter.crossover(other, rng)`; otherwise copy a random parent. Every child is mutated. The species' representative becomes a copy of its best member.
5. **Fill or trim.** While there are fewer children than `population_size`, add a mutated copy of the best genome in the whole population. Then cut to `population_size`.

The `adjusted` attribute is set on the genome objects on the fly; it is not a field of `NeatGenome`.

#### `step(on_progress=None)`

```python
def step(self, on_progress: Callable[[int, int], None] | None = None) -> IterationStats:
```

One generation, in this order:

1. Start the clocks.
2. `_apply_curriculum()`.
3. `telemetry, showcase = self.evaluate(on_progress)`.
4. The generation's champion is the fittest genome, and `val_score, val_win_rate = self.validate(champion)` scores it on the fixed seeds.
5. `key = champion_key(stage, val_win_rate, val_score)` ([common.md](common.md)). If there is no best yet or `key` beats `best_key`, store a copy as `self.best` with the key, and log a `"record"` event with its fitness, hidden node count and connection count. The overall champion is therefore the best genome at the highest curriculum stage reached, by validation wins, not the highest training fitness.
6. Merge the telemetry with `BehaviorTelemetry.merge`.
7. `scores` is every genome's fitness; `mean_score` their mean.
8. `speciate()`, remember `best_species_id` when this generation's champion is the overall champion (`key >= best_key`), log an `"evolution"` event (`generation G: S species, best B, mean M`), then `reproduce()`.
9. Build the `IterationStats` with `iteration=generation`, the scores, `entropy` and `mean_length` (`mean_survival_ticks`) from the merged telemetry, `win_rate` = the mean of `_last_wins` (game-level, over every evaluation game), `val_score`, `val_win_rate`, timings, the curriculum's stage and opponents (or `num_players - 1`), `extra={"species", "hidden_nodes", "connections", "threshold"}`, `learner=champion.to_dict()`, the telemetry and the showcase. Append it to `history`.
10. Update `best_mean_score`. Then the curriculum: `judged_win` is `val_win_rate` when `validation_games > 0`, else `stats.win_rate`; call `curriculum.observe(mean_score, judged_win)` and log a `"curriculum"` event on promotion.
11. `generation += 1` and return the stats.

The population is replaced in step 8, so `scores` and `learner` describe the population that was just scored, and the champion survives as an elite only if its species is large enough.

#### `run(on_iteration=None, on_progress=None)`

```python
def run(self, on_iteration: Callable[[IterationStats], None] | None = None, on_progress: Callable[[int, int], None] | None = None) -> list[IterationStats]:
```

Clears the stop flag, loops `step` while `generation < neat.generations` and not stopped, calling `on_iteration(stats)` after each. Returns `history`. A second `run()` continues.

#### `stop()`

```python
def stop(self) -> None:
```

Sets the flag; `run()` returns after the current generation.

#### `champion` (property)

```python
@property
def champion(self) -> dict | None:
```

The best genome so far as a `to_dict()` dictionary, or `None` before the first generation. Unlike the other trainers, the champion is a dictionary, not an array.

#### `champion_spec()` and `learner_spec()`

```python
def champion_spec(self) -> LearnerSpec:
def learner_spec(self) -> LearnerSpec:
```

Both return `LearnerSpec("neat", self.champion)`. `learner_spec` is the same as `champion_spec` here, because a population has no single "current" learner.

#### `champion_brain(chaos=0.0)`

```python
def champion_brain(self, chaos: float = 0.0):
```

A `NeatBrain` wrapping a copy of `best`, or of `population[0]` if nothing has been evaluated.

#### `save_champion(path)`

```python
def save_champion(self, path: str | Path) -> None:
```

Writes JSON in the shared champion file shape:

| Key | Value |
| --- | --- |
| `brain_name` | `"neat"` |
| `genome` | The champion dictionary (`nodes`, `connections`, `fitness`) |
| `fitness` | `best.fitness`, or `0.0` |
| `generations` | `len(history)` |
| `method` | `"neat"` |

There is no `neural` key. `GeneticTrainer.load_champion` leaves a dictionary genome as it is, and the dashboard's "Load champion into all" gives it to tributes as `brain_name="neat"`.

#### `history_rows()`

```python
def history_rows(self) -> list[dict]:
```

`[stats.to_row() for stats in self.history]`. Because `history` is the `IterationStats` list, the rows have `iteration`, `mean_score`, `best_score`, `win_rate`, `val_score`, `val_win_rate` and `extra_species`, `extra_hidden_nodes`, `extra_connections`, `extra_threshold`. `training_run_plots` reads exactly these for its `"neat"` branch.

## How to use it / experiment

**A minimal run.**

```python
from hunger_games.config import SimulationConfig
from hunger_games.training import NeatTrainer, NeatTrainerConfig

config = SimulationConfig(width=80, height=80, max_days=8)
trainer = NeatTrainer(config, NeatTrainerConfig(population_size=24, generations=10, seed=0))
trainer.run(on_iteration=lambda s: print(s.iteration, round(s.best_score, 2), round(s.val_win_rate, 2), s.extra["species"], s.extra["hidden_nodes"]))
trainer.save_champion("neat_champion.json")
```

**Grow faster.** `NeatTrainerConfig(neat=NeatConfig(add_node_rate=0.1, add_connection_rate=0.2))` adds structure sooner. Watch `neat_structure.png` in the run folder: hidden nodes should climb in steps as species with new nodes take over.

**Keep more species.** `target_species=12` drives the threshold lower. Fewer members per species means weaker fitness sharing per genome and more elites (species must reach `elite_species_size`).

**Warm start from a saved NEAT champion.**

```python
from hunger_games.training import GeneticTrainer
data = GeneticTrainer.load_champion("neat_champion.json")
trainer = NeatTrainer(config, NeatTrainerConfig(seed=1), initial_genome=data["genome"])
```

**With the curriculum.** `NeatTrainer(config, settings, curriculum=Curriculum(CurriculumConfig()))`. Every generation's games are sized to the stage, and `curriculum.png` shows the ladder. The champion moves up once it has won at least half of its validation games over the last five generations; there is no timeout by default.

**Watch the champion.** `trainer.champion_brain()` gives a `NeatBrain`; the dashboard's Network tab draws the champion as a graph, columns by depth.

**Use the cores.** `workers=4` parallelises the 48 games of a generation. A `__main__` guard is needed on macOS; see [genetic.md](genetic.md).

## Gotchas

- **Still slow per generation.** A default generation is 48 games. The forward pass is now compiled (about 30 microseconds per decision, see [../brain/neat.md](../brain/neat.md)), so the cost is the games themselves, not the network. Expect a generation to take many times longer than a REINFORCE epoch on the same config. Lower `population_size` or the map size for experiments.
- `validate` scores the generation's champion, while `champion` and `champion_brain` give the best genome ever. After a bad generation the two differ, and so does the win rate the curriculum sees.
- `win_rate` on the record is over every evaluation game of the population, not the champion's games. It says how often any genome's copies won. `val_win_rate` is the champion's.
- With `validation_games=2` the validation win rate per generation is 0, 0.5 or 1. The curriculum averages five generations, so one lucky pair cannot promote on its own, but the signal is coarse. Raise `validation_games` for a steadier ladder.
- Stagnation never removes the champion's species: `step()` records `best_species_id` after speciation, and `reproduce()` keeps that species alive however stale it is.
- `evaluate` plays greedily (chaos 0). A genome's fitness is deterministic for a given seed, so `rounds_per_generation=1` scores each genome on one game. Raise it, or `learners_per_game`, for a steadier score.
- The first generation is a single species with 48 members, so the elite rule fires once and 47 children are bred from the top `ceil(48 * 0.3) = 15` parents.
- `IterationStats.learner` is a dictionary. Code that expects a flat array (the dashboard's gene history reads the connection weights out of it) must check the type.
- `history` and `learning_history` are the same list, so `save_run` writes the same rows to `history.json` and `learning.json`.
- `save_run(trainer, "neat", ...)` is the only method name whose x axis is `iteration` in `training_run_plots`; pass another name and the plots raise `KeyError`.
- Champion files have no `neural` key. Code that reads `data["neural"]` without a `.get` fails on a NEAT champion.
- `workers > 1` pickles a `LearnerSpec` holding the genome dictionary for every job, once per genome per round. Large genomes make that noticeable. Each worker also recompiles the genome's evaluation plan, because the plan is a cache that is not part of `to_dict()`.
- `initial_genome` must be a dictionary from `to_dict()`. A flat array raises inside `from_dict`.
