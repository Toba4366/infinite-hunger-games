# `comparison.py`

**Source:** [hunger_games/research/comparison.py](../../hunger_games/research/comparison.py)
**Depends on:** `copy`, `json`, `time`, `warnings`, `dataclasses`, `pathlib`, `typing` (standard library); `numpy`; `pandas`; [config.py](../config.md) (`SimulationConfig`); [research/plots.py](plots.md) (`overlay_curves`, `bars`); [research/experiments.py](experiments.md) (`make_run_dir`); [training/common.py](../training/common.md) (`Curriculum`, `CurriculumConfig`, `LearnerSpec`, `learner_ids`); [training/genetic.py](../training/genetic.md) (`GeneticTrainer`, `TrainingConfig`); [training/imitation.py](../training/imitation.md) (`ImitationConfig`, `ImitationTrainer`); [training/neat.py](../training/neat.md) (`NeatTrainer`, `NeatTrainerConfig`); [training/ppo.py](../training/ppo.md) (`PPOConfig`, `PPOTrainer`); [training/reinforce.py](../training/reinforce.md) (`ReinforceTrainer`, `RLConfig`, `_run_episode_job`); [training/runs.py](../training/runs.md) (`save_run`)
**Used by:** [experiments/run_comparison.py](../experiments/run_comparison.md); `tests/test_comparison.py` ([test_comparison.md](../tests/test_comparison.md))

## Purpose

The project exists to answer one question: which way of training a brain makes the most sense for the Hunger Games? This module is the experiment that answers it. It takes a list of *variants* (a method plus its settings, and optionally a network size or an initializer to compare), trains each one for the same number of iterations or the same wall-clock budget, keeps the shared learning curves, and then runs a *tournament*: every champion plays the same seeded games as the learner against voting opponents. It writes one run folder with a results table (CSV and LaTeX), one PNG per chart, a `summary.json`, and a generated `report.md` that ranks the methods and explains the trade-offs.

Nothing here knows how a method learns. That is the point. Every trainer exposes the same `step()`, `learning_history`, `champion_spec()` and `settings`, so this module can treat imitation, a genetic algorithm, NEAT, REINFORCE and PPO alike.

## Concepts you need

**Method versus variant.** A method is one of the five trainers. A variant is one thing to compare: a method, its settings, and any tweak to the simulation it trains in. Comparing `ppo` with a 16-unit hidden layer against `ppo` with 64 by 32 is two variants of one method.

**Budget.** Every variant gets the same `iterations` (epochs for imitation, REINFORCE and PPO; generations for the genetic algorithm and NEAT) and, optionally, the same `time_budget` in seconds. That is what makes the learning curves comparable.

**Champion.** Each trainer keeps its best learner: the lowest validation loss for imitation, the best validation return for REINFORCE and PPO, the best fitness for the genetic algorithm and NEAT. `champion_spec()` returns it as a `LearnerSpec` (its kind, its genome, and its neural architecture) that a worker process can rebuild.

**Seeded games.** `play_rl_episode` builds a config copy with a given seed, so two champions given the same seed play the same arena, the same roster and the same dice. The tournament uses seeds `50000 + i` for game `i`, for every champion.

**Dotted overrides.** `{"neural.hidden_layers": (16,)}` means "set `config.neural.hidden_layers` to `(16,)`". `set_overrides` walks the dots.

**Warm start.** A trainer's `initial_genome` starts it from an existing learner. Here the genome comes from an earlier variant's champion, named by `warm_from`.

**Curriculum.** `Curriculum(CurriculumConfig())` grows the number of voting opponents from 1 to 23 as the learner improves. See [../training/common.md](../training/common.md).

**Lines of code.** A rough measure of how much there is to implement. `count_lines` sums the lines of the files a method needs.

## Walkthrough

### Imports

Standard library first, then `numpy` and `pandas`, then the project: `SimulationConfig`, the `plots` module and `make_run_dir`, the shared training pieces, the five trainer classes with their settings classes, `_run_episode_job` (the tournament's game player), and `save_run`.

### `METHODS`

```python
METHODS: dict[str, tuple[type, Callable[[], Any]]]
```

The trainer class and the default settings factory for every method name.

| Name | Trainer | Default settings |
| --- | --- | --- |
| `"imitation"` | `ImitationTrainer` | `ImitationConfig` |
| `"genetic"` | `GeneticTrainer` | `TrainingConfig` |
| `"neat"` | `NeatTrainer` | `NeatTrainerConfig` |
| `"reinforce"` | `ReinforceTrainer` | `RLConfig` |
| `"ppo"` | `PPOTrainer` | `PPOConfig` |

Any other method name raises `KeyError` in `_build`.

### `METHOD_NOTES`

One line per method, printed in the report's "Why the methods differ" section.

| Method | Note |
| --- | --- |
| imitation | Supervised learning: copies the voting brain. Needs a teacher; cannot exceed it. |
| genetic | Evolves weights only. No gradients; simple; scales poorly with weight count. |
| neat | Evolves weights and structure with species. More machinery; small networks; slow per generation. |
| reinforce | Policy gradient with a value baseline. One pass per batch; high variance. |
| ppo | Clipped policy gradient with GAE and several passes per batch. Most stable of the reward methods. |

### `Variant`

```python
@dataclass
class Variant:
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `name` | `str` | required | Label in tables, plots and the `runs/` folder |
| `method` | `str` | required | A key of `METHODS` |
| `settings` | `Any` | `None` | The method's settings dataclass; `None` means the defaults |
| `config_overrides` | `dict` | `{}` | Dotted overrides on the simulation config, e.g. `{"neural.hidden_layers": (16,)}` |
| `curriculum` | `bool` | `False` | Train with the opponent curriculum |
| `warm_from` | `str \| None` | `None` | Name of an earlier variant whose champion seeds this one |

### `ComparisonConfig`

```python
@dataclass
class ComparisonConfig:
```

| Field | Default | Meaning |
| --- | --- | --- |
| `name` | `"comparison"` | Run folder label |
| `iterations` | `20` | Iterations per variant (epochs or generations) |
| `time_budget` | `None` | Optional seconds per variant; training stops early once reached |
| `tournament_games` | `75` | Games each champion plays in the tournament |
| `tournament_learners` | `6` | Learner copies per tournament game |
| `workers` | `1` | CPU workers for the trainers and the tournament |
| `seed` | `0` | Seed for the simulation config and every trainer |
| `results_dir` | `"results"` | Where run folders go |

### `set_overrides(config: SimulationConfig, overrides: dict) -> SimulationConfig`

Deep-copies the config, then for each `"a.b.c": value` walks `getattr` through `a` and `b` and calls `setattr` on the last part. The original is untouched. A key with no dot sets a top-level field, e.g. `{"chaos": 0.0}`.

### `count_lines(method: str) -> int`

Sums `len(read_text().splitlines())` over the files a method needs, relative to `hunger_games/training/`.

| Method | Files counted | Lines at the time of writing |
| --- | --- | --- |
| imitation | `imitation.py` | 523 |
| genetic | `genetic.py` | 748 |
| neat | `neat.py`, `../brain/neat.py` | 481 + 464 = 945 |
| reinforce | `reinforce.py` | 745 |
| ppo | `reinforce.py`, `ppo.py` | 745 + 141 = 886 |

PPO inherits from REINFORCE, so it counts both files. NEAT needs its own genome module, so it counts that too. The numbers change whenever the files do; the table above is a snapshot.

### `MethodComparison`

```python
class MethodComparison:
    def __init__(self, base_config: SimulationConfig, comparison: ComparisonConfig, variants: list[Variant]) -> None:
```

Stores the three inputs and empty result stores:

| Attribute | Filled by | Contents |
| --- | --- | --- |
| `learning: dict[str, list[dict]]` | `train_all` | Per variant, one `IterationStats.to_row()` per iteration |
| `champions: dict[str, LearnerSpec]` | `train_all` | Per variant, the champion |
| `train_seconds: dict[str, float]` | `train_all` | Wall-clock seconds of training |
| `tournament: dict[str, dict]` | `run_tournament` | Per variant, the tournament numbers |
| `run_dir: Path \| None` | `run` | The run folder (`None` until `run` is called) |
| `_stop` | `stop` | Stop flag |

#### `_build(self, variant: Variant) -> Any`

Constructs the trainer for one variant.

1. Looks up the trainer class and default settings in `METHODS`. Uses `variant.settings` if given, else the defaults.
2. Copies the shared knobs onto the settings: `workers` and `seed` from `ComparisonConfig`, but only if the settings object has that attribute (all five do).
3. Applies `variant.config_overrides` to a copy of `base_config`, then sets `config.seed = comparison.seed`.
4. Builds `Curriculum(CurriculumConfig())` if `variant.curriculum` is true, else `None`.
5. Warm start: if `warm_from` names a variant that has already trained, takes its champion spec. The genome is passed only when the kinds match: a NEAT champion goes to a NEAT trainer, a neural champion goes to a non-NEAT trainer. Otherwise `initial_genome` is `None` and the trainer starts fresh.
6. Returns `trainer_class(config, settings, initial_genome=initial, curriculum=curriculum)`.

#### `train_all(self, on_progress: Callable[[str, int], None] | None = None) -> None`

Trains every variant in list order, so a warm start can use an earlier champion. For each variant: build the trainer, note the start time, then call `trainer.step()` up to `iterations` times. After every step it calls `on_progress(variant.name, iteration + 1)` and stops early if `time_budget` is set and has elapsed. Then it records `train_seconds`, `learning` (every `IterationStats` as a row) and the champion. If `run_dir` is set, `save_run(trainer, variant.method, variant.name, run_dir / "runs")` writes the variant's own run folder. The `_stop` flag is checked before each variant and each step.

#### `run_tournament(self, on_progress: Callable[[str, int, int], None] | None = None) -> None`

Every champion plays the same seeded games as the learner against voting opponents.

- The game config is `base_config` with `seed = comparison.seed`. No variant overrides apply: everyone fights on the same arena settings. A champion still plays with its own architecture, because its `LearnerSpec` carries its `NeuralConfig` (or its NEAT genome).
- `learners = learner_ids(config.num_players, tournament_learners)`: with 24 players and 6 learners, slots 0, 4, 8, 12, 16, 20.
- One job per game: `(config, None, spec, learners, 50000 + i, True, False)`. That is no scenario, seed `50000 + i`, greedy play (chaos 0, argmax), no recording. The seeds are the same for every champion.
- Runs the jobs through `_run_episode_job`, with a `ProcessPoolExecutor` when `workers > 1`.
- Flattens every learner outcome of every game and stores, per variant:

| Key | Value |
| --- | --- |
| `mean_score` | Mean episode return under `RewardConfig` |
| `win_rate` | Fraction of learner episodes that won |
| `mean_survival` | Mean ticks survived |
| `mean_kills` | Mean kills per learner episode |
| `games` | Number of games played |

With the defaults, each mean is over 75 games times 6 learners, 450 learner episodes. `on_progress(name, done, total)` is called after each champion.

#### `run(self, on_progress: Callable[[str, str], None] | None = None) -> Path`

Train, fight, write. Clears the stop flag, makes the run folder with `make_run_dir(results_dir, name)`, writes `config.json` (the base config, the comparison settings, and each variant with its settings as a dictionary), then calls `train_all`, `run_tournament` and `write`. The progress callback receives `(name, "iteration 3")` during training and `(name, "tournament 2/5")` during the tournament. Returns the run folder.

#### `stop(self) -> None`

Sets `_stop`, so training stops after the current step and the tournament after the current champion.

#### `table(self) -> pd.DataFrame`

One row per variant, in variant order:

| Column | Source |
| --- | --- |
| `variant` | `Variant.name` |
| `method` | `Variant.method` |
| `iterations` | Number of learning rows actually completed |
| `train_seconds` | Training time, rounded to one decimal |
| `final_mean_score` | `mean_score` of the last iteration |
| `best_val_score` | The highest `val_score` over the run |
| `tournament_score` | `mean_score` from the tournament |
| `tournament_win_rate` | `win_rate` from the tournament |
| `tournament_survival` | `mean_survival` from the tournament |
| `tournament_kills` | `mean_kills` from the tournament |
| `lines_of_code` | `count_lines(method)` |

A variant that has not trained or fought gets `NaN` in the missing columns.

#### `write(self) -> None`

Does nothing without a `run_dir`. Otherwise writes:

| File | Contents |
| --- | --- |
| `results.csv` | The table |
| `summary.json` | `{"table": rows, "tournament": self.tournament, "learning": self.learning}` |
| `results_table.tex` | `table.to_latex(index=False, float_format="%.2f")`, with pandas' `FutureWarning` silenced |
| `plots/score_by_method.png` | `mean_score` against `iteration`, one line per variant |
| `plots/score_by_time.png` | `mean_score` against `cumulative_seconds` |
| `plots/validation_by_method.png` | `val_score` against `iteration` |
| `plots/entropy_by_method.png` | `entropy` (nats) against `iteration` |
| `plots/length_by_method.png` | `mean_length` (ticks) against `iteration` |
| `plots/tournament_mean_score.png`, `tournament_win_rate.png`, `tournament_mean_survival.png`, `tournament_mean_kills.png` | Bar charts, only if the tournament ran |
| `plots/lines_of_code.png` | Bar chart of `lines_of_code` |
| `plots/train_seconds.png` | Bar chart of `train_seconds` |
| `report.md` | `self.report(table)` |

The overlay charts come from `plots.overlay_curves`, the bars from `plots.bars`.

#### `report(self, table: pd.DataFrame) -> str`

A generated Markdown write-up:

1. `# Method comparison: <name>` and one sentence stating the budget (`iterations`, the time budget if any, and `tournament_games`).
2. If the tournament ran, `## Ranking by tournament score`: a table sorted by `tournament_score` descending with columns rank, variant, method, tournament score, win rate, survival, train seconds, lines of code. Then three bold lines: **Best in the tournament** (top row), **Simplest to implement** (the method with the fewest lines), **Fastest to train under this budget** (the variant with the fewest `train_seconds`).
3. `## Why the methods differ`: one bullet per distinct method with its `METHOD_NOTES` line.
4. `## Charts`: the list of PNG names and a note that each variant's own run folder is under `runs/`.

## How to use it / experiment

**All five methods, defaults.**

```python
from hunger_games.config import SimulationConfig
from hunger_games.research.comparison import ComparisonConfig, MethodComparison, Variant

config = SimulationConfig(width=120, height=120)
variants = [Variant(m, m) for m in ("imitation", "genetic", "neat", "reinforce", "ppo")]
folder = MethodComparison(config, ComparisonConfig(iterations=20, workers=4), variants).run(print)
```

**Network sizes.** Use `config_overrides` on `neural.hidden_layers`. The tournament runs on the base config, but each champion keeps its own layer sizes through its `LearnerSpec`.

```python
variants = [
    Variant("ppo_16", "ppo", config_overrides={"neural.hidden_layers": (16,)}),
    Variant("ppo_64x32", "ppo", config_overrides={"neural.hidden_layers": (64, 32)}),
    Variant("ppo_128x64", "ppo", config_overrides={"neural.hidden_layers": (128, 64)}),
]
```

**Initializers.** Same idea with `neural.initializer`, any name from [../brain/initializers.md](../brain/initializers.md).

```python
variants = [Variant(f"ppo_{i}", "ppo", config_overrides={"neural.initializer": i}) for i in ("xavier_uniform", "he_uniform", "zeros")]
```

**Warm starts and the curriculum.** Put the imitation variant first and name it in `warm_from`.

```python
variants = [
    Variant("imitation", "imitation"),
    Variant("ppo_warm", "ppo", warm_from="imitation", curriculum=True),
    Variant("ppo_cold", "ppo", curriculum=True),
]
```

**A time budget instead of an iteration count.** `ComparisonConfig(iterations=1000, time_budget=600)` gives every variant ten minutes; `score_by_time.png` is then the fair chart, because a NEAT generation and an imitation epoch take very different times.

**Custom settings.** Pass the method's settings dataclass: `Variant("ppo_big_batch", "ppo", PPOConfig(episodes_per_epoch=8, update_epochs=8))`. `workers` and `seed` on it are overwritten by the comparison's values.

**Reading the result.** Open `report.md` first for the ranking. `results.csv` has every number; `summary.json` adds the full learning curves. `runs/<variant>_<timestamp>/` holds each variant's own charts (`score.png`, `entropy_shared.png`, `game_length.png`, `curriculum.png` and the method's own plots).

## Gotchas

- **Order matters for warm starts.** `warm_from` only works if the named variant appears earlier in the list. If it does not, or the name is misspelled, the trainer silently starts fresh. Check the first iteration's score against the imitation variant's if you doubt it.
- **Kinds must match.** A neural champion cannot seed NEAT and a NEAT champion cannot seed a neural method; those combinations start fresh. A genetic variant with `TrainingConfig(brain_name="voting")` has kind `"voting"`, which the rule treats like neural, so do not warm a neural method from a voting-brain champion.
- **Settings are mutated.** `_build` sets `workers` and `seed` on the settings object you pass in. Make a new object per variant.
- **The time budget is checked after each step.** The last step can overrun it. A NEAT generation with a large population can take minutes.
- **The tournament ignores config overrides.** By design, every champion plays the same base arena, roster size and opponent brain. An override such as `chaos` or `num_players` shapes training only.
- **Imitation's curriculum flag does nothing.** `ImitationTrainer` stores the curriculum but never resizes its roster. `run_comparison.py` only enables the curriculum for the four reward methods.
- **Tournament numbers are means of learner episodes**, not of games. With 6 learners per game a win rate of 1/6 means one learner won every game.
- **NaN rows.** If `stop()` interrupts the run, later variants have no learning rows and no tournament entry; the table shows `NaN` and the report's ranking sorts them last.
- **`count_lines` reads files by path.** Moving or renaming a training module breaks it with `FileNotFoundError`.
