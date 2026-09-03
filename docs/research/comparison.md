# `comparison.py`

**Source:** [hunger_games/research/comparison.py](../../hunger_games/research/comparison.py)
**Depends on:** `copy`, `json`, `time`, `warnings`, `dataclasses`, `pathlib`, `typing` (standard library); `numpy`; `pandas`; [config.py](../config.md) (`SimulationConfig`); [research/plots.py](plots.md) (`overlay_curves`, `bars`); [research/experiments.py](experiments.md) (`make_run_dir`); [training/common.py](../training/common.md) (`Curriculum`, `CurriculumConfig`, `LearnerSpec`, `learner_ids`); [training/genetic.py](../training/genetic.md) (`GeneticTrainer`, `TrainingConfig`); [training/imitation.py](../training/imitation.md) (`ImitationConfig`, `ImitationTrainer`); [training/neat.py](../training/neat.md) (`NeatTrainer`, `NeatTrainerConfig`); [training/ppo.py](../training/ppo.md) (`PPOConfig`, `PPOTrainer`); [training/reinforce.py](../training/reinforce.md) (`ReinforceTrainer`, `RLConfig`, `_run_episode_job`); [training/runs.py](../training/runs.md) (`save_run`)
**Used by:** [experiments/run_comparison.py](../experiments/run_comparison.md); `tests/test_comparison.py` ([test_comparison.md](../tests/test_comparison.md))

## Purpose

The project exists to answer one question: which way of training a brain makes the most sense for the Hunger Games? This module is the experiment that answers it. It takes a list of *variants* (a method plus its settings, and optionally a network size or an initializer to compare), trains each one until it meets a win criterion or runs out of iterations or time, keeps the shared learning curves, and then runs a *tournament*: every champion plays the same seeded games as the learner against voting opponents. It writes one run folder with a results table (CSV and LaTeX), one PNG per chart, a `summary.json`, and a generated `report.md` that ranks the methods, reports how long each took to reach the win criterion, compares warm and cold starts, and explains the trade-offs.

Nothing here knows how a method learns. That is the point. Every trainer exposes the same `step()`, `learning_history`, `champion_spec()`, `curriculum` and `settings`, so this module can treat imitation, a genetic algorithm, NEAT, REINFORCE and PPO alike.

## Concepts you need

**Method versus variant.** A method is one of the five trainers. A variant is one thing to compare: a method, its settings, and any tweak to the simulation it trains in. Comparing `ppo` with a 16-unit hidden layer against `ppo` with 64 by 32 is two variants of one method.

**Budget.** Every variant gets the same `iterations` (epochs for imitation, REINFORCE and PPO; generations for the genetic algorithm and NEAT) and, optionally, the same `time_budget` in seconds. That is what makes the learning curves comparable.

**Win criterion.** A variant can stop before its budget is spent. Once its validation win rate, averaged over the last `win_window` iterations all played at the final curriculum stage (if there is one), reaches `until_win_rate`, training stops and the iteration count and the seconds it took are recorded. This turns "how good after N iterations" into "how long to get good", which is the fairer question when the methods cost different amounts per iteration.

**Game-level win rate.** A learner plays several copies of itself in one game (6 by default), and one game has one victor. A game counts as won when any learner copy was the victor. Every win rate here, in training rows and in the tournament, is a fraction of games, not of learner copies. So the tournament's `win_rate` can reach 1.0, and 0.5 means the learner won half of the 75 games.

**Champion.** Each trainer keeps its best learner: the lowest validation loss for imitation, the best validation return for REINFORCE and PPO, the best fitness for the genetic algorithm and NEAT. `champion_spec()` returns it as a `LearnerSpec` (its kind, its genome, and its neural architecture) that a worker process can rebuild.

**Seeded games.** `play_rl_episode` builds a config copy with a given seed, so two champions given the same seed play the same arena, the same roster and the same dice. The tournament uses seeds `50000 + i` for game `i`, for every champion.

**Dotted overrides.** `{"neural.hidden_layers": (16,)}` means "set `config.neural.hidden_layers` to `(16,)`". `set_overrides` walks the dots.

**Warm start.** A trainer's `initial_genome` starts it from an existing learner. Here the genome comes from an earlier variant's champion, named by `warm_from`.

**Warm versus cold pairs.** Two variants of one method that differ only in the start: `<method>_cold` from random weights and `<method>_warm` from the imitation champion. The report pairs them by name and says which one won more tournament games. `run_comparison.py --pairs` builds them.

**Curriculum.** `Curriculum(CurriculumConfig())` grows the number of voting opponents from 1 to 23 as the learner wins. With the defaults a stage is cleared by winning at least half of the validation games over five iterations, with no timeout. See [../training/common.md](../training/common.md).

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

A name ending in `_cold` with a twin ending in `_warm` makes a pair for the report. Nothing else reads the name.

### `ComparisonConfig`

```python
@dataclass
class ComparisonConfig:
```

| Field | Default | Meaning |
| --- | --- | --- |
| `name` | `"comparison"` | Run folder label |
| `iterations` | `20` | Iterations per variant (epochs or generations), the most a variant will train |
| `time_budget` | `None` | Optional seconds per variant; training stops early once reached |
| `until_win_rate` | `0.5` | Stop a variant early once its validation win rate over `win_window` iterations reaches this; `None` never stops early |
| `win_window` | `5` | Iterations averaged for the criterion |
| `extended_iterations` | `0` | After every variant has had `iterations`, those still short of the criterion keep training for up to this many more iterations; `0` never extends |
| `extended_time_budget` | `None` | Wall-clock cap in seconds for that extension, per variant; `None` leaves only the iteration cap |
| `tournament_games` | `75` | Games each champion plays in the tournament |
| `tournament_learners` | `6` | Learner copies per tournament game |
| `workers` | `1` | CPU workers for the trainers and the tournament |
| `seed` | `0` | Seed for the simulation config and every trainer |
| `results_dir` | `"results"` | Where run folders go |

With the defaults a variant trains until it has won a majority of its validation games over five iterations, or for 20 iterations, whichever comes first. With `extended_iterations` set, a variant that hits the 20 without meeting the criterion gets a second chance after every other variant has had its first budget: it keeps training, with the same population or weights, until it meets the criterion or runs out of the extension. That answers "how long does a slow starter really need" instead of cutting it off, and its final network still enters the tournament, so the report can say whether the wait was worth it.

### `set_overrides(config: SimulationConfig, overrides: dict) -> SimulationConfig`

Deep-copies the config, then for each `"a.b.c": value` walks `getattr` through `a` and `b` and calls `setattr` on the last part. The original is untouched. A key with no dot sets a top-level field, e.g. `{"chaos": 0.0}`.

### `count_lines(method: str) -> int`

Sums `len(read_text().splitlines())` over the files a method needs, relative to `hunger_games/training/`.

| Method | Files counted | Lines at the time of writing |
| --- | --- | --- |
| imitation | `imitation.py` | 525 |
| genetic | `genetic.py` | 757 |
| neat | `neat.py`, `../brain/neat.py` | 492 + 489 = 981 |
| reinforce | `reinforce.py` | 753 |
| ppo | `reinforce.py`, `ppo.py` | 753 + 141 = 894 |

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
| `criterion: dict[str, tuple[int \| None, float \| None]]` | `train_all` | Per variant, `(iterations, seconds)` at which the win criterion was first met, or `(None, None)` |
| `extended: dict[str, int]` | `train_all` | Per extended variant, the iterations trained in the extension phase |
| `_pending: dict[str, Any]` | `train_all` | Trainers waiting for the extension phase (emptied as they run) |
| `run_dir: Path \| None` | `run` | The run folder (`None` until `run` is called) |
| `_stop` | `stop` | Stop flag |

#### `_build(self, variant: Variant) -> Any`

Constructs the trainer for one variant.

1. Looks up the trainer class and default settings in `METHODS`. Uses `variant.settings` if given, else the defaults.
2. Copies the shared knobs onto the settings: `workers` and `seed` from `ComparisonConfig`, but only if the settings object has that attribute (all five do).
3. Applies `variant.config_overrides` to a copy of `base_config`, then sets `config.seed = comparison.seed`.
4. Builds `Curriculum(CurriculumConfig())` if `variant.curriculum` is true, else `None`. Always the default curriculum: promotion on a majority of validation wins, no timeout.
5. Warm start: if `warm_from` names a variant that has already trained, takes its champion spec. The genome is passed only when the kinds match: a NEAT champion goes to a NEAT trainer, a neural champion goes to a non-NEAT trainer. Otherwise `initial_genome` is `None` and the trainer starts fresh.
6. Returns `trainer_class(config, settings, initial_genome=initial, curriculum=curriculum)`.

#### `_train_steps(self, variant, trainer, iterations, started, deadline, on_progress) -> tuple[int | None, float | None]`

Runs up to `iterations` steps of one trainer and returns `(iterations completed, seconds since started)` the moment the win criterion is met, or `(None, None)` when the steps or the deadline run out first. Both training phases use it. Each pass:

1. Stops if `_stop` is set.
2. `stats = trainer.step()`, then `done = len(trainer.learning_history)`: the iterations this trainer has completed over both phases, because every trainer numbers its records from 0 and keeps counting. `on_progress(variant.name, done, stats)` lets the caller show the stage, opponents and win rate of the iteration just finished.
3. If `save_replays_every` is set and this iteration number divides by it, the showcase recording is saved to `<run>/replays/<variant>/iteration_NNNN.replay` (the variant name reduced to a single path component, as `make_run_dir` does).
4. **The win criterion.** `recent` is the last `win_window` entries of `trainer.learning_history`. `at_final_stage` is true when the trainer has no curriculum or its curriculum is `finished`. `played_at_final` is true when there is no curriculum or every entry in `recent` records the curriculum's current (final) stage: each `IterationStats` carries the stage its games were played at, and promotion happens after the record, so the wins that earned the last promotion cannot double as wins against the full field. If `until_win_rate` is not `None`, the variant is at the final stage, every window entry was played there, `recent` has `win_window` entries, and the mean of their `val_win_rate` is at least `until_win_rate`, `on_progress(variant.name, done, "reached the win criterion")` is called (the third argument is a string this time) and `(done, time.time() - started)` is returned.
5. Otherwise, if `deadline` (an absolute time, or `None`) has passed, the loop breaks.

The criterion is tested before the deadline, so a variant that meets both in the same step is recorded as having reached the criterion.

#### `_finish_variant(self, variant, trainer, reached, started, subdir) -> None`

Records one variant's results: `criterion[name] = reached`, `train_seconds[name] = time.time() - started`, `learning[name]` (every `IterationStats` as a row) and the champion. If `run_dir` is set, `save_run(trainer, variant.method, variant.name, run_dir / subdir)` writes the variant's own run folder under `runs/` or `runs_first_budget/`.

#### `train_all(self, on_progress: Callable[[str, int, Any], None] | None = None) -> None`

Two phases.

**Phase one: the first budget.** Every variant in list order (so a warm start can use an earlier champion): build the trainer, note the start time, compute the deadline from `time_budget` (or `None`), and call `_train_steps` with `iterations`. Then `_finish_variant`. If extension is on (`extended_iterations > 0` and `until_win_rate` is not `None`), the variant missed the criterion, and no stop was requested, the results go under `runs_first_budget/` and the trainer is kept in `_pending`; otherwise they go under `runs/`.

**Phase two: the extension.** Again in list order, every pending trainer keeps training from where it stopped, same population or weights. The callback first gets `(name, iterations so far, "extending: the win criterion was not met within the first budget")`. `started` is set to `time.time() - train_seconds[name]`, so the clock continues from the variant's first iteration and `seconds_to_criterion` stays comparable with variants that finished in phase one (idle time waiting for the other variants is not counted). The deadline is `now + extended_time_budget`, or `None`. `_train_steps` runs with `extended_iterations`; `extended[name]` records how many iterations that added; `_finish_variant` writes the final state under `runs/`, replacing the recorded champion, learning rows, criterion and seconds.

So `runs/` always holds the final state of every variant, and `runs_first_budget/` the snapshot of the extended ones at the end of phase one. The `_stop` flag is checked before each variant and each step in both phases.

#### `run_tournament(self, on_progress: Callable[[str, int, int], None] | None = None) -> None`

Every champion plays the same seeded games as the learner against voting opponents.

- The game config is `base_config` with `seed = comparison.seed`. No variant overrides apply: everyone fights on the same arena settings. A champion still plays with its own architecture, because its `LearnerSpec` carries its `NeuralConfig` (or its NEAT genome).
- `learners = learner_ids(config.num_players, tournament_learners)`: with 24 players and 6 learners, slots 0, 4, 8, 12, 16, 20.
- One job per game: `(config, None, spec, learners, 50000 + i, True, False)`. That is no scenario, seed `50000 + i`, greedy play (chaos 0, argmax), no recording. The seeds are the same for every champion.
- Runs the jobs through `_run_episode_job`, with a `ProcessPoolExecutor` when `workers > 1`.
- Stores, per variant:

| Key | Value | Averaged over |
| --- | --- | --- |
| `mean_score` | Mean episode return under `RewardConfig` | Every learner outcome: 75 games times 6 learners, 450 learner episodes |
| `win_rate` | Mean of `learner_won`: the fraction of games any learner copy won | The 75 games |
| `mean_survival` | Mean ticks survived | Every learner outcome |
| `mean_kills` | Mean kills | Every learner outcome |
| `games` | Number of games played | |

`on_progress(name, done, total)` is called after each champion.

#### `run(self, on_progress: Callable[[str, str], None] | None = None) -> Path`

Train, fight, write. Clears the stop flag, makes the run folder with `make_run_dir(results_dir, name)`, writes `config.json` (the base config, the comparison settings including `until_win_rate` and `win_window`, and each variant with its settings as a dictionary), then calls `train_all`, `run_tournament` and `write`. The progress callback receives `(name, "iteration 3: stage 1 (2 opponents), validation win rate 0.50, mean score 0.12, 9.8s")` during training (`run` formats each `IterationStats` into that line, or passes the criterion message through unchanged) and `(name, "tournament 2/5")` during the tournament. Returns the run folder.

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
| `reached_criterion` | `True` when `criterion[name]` has an iteration count |
| `iterations_to_criterion` | The iteration count from `criterion`, or `None` |
| `seconds_to_criterion` | The seconds from `criterion`, or `None` |
| `extended_iterations` | Iterations trained in the extension phase (`0` when not extended) |
| `final_val_win_rate` | `val_win_rate` of the last iteration |
| `tournament_score` | `mean_score` from the tournament |
| `tournament_win_rate` | `win_rate` from the tournament |
| `tournament_survival` | `mean_survival` from the tournament |
| `tournament_kills` | `mean_kills` from the tournament |
| `lines_of_code` | `count_lines(method)` |

A variant that has not trained or fought gets `NaN` in the missing columns. A variant that never met the criterion has `reached_criterion` false and `None` in the two criterion columns.

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
| `plots/win_rate_by_method.png` | `val_win_rate` against `iteration` (titled "games won by the learner") |
| `plots/win_rate_by_time.png` | `val_win_rate` against `cumulative_seconds` |
| `plots/curriculum_by_method.png` | `opponents` against `iteration`: the curriculum ladder each variant climbed |
| `plots/length_by_method.png` | `mean_length` (ticks) against `iteration` |
| `plots/tournament_mean_score.png`, `tournament_win_rate.png`, `tournament_mean_survival.png`, `tournament_mean_kills.png` | Bar charts, only if the tournament ran |
| `plots/lines_of_code.png` | Bar chart of `lines_of_code` |
| `plots/train_seconds.png` | Bar chart of `train_seconds` |
| `report.md` | `self.report(table)` |

The overlay charts come from `plots.overlay_curves`, the bars from `plots.bars`. A variant without a curriculum draws a flat line at `num_players - 1` on the curriculum chart.

#### `report(self, table: pd.DataFrame) -> str`

A generated Markdown write-up:

1. `# Method comparison: <name>` and one sentence stating the budget (`iterations`, the time budget if any, and `tournament_games`). The sentence says "up to" the iteration count, because the win criterion can stop a variant early.
2. If the tournament ran, `## Ranking by tournament score`: a table sorted by `tournament_win_rate` first and `tournament_score` second, both descending, with columns rank, variant, method, tournament score, tournament win rate, survival, iterations, train seconds, lines of code. The heading still says "score"; the sort key is the win rate. Then three bold lines: **Best in the tournament** (top row, with its score and win rate), **Simplest to implement** (the method with the fewest lines), **Fastest to train under this budget** (the variant with the fewest `train_seconds`).
3. If `until_win_rate` is not `None`, `## Training to the win criterion (...)` with the threshold as a percentage, the window and "at the final curriculum stage" in the heading. When `extended_iterations` is set, a paragraph first explains the two phases and their caps. One row per variant: reached (`yes` or `no`), iterations to criterion, seconds to criterion (a dash when never reached), iterations trained, iterations added by the extension, and the final validation win rate.
4. If any `_cold` variant has a `_warm` twin and the tournament ran, `## Warm start against cold start`: one row per pair with the method name (the `_cold` suffix stripped), the cold and warm tournament win rates, the cold and warm iterations to criterion, and the cold and warm seconds to criterion (a dash when never reached). Then one sentence naming the variant with the higher tournament win rate in each pair. On a tie the cold variant is named.
5. `## Why the methods differ`: one bullet per distinct method with its `METHOD_NOTES` line.
6. `## Charts`: a fixed list of PNG names (`score_by_method`, `score_by_time`, `validation_by_method`, `entropy_by_method`, `length_by_method`, `tournament_*`, `lines_of_code`, `train_seconds`) and a note that each variant's own run folder is under `runs/`. The three win-rate and curriculum charts are written but not named in this list.

## How to use it / experiment

**All five methods, defaults.**

```python
from hunger_games.config import SimulationConfig
from hunger_games.research.comparison import ComparisonConfig, MethodComparison, Variant

config = SimulationConfig(width=120, height=120)
variants = [Variant(m, m) for m in ("imitation", "genetic", "neat", "reinforce", "ppo")]
folder = MethodComparison(config, ComparisonConfig(iterations=20, workers=4), variants).run(print)
```

Each variant stops as soon as it has won at least half of its validation games over five iterations, so the `iterations` column of the table can differ between rows.

**Train for a fixed budget instead.** `ComparisonConfig(iterations=20, until_win_rate=None)` turns the criterion off; every variant then runs the full 20 iterations (or the time budget).

**A stricter criterion.** `ComparisonConfig(iterations=200, until_win_rate=0.75, win_window=10)` asks for three wins in four over ten iterations, with room to get there.

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

**Warm versus cold pairs with the curriculum.** Put the imitation variant first, name it in `warm_from`, and use the `_cold` and `_warm` suffixes so the report pairs them.

```python
variants = [
    Variant("imitation", "imitation"),
    Variant("ppo_cold", "ppo", curriculum=True),
    Variant("ppo_warm", "ppo", warm_from="imitation", curriculum=True),
]
```

The report then has a "Warm start against cold start" table with both tournament win rates and both iteration and second counts to the criterion.

**Let slow starters finish.** `ComparisonConfig(iterations=150, extended_iterations=1000, extended_time_budget=2 * 3600)`: every variant gets 150 iterations first; those still short of the criterion then continue for up to 1000 more iterations or two hours each, after the quick ones are done. The criterion table shows how many iterations the slow ones needed, and the tournament shows whether their final networks compete.

**A time budget instead of an iteration count.** `ComparisonConfig(iterations=1000, time_budget=600)` gives every variant ten minutes; `score_by_time.png` and `win_rate_by_time.png` are then the fair charts, because a NEAT generation and an imitation epoch take very different times.

**Custom settings.** Pass the method's settings dataclass: `Variant("ppo_big_batch", "ppo", PPOConfig(episodes_per_epoch=8, update_epochs=8))`. `workers` and `seed` on it are overwritten by the comparison's values.

**Reading the result.** Open `report.md` first for the ranking, the criterion table and the pair table. `results.csv` has every number; `summary.json` adds the full learning curves. `runs/<variant>_<timestamp>/` holds each variant's own charts (`score.png`, `entropy_shared.png`, `game_length.png`, `curriculum.png` and the method's own plots); with extension on, `runs_first_budget/` holds the extended variants as they stood after the first budget.

## Gotchas
- `ComparisonConfig.save_replays_every` (default 0) saves the showcase recording of every Nth iteration per variant to `<run>/replays/<variant>/iteration_NNNN.replay`, which the dashboard's Play tab can load; with 0 nothing is saved. Replays are pickle files of the app's own recordings.

- **The criterion needs the final curriculum stage.** With `curriculum=True` a variant is only tested once `curriculum.finished` is true. A learner that never climbs the ladder is never tested and runs its full budget, however well it wins on stage 0.
- **A variant that never wins never reaches the final stage.** With no curriculum timeout, a learner that cannot win half of its validation games at a stage stays there for the whole budget; `reached_criterion` is then false and `curriculum_by_method.png` shows where it stalled. That is a result to report, not an error.
- **Validation win rates are coarse.** REINFORCE, PPO, the GA and NEAT play 2 validation games per iteration by default and imitation plays 1, so each iteration's `val_win_rate` is 0, 0.5 or 1 (imitation: 0 or 1). Averaged over `win_window=5` that is a mean of ten (or five) coin-flip-like results. Raise the trainer's `validation_games` through `Variant.settings` before trusting a criterion crossing.
- **Order matters for warm starts.** `warm_from` only works if the named variant appears earlier in the list. If it does not, or the name is misspelled, the trainer silently starts fresh. Check the first iteration's score against the imitation variant's if you doubt it.
- **Kinds must match.** A neural champion cannot seed NEAT and a NEAT champion cannot seed a neural method; those combinations start fresh. A genetic variant with `TrainingConfig(brain_name="voting")` has kind `"voting"`, which the rule treats like neural, so do not warm a neural method from a voting-brain champion.
- **Pairs are matched by name only.** `ppo_cold` and `ppo_warm` are paired whether or not `ppo_warm` actually has a `warm_from`. Name variants carefully.
- **Settings are mutated.** `_build` sets `workers` and `seed` on the settings object you pass in. Make a new object per variant.
- **The time budget is checked after each step.** The last step can overrun it. A NEAT generation with a large population can take minutes.
- **The tournament ignores config overrides.** By design, every champion plays the same base arena, roster size and opponent brain. An override such as `chaos` or `num_players` shapes training only.
- **Imitation's curriculum flag does nothing.** `ImitationTrainer` stores the curriculum but never resizes its roster and never calls `observe`, so its curriculum is never `finished` and an imitation variant with `curriculum=True` can never meet the criterion. `run_comparison.py` only enables the curriculum for the four reward methods.
- **Score, survival and kills are means of learner episodes; the win rate is a mean of games.** With 6 learners per game the first three average 450 numbers and the win rate averages 75.
- **The ranking heading says score but sorts by win rate.** Two variants with equal win rates are ordered by score. The "best in the tournament" line follows the same order.
- **NaN rows.** If `stop()` interrupts the run, later variants have no learning rows and no tournament entry; the table shows `NaN` and the report's ranking sorts them last.
- **`count_lines` reads files by path.** Moving or renaming a training module breaks it with `FileNotFoundError`.
