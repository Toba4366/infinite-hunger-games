# `run_comparison.py`

**Source:** [experiments/run_comparison.py](../../experiments/run_comparison.py)
**Depends on:** `argparse`, `sys`, `pathlib` (standard library); [hunger_games/config.py](../config.md) (`SimulationConfig`); [research/comparison.py](../research/comparison.md) (`ComparisonConfig`, `MethodComparison`, `Variant`)
**Used by:** nobody imports it. It is run from the command line.

## Purpose

A command-line wrapper around [research/comparison.md](../research/comparison.md). It turns flags into a list of `Variant`s (methods, optionally crossed with network sizes or initializers, optionally warm-started, paired cold and warm, and trained with the curriculum), runs the comparison to the win criterion and the tournament, prints the results table, and says where the run folder is.

```
python experiments/run_comparison.py --iterations 20 --workers 4
python experiments/run_comparison.py --methods imitation,ppo,neat --iterations 30 --games 75
python experiments/run_comparison.py --sizes 16,64x32 --methods ppo
python experiments/run_comparison.py --initializers xavier_uniform,he_uniform,zeros --methods ppo
```

## Concepts you need

**Variant.** One thing to compare: a method plus its settings and any tweak to the simulation config. See [../research/comparison.md](../research/comparison.md). This script builds variants from flags; it never edits a method's settings dataclass, so every trainer runs with its defaults apart from `workers` and `seed`.

**Iterations.** One `step()` of a trainer: an epoch for imitation, REINFORCE and PPO, a generation for the genetic algorithm and NEAT. `--iterations` is the most a variant will train.

**Win criterion.** A variant stops early once its validation win rate over the last `--window` iterations reaches `--until-win`, at the final curriculum stage. A win is game-level: a game is won when any learner copy was the victor. The iterations and seconds it took are recorded in the results table.

**Extension.** With `--extend-iterations`, a variant that has not met the criterion by `--iterations` is not cut off for good: once every variant has had its first budget, the slow ones keep training (same population or weights) for up to that many more iterations, or `--extend-hours` each. Quick variants finish first, slow ones are measured for how long they really need, and every final network enters the tournament.

**Warm start.** Starting a trainer from an existing network instead of random weights. Here it means starting from the imitation champion.

**Cold and warm pairs.** With `--pairs`, every reward or evolution method (other than NEAT) gets two variants: `<method>_cold` from random weights and `<method>_warm` from the imitation champion. The report compares each pair.

**Curriculum.** The opponent ladder from [../training/common.md](../training/common.md): 1, 3, 7, 11, then 23 voting opponents, promoted when the learner has won at least half of its validation games over the last five iterations. There is no timeout.

**`sys.path` and the `__main__` guard.** Same as in [run_ga.md](run_ga.md): the script puts the repo root first on `sys.path` so `import hunger_games` works from anywhere, and `main()` is guarded so that `spawn`-started worker processes on macOS import the file without starting a second comparison.

## Walkthrough

### Path setup

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

Then the two package imports with `# noqa: E402`, because they must come after the path change.

### `parse_layers(text: str) -> tuple[int, ...]`

`"64x32"` becomes `(64, 32)`; `"16"` becomes `(16,)`. Splits on `x` and skips empty parts, so `"64x"` is `(64,)`.

### `main() -> None`

**Flags.**

| Flag | Default | Goes to | Meaning |
| --- | --- | --- | --- |
| `--methods` | `imitation,genetic,neat,reinforce,ppo` | one or two `Variant`s per name | Comma-separated method names |
| `--iterations` | `20` | `ComparisonConfig.iterations` | Most iterations per variant |
| `--time-budget` | `None` | `ComparisonConfig.time_budget` | Seconds per variant; stops a variant early once reached |
| `--games` | `75` | `ComparisonConfig.tournament_games` | Tournament games per champion |
| `--until-win` | `0.5` | `ComparisonConfig.until_win_rate` | Stop a variant once it wins this share of validation games over `--window` iterations; a negative value means never (`None`) |
| `--window` | `5` | `ComparisonConfig.win_window` | Iterations averaged for the win criterion |
| `--extend-iterations` | `0` | `ComparisonConfig.extended_iterations` | After every variant has had `--iterations`, keep training those short of the criterion for up to this many more; `0` never extends |
| `--extend-hours` | `None` | `ComparisonConfig.extended_time_budget` (hours times 3600) | Wall-clock cap per variant for that extension |
| `--pairs` | off | two `Variant`s per method | Train a cold and a warm-started variant of every reward or evolution method; needs `imitation` in `--methods` |
| `--workers` | `1` | `ComparisonConfig.workers` | CPU workers for trainers and the tournament |
| `--seed` | `0` | `ComparisonConfig.seed` | Seed for the config and every trainer |
| `--curriculum` | off | `Variant.curriculum` | Train `reinforce`, `ppo`, `genetic` and `neat` with the opponent curriculum |
| `--warm` | off | `Variant.warm_from` | Warm-start every method except `imitation` and `neat` from the imitation champion; needs `imitation` in `--methods` |
| `--sizes` | `None` | `neural.hidden_layers` override | Hidden-layer variants, e.g. `16,64x32,128x64` |
| `--initializers` | `None` | `neural.initializer` override | Initializer variants, e.g. `xavier_uniform,he_uniform,zeros` |
| `--size` | `120` | `SimulationConfig.width` and `height` | Arena size |
| `--days` | `24` (`SimulationConfig.max_days`) | `SimulationConfig.max_days` | Day cutoff |
| `--name` | `comparison` | `ComparisonConfig.name` | Run folder prefix |
| `--results` | `results` | `ComparisonConfig.results_dir` | Where the folder goes |

**Base config.** `SimulationConfig(width=size, height=size, max_days=days)`. Everything else is the default: 24 tributes, the voting brain as opponents at chaos 0.5, a 64 by 32 tanh network with the Xavier uniform initializer.

**Variants.** The method names are sorted so that `imitation` comes first (the others keep their order), because the warm starts need its champion. Then for each method:

- `curriculum` is true when `--curriculum` is set and the method is one of `reinforce`, `ppo`, `genetic`, `neat`.
- `can_warm` is true when `--warm` is set, the method is not `imitation` or `neat`, and `imitation` is in the list.
- With `--sizes`, one variant per size named `<method>_<size>` with `config_overrides={"neural.hidden_layers": parse_layers(size)}`, warm-started from `imitation_<size>` when `can_warm`.
- Else with `--initializers`, one variant per name, `<method>_<initializer>`, with `config_overrides={"neural.initializer": name}`, warm-started from `imitation_<initializer>` when `can_warm`.
- Else with `--pairs`, when the method is not `imitation` or `neat` and `imitation` is in the list: two variants, `<method>_cold` with no warm start and `<method>_warm` with `warm_from="imitation"`. `--warm` is not needed for the warm twin.
- Else one variant named after the method, warm-started from `imitation` when `can_warm`.

`--sizes` wins over `--initializers`, and both win over `--pairs`. Under `--pairs`, `imitation` and `neat` still get one plain variant each.

**Run.** `MethodComparison(config, ComparisonConfig(...), variants).run(on_progress=...)`. `until_win_rate` is `None` when `--until-win` is negative, else the given value. The callback prints one line per event. Training lines carry the curriculum stage, the number of voting opponents at that stage, the validation win rate, the mean score and the seconds the iteration took, so a log tail shows at a glance whether a variant is still winning and climbing the ladder:

```
imitation: iteration 1: stage 0 (23 opponents), validation win rate 0.00, mean score -1.56, 16.2s
imitation: iteration 2: stage 0 (23 opponents), validation win rate 0.00, mean score -1.41, 15.9s
...
genetic_warm: iteration 12: stage 2 (4 opponents), validation win rate 1.00, mean score 0.83, 9.7s
genetic_warm: iteration 17: reached the win criterion
...
ppo: tournament 5/5
```

**Print.** `comparison.table().to_string(index=False)`, then `saved to results/comparison_<timestamp>`.

### The guard

```python
if __name__ == "__main__":
    main()
```

"Needed for multiprocessing on macOS", as the source comment says.

## How to use it / experiment

**The headline experiment: five methods to the win criterion, then the tournament.**

```bash
python experiments/run_comparison.py --iterations 20 --games 75 --workers 4
```

Each variant stops as soon as it has won at least half of its validation games over five iterations. `--iterations` is the ceiling.

**Warm against cold, with the curriculum.** Imitation first, then a cold and a warm variant of REINFORCE, PPO and the genetic algorithm, all against a growing field.

```bash
python experiments/run_comparison.py --methods imitation,genetic,reinforce,ppo --pairs --curriculum --iterations 200 --workers 4
```

Read the "Warm start against cold start" table in `report.md`.

**Let the cold starts finish.** The full experiment in `results/run_full.sh`: 150 iterations each first, then up to 1000 more iterations or two hours for any variant still short of the criterion.

```bash
python experiments/run_comparison.py --methods imitation,genetic,neat,reinforce,ppo --pairs --curriculum \
  --iterations 150 --until-win 0.5 --window 5 --extend-iterations 1000 --extend-hours 2 --games 75 --workers 6
```

The criterion table then shows how many iterations and seconds each slow starter needed (the `extended` column counts the extra iterations), and the tournament shows whether its final network beats the warm one that finished sooner.

**A fixed budget with no early stop.**

```bash
python experiments/run_comparison.py --iterations 30 --until-win -1 --workers 4
```

**A stricter criterion.** `--until-win 0.75 --window 10 --iterations 300` asks for three wins in four over ten iterations.

**Warm starts and the curriculum, single variants.** Imitation first, then PPO and REINFORCE from its champion.

```bash
python experiments/run_comparison.py --methods imitation,reinforce,ppo --warm --curriculum --iterations 30 --workers 4
```

**Network sizes.** Three PPO variants that differ only in hidden layers.

```bash
python experiments/run_comparison.py --methods ppo --sizes 16,64x32,128x64 --iterations 30 --workers 4
```

**Initializers.**

```bash
python experiments/run_comparison.py --methods ppo --initializers xavier_uniform,he_uniform,zeros --iterations 30
```

**Sizes across two methods.** `--methods genetic,ppo --sizes 16,64x32` gives four variants: `genetic_16`, `genetic_64x32`, `ppo_16`, `ppo_64x32`. The genetic algorithm's weakness with weight count shows up as the gap between its two rows.

**A fair time budget.** `--iterations 1000 --time-budget 600` gives each variant ten minutes; read `plots/score_by_time.png` and `plots/win_rate_by_time.png`.

**A quick smoke test.** `--iterations 2 --games 4 --size 60 --days 4` finishes in a few minutes on one core and writes every file.

### The results folder

`results/<name>_<timestamp>/`:

| Path | Contents |
| --- | --- |
| `config.json` | The base config, the comparison settings (including `until_win_rate` and `win_window`), every variant and its settings |
| `results.csv` | One row per variant: `variant`, `method`, `iterations`, `train_seconds`, `final_mean_score`, `best_val_score`, `reached_criterion`, `iterations_to_criterion`, `seconds_to_criterion`, `final_val_win_rate`, `tournament_score`, `tournament_win_rate`, `tournament_survival`, `tournament_kills`, `lines_of_code` |
| `summary.json` | The table rows, the tournament dictionary, and every learning curve row |
| `results_table.tex` | The same table for a paper |
| `report.md` | The generated ranking (by tournament win rate, then score), the best, simplest and fastest, the criterion table, the warm-versus-cold table when there are pairs, the method notes, the chart list |
| `plots/score_by_method.png` | Mean score per iteration, one line per variant |
| `plots/score_by_time.png` | Mean score against training seconds |
| `plots/validation_by_method.png` | Validation score per iteration |
| `plots/entropy_by_method.png` | Policy entropy per iteration |
| `plots/win_rate_by_method.png` | Validation win rate per iteration (games won by the learner) |
| `plots/win_rate_by_time.png` | Validation win rate against training seconds |
| `plots/curriculum_by_method.png` | Opponents per iteration: the curriculum ladder each variant climbed |
| `plots/length_by_method.png` | Average game length per iteration |
| `plots/tournament_mean_score.png`, `tournament_win_rate.png`, `tournament_mean_survival.png`, `tournament_mean_kills.png` | Tournament bars |
| `plots/lines_of_code.png`, `plots/train_seconds.png` | Implementation size and training time bars |
| `runs/<variant>_<timestamp>/` | Each variant's own `save_run` folder: `config.json`, `history.json`, `learning.json`, `events.txt`, `champion.json`, `plots/` |

## Gotchas
- `--save-replays-every N` writes a `.replay` of every Nth iteration's training game per variant under `results/<run>/replays/<variant>/`, so a run can be watched afterwards in the dashboard (Play tab, Load replay). Off by default because a 120-cell game is a few megabytes per replay.

- **The default stops early.** `--until-win 0.5` is on unless you pass a negative value, so the `iterations` column differs between rows and `--iterations` is only a ceiling. Pass `--until-win -1` for the old fixed-budget behaviour.
- **`--pairs` needs `imitation` in `--methods`.** Without it every method falls through to a single plain variant and the report has no pair table.
- **`--pairs` is ignored with `--sizes` or `--initializers`.** Those branches come first in the code.
- **`--warm` needs `imitation` in `--methods`.** The script moves the imitation variants to the front of the list itself, and with `--sizes` or `--initializers` each variant warm-starts from the imitation variant with the same suffix (`ppo_64x32` from `imitation_64x32`). Without imitation in `--methods`, `--warm` does nothing.
- **NEAT is never warm-started** from imitation and never paired, because an imitation champion is a fixed-shape network and NEAT needs a NEAT genome.
- **`--curriculum` has no effect on imitation**, which never plays training games.
- **`--curriculum` applies to genetic, neat, reinforce and ppo.** Each judges promotion on the validation win rate (or training-game wins when `validation_games` is 0), so every one of them can climb the ladder; imitation ignores the curriculum.
- **The tournament always uses the base config.** `--sizes` and `--initializers` shape training; in the tournament every champion plays the same 24-tribute arena with its own network.
- **Runtime adds up.** Five methods at 20 iterations plus 75 games each is hours on one core, and a high `--until-win` with a large `--iterations` can be much longer. Use `--workers` and start with a smoke test.
- **The unknown method name error is a `KeyError`** from `METHODS`, raised when the first variant is built, after the run folder has already been created.
