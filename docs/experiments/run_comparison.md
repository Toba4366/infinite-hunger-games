# `run_comparison.py`

**Source:** [experiments/run_comparison.py](../../experiments/run_comparison.py)
**Depends on:** `argparse`, `sys`, `pathlib` (standard library); [hunger_games/config.py](../config.md) (`SimulationConfig`); [research/comparison.py](../research/comparison.md) (`ComparisonConfig`, `MethodComparison`, `Variant`)
**Used by:** nobody imports it. It is run from the command line.

## Purpose

A command-line wrapper around [research/comparison.md](../research/comparison.md). It turns flags into a list of `Variant`s (methods, optionally crossed with network sizes or initializers, optionally warm-started and trained with the curriculum), runs the comparison and the tournament, prints the results table, and says where the run folder is.

```
python experiments/run_comparison.py --iterations 20 --workers 4
python experiments/run_comparison.py --methods imitation,ppo,neat --iterations 30 --games 75
python experiments/run_comparison.py --sizes 16,64x32 --methods ppo
python experiments/run_comparison.py --initializers xavier_uniform,he_uniform,zeros --methods ppo
```

## Concepts you need

**Variant.** One thing to compare: a method plus its settings and any tweak to the simulation config. See [../research/comparison.md](../research/comparison.md). This script builds variants from flags; it never edits a method's settings dataclass, so every trainer runs with its defaults apart from `workers` and `seed`.

**Iterations.** One `step()` of a trainer: an epoch for imitation, REINFORCE and PPO, a generation for the genetic algorithm and NEAT. `--iterations` is the shared budget.

**Warm start.** Starting a trainer from an existing network instead of random weights. Here it means starting from the imitation champion.

**Curriculum.** The opponent ladder from [../training/common.md](../training/common.md): 1, 3, 7, 11, then 23 voting opponents, promoted when the recent mean score clears a threshold or after enough iterations.

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
| `--methods` | `imitation,genetic,neat,reinforce,ppo` | one `Variant` per name | Comma-separated method names, in training order |
| `--iterations` | `20` | `ComparisonConfig.iterations` | Iterations per variant |
| `--time-budget` | `None` | `ComparisonConfig.time_budget` | Seconds per variant; stops a variant early once reached |
| `--games` | `75` | `ComparisonConfig.tournament_games` | Tournament games per champion |
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

**Variants.** For each method name in `--methods` order:

- `warm` is `"imitation"` when `--warm` is set, the method is not `imitation` or `neat`, and `imitation` is in the list. Otherwise `None`.
- `curriculum` is true when `--curriculum` is set and the method is one of `reinforce`, `ppo`, `genetic`, `neat`.
- With `--sizes`, one variant per size named `<method>_<size>` with `config_overrides={"neural.hidden_layers": parse_layers(size)}`.
- Else with `--initializers`, one variant per name, `<method>_<initializer>`, with `config_overrides={"neural.initializer": name}`.
- Else one variant named after the method.

`--sizes` wins over `--initializers` when both are given.

**Run.** `MethodComparison(config, ComparisonConfig(...), variants).run(on_progress=...)`. The callback prints one line per event:

```
imitation: iteration 1
imitation: iteration 2
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

**The headline experiment: five methods, then the tournament.**

```bash
python experiments/run_comparison.py --iterations 20 --games 75 --workers 4
```

**Warm starts and the curriculum.** Imitation first, then PPO and REINFORCE from its champion, all against a growing field.

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

**A fair time budget.** `--iterations 1000 --time-budget 600` gives each variant ten minutes; read `plots/score_by_time.png`.

**A quick smoke test.** `--iterations 2 --games 4 --size 60 --days 4` finishes in a few minutes on one core and writes every file.

### The results folder

`results/<name>_<timestamp>/`:

| Path | Contents |
| --- | --- |
| `config.json` | The base config, the comparison settings, every variant and its settings |
| `results.csv` | One row per variant: `variant`, `method`, `iterations`, `train_seconds`, `final_mean_score`, `best_val_score`, `tournament_score`, `tournament_win_rate`, `tournament_survival`, `tournament_kills`, `lines_of_code` |
| `summary.json` | The table rows, the tournament dictionary, and every learning curve row |
| `results_table.tex` | The same table for a paper |
| `report.md` | The generated ranking, the best, simplest and fastest, the method notes, the chart list |
| `plots/score_by_method.png` | Mean score per iteration, one line per variant |
| `plots/score_by_time.png` | Mean score against training seconds |
| `plots/validation_by_method.png` | Validation score per iteration |
| `plots/entropy_by_method.png` | Policy entropy per iteration |
| `plots/length_by_method.png` | Average game length per iteration |
| `plots/tournament_mean_score.png`, `tournament_win_rate.png`, `tournament_mean_survival.png`, `tournament_mean_kills.png` | Tournament bars |
| `plots/lines_of_code.png`, `plots/train_seconds.png` | Implementation size and training time bars |
| `runs/<variant>_<timestamp>/` | Each variant's own `save_run` folder: `config.json`, `history.json`, `learning.json`, `events.txt`, `champion.json`, `plots/` |

## Gotchas

- **`--warm` needs `imitation` in `--methods`.** The script moves the imitation variants to the front of the list itself, and with `--sizes` or `--initializers` each variant warm-starts from the imitation variant with the same suffix (`ppo_64x32` from `imitation_64x32`). Without imitation in `--methods`, `--warm` does nothing.
- **`--warm` does nothing with `--sizes` or `--initializers`.** The warm-start name is always `"imitation"`, but the variants are then named `imitation_16` and so on, so the lookup never matches and every variant starts fresh. Build the variants in Python (see [../research/comparison.md](../research/comparison.md)) if you need both.
- **NEAT is never warm-started** from imitation, because an imitation champion is a fixed-shape network and NEAT needs a NEAT genome.
- **`--curriculum` has no effect on imitation**, which never plays training games.
- **The tournament always uses the base config.** `--sizes` and `--initializers` shape training; in the tournament every champion plays the same 24-tribute arena with its own network.
- **Runtime adds up.** Five methods at 20 iterations plus 75 games each is hours on one core. Use `--workers` and start with a smoke test.
- **The unknown method name error is a `KeyError`** from `METHODS`, raised when the first variant is built, after the run folder has already been created.
