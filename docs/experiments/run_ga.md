# `run_ga.py`

**Source:** [experiments/run_ga.py](../../experiments/run_ga.py)
**Depends on:** `argparse`, `sys`, `pathlib` (standard library); [hunger_games/config.py](../config.md) (`NeuralConfig`, `SimulationConfig`); [training/__init__.py](../training/init.md) (`GeneticTrainer`, `TrainingConfig`, `save_run`)
**Used by:** nobody imports it. It is run from the command line.

## Purpose

A command-line wrapper around [training/genetic.md](../training/genetic.md). It builds a `SimulationConfig` and a `TrainingConfig` from flags, runs the genetic algorithm while printing one line per generation, and hands the trainer to `save_run` so the results land in a timestamped folder under `results/`. The whole script is 68 lines; the value is in not having to write the boilerplate each time.

```
python experiments/run_ga.py --brain neural --population 48 --generations 20 --workers 4
```

## Concepts you need

**`argparse`.** The standard library's flag parser. Each `add_argument` call declares a flag, its type and its default. `parser.parse_args()` returns a namespace whose attributes are the flag names with dashes turned into underscores.

**`sys.path`.** The list of folders Python searches for imports. A script in `experiments/` cannot see the `hunger_games` package next door unless the repo root is on that list. The script inserts it itself so it works from any working directory without installing the package.

**The `__main__` guard.** `if __name__ == "__main__": main()` runs `main()` only when the file is executed directly, not when it is imported. On macOS, `multiprocessing` starts worker processes with `spawn`, which imports the main script again in each worker. Without the guard, every worker would start its own training run. With it, workers import the module, find the functions they need, and go no further.

## Walkthrough

### Path setup

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

`Path(__file__).resolve()` is the absolute path of the script, `.parent.parent` is the repo root. Inserting at index 0 puts it ahead of everything else. The two package imports that follow carry `# noqa: E402` because ruff would otherwise complain about imports that are not at the top of the file; they cannot be, since the path must be set first.

### `main()`

```python
def main() -> None:
```

**Flags.**

| Flag | Default | Goes to | Meaning |
| --- | --- | --- | --- |
| `--brain` | `neural` | `TrainingConfig.brain_name` | `neural` or `voting` (choices enforced) |
| `--population` | `48` | `TrainingConfig.population_size` | Genomes alive at once |
| `--generations` | `20` | `TrainingConfig.generations` | Generations to run |
| `--rounds` | `2` | `TrainingConfig.rounds_per_generation` | Games per genome per generation |
| `--workers` | `1` | `TrainingConfig.workers` | CPU cores |
| `--seed` | `0` | `SimulationConfig.seed` and `TrainingConfig.seed` | Reproducibility |
| `--hidden` | `16` | `NeuralConfig.hidden_layers` | Comma-separated widths, e.g. `32,16` |
| `--activation` | `tanh` | `NeuralConfig.activation` | Activation name |
| `--initializer` | `xavier_uniform` | `NeuralConfig.initializer` | Initializer name |
| `--size` | `120` | `SimulationConfig.width` and `height` | Arena size in cells |
| `--days` | `24` (`SimulationConfig.max_days`) | `SimulationConfig.max_days` | Day cutoff |
| `--name` | `ga` | `save_run(name=...)` | Run folder prefix |
| `--results` | `results` | `save_run(results_dir=...)` | Where the folder goes |

`--hidden` is parsed with `tuple(int(w) for w in args.hidden.split(",") if w)`, so `--hidden ""` gives no hidden layers (a linear policy) and `--hidden 32,16` gives two layers.

**Configs.** `NeuralConfig(hidden_layers, activation, initializer)`; `SimulationConfig(width=size, height=size, max_days=days, seed=seed, neural=neural)`; `TrainingConfig(brain_name, population_size, generations, rounds_per_generation, workers, seed)`. Every `TrainingConfig` field not listed keeps its default: `elite_fraction=0.1`, `tournament_size=3`, `crossover_rate=0.5`, `mutation_rate=0.1`, `mutation_scale=0.1`, `kills_weight=0.05`, `days_weight=0.01`, `validation_games=2`, `validation_seed=90000`, `collect_telemetry=True`. Every `SimulationConfig` field not listed keeps its default too, so games have 24 tributes and validation opponents use the `voting` brain.

**Train.** `GeneticTrainer(config, training).run(on_generation=...)`. The callback prints

```
gen   3  best 0.812  mean 0.541  val 0.467  12.4s
```

per generation: `generation`, `best_fitness`, `mean_fitness`, `val_fitness`, `seconds`.

**Save.** `folder = save_run(trainer, "genetic", args.name, args.results)`, then `print(f"saved to {folder}")`.

### The guard

```python
if __name__ == "__main__":
    main()
```

See Concepts. The comment in the source says exactly why it is there: "Needed for multiprocessing on macOS."

## How to use it / experiment

**Smoke test in a minute.**

```
python experiments/run_ga.py --brain voting --population 24 --generations 3 --rounds 1 --size 60 --days 6
```

**A serious neural run on four cores.**

```
python experiments/run_ga.py --brain neural --hidden 32,16 --population 96 --generations 200 --rounds 3 --workers 4 --name neural_32_16
```

**Compare initializers.** Run twice with `--initializer xavier_uniform` and `--initializer he_normal`, same seed, then plot `val_fitness` from both `history.json` files.

**What the results folder contains.** `results/<name>_<timestamp>/` with `config.json` (method, simulation config, trainer config), `history.json` (one row per generation: generation, best, mean, worst and validation fitness, seconds, cumulative seconds), `champion.json` (the best genome and its `NeuralConfig`), and `plots/` with `fitness.png`, `fitness.gif`, `timing.png` and the behaviour charts. The full list is in [../training/runs.md](../training/runs.md).

**Play the champion.** Open the dashboard, Train tab, "Load champion into all", pick `champion.json`.

**Add a flag.** To expose `mutation_scale`, add `parser.add_argument("--mutation-scale", type=float, default=0.1)` and pass `mutation_scale=args.mutation_scale` to `TrainingConfig`.

## Gotchas

- Run it from a file, never by pasting into the interactive prompt, when `--workers` is above 1. The spawn start method needs an importable script.
- `--seed` seeds the trainer; the game seed in `SimulationConfig` is overwritten per game by the trainer's own job seeds. Two runs with the same `--seed` and settings reproduce each other; changing only `--seed` changes everything.
- `--population` should be a multiple of 24 (the default player count). Otherwise `_make_jobs` pads each round with random duplicates and some genomes get extra games.
- `--hidden` and friends only matter with `--brain neural`. The voting brain ignores `NeuralConfig`.
- Fitness weights, elite fraction, tournament size, mutation, validation games and telemetry are not exposed as flags. Edit the `TrainingConfig(...)` call or write your own script from this one.
- The default 20 generations of a neural brain is a smoke test. Do not expect `val_fitness` to move much.
- Telemetry is on by default, which slows evaluation. There is no flag to turn it off; set `collect_telemetry=False` in the script if you want speed over behaviour charts.
