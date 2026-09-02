# `__init__.py`

**Source:** [hunger_games/training/__init__.py](../../hunger_games/training/__init__.py)
**Depends on:** [training/genetic.py](genetic.md) (`GenerationStats`, `GeneticTrainer`, `TrainingConfig`); [training/reinforce.py](reinforce.md) (`EpochStats`, `ReinforceTrainer`, `RLConfig`); [training/runs.py](runs.md) (`save_run`)
**Used by:** [hunger_games/ui/session.py](../ui/session.md) (`GeneticTrainer`, `ReinforceTrainer`, `RLConfig`, `TrainingConfig`, `save_run`); [hunger_games/ui/app.py](../ui/app.md) (`RLConfig`, `TrainingConfig` back the Train tab's controls); [experiments/run_ga.py](../experiments/run_ga.md) (`GeneticTrainer`, `TrainingConfig`, `save_run`); [experiments/run_rl.py](../experiments/run_rl.md) (`ReinforceTrainer`, `RLConfig`, `save_run`); `tests/test_recorder_training.py`; `tests/test_research.py`; `tests/test_ui_session.py`

## Purpose

The `training` package is where brains get better. It holds two trainers and one writer:

- [genetic.py](genetic.md): a genetic algorithm. It keeps a population of genomes, plays them against each other, and breeds the winners. It scores whole games.
- [reinforce.py](reinforce.md): REINFORCE with a value baseline. It trains the neural brain by policy gradient, scoring every single action with the reward weights from [../config.md](../config.md).
- [runs.py](runs.md): `save_run`, which writes either trainer's results to a timestamped folder with one PNG per chart.

This `__init__.py` is a thin front door. It re-exports the seven public names so callers can write

```python
from hunger_games.training import GeneticTrainer, TrainingConfig, ReinforceTrainer, RLConfig, save_run
```

instead of reaching into the three submodules. The docstring states who drives the package: the dashboard's Train tab and the scripts in `experiments/`.

## Concepts you need

**Package front door.** Python runs `__init__.py` when you import the package. Anything imported here becomes an attribute of `hunger_games.training`. The same pattern is used in [../brain/init.md](../brain/init.md) and [../init.md](../init.md).

**`__all__`.** The list of names that `from hunger_games.training import *` brings in. It doubles as a statement of the public API.

**Two ways to learn.** A genetic algorithm needs only a score per game, so it works for any brain with a `genome()` / `set_genome()` pair (the voting brain's eight genes or the neural brain's 1088 weights). Policy gradient needs a probability for each action and a reward after each tick, so it works only for the neural brain. Both trainers share the same outward shape: `run`, `stop`, `history`, `history_rows`, `champion`, `champion_brain` and a save method. That shared shape is what lets the dashboard and `save_run` treat them alike.

**Where training sits.** The simulator is layered: `config` describes settings, `game` plays one game, `runner` plays batches and writes CSVs, `research` measures behaviour and draws charts, and `training` plays batches *to improve a brain*. Training depends on `game` and `research`; nothing in `game` depends on training. A trained genome is loaded back through the ordinary roster and config routes with no special cases.

## Walkthrough

### Imports

Three lines, one per submodule, in this order:

1. `from hunger_games.training.genetic import GenerationStats, GeneticTrainer, TrainingConfig`
2. `from hunger_games.training.reinforce import EpochStats, ReinforceTrainer, RLConfig`
3. `from hunger_games.training.runs import save_run`

Importing `genetic.py` pulls in `Game`, `create_brain`, `SimulationConfig` and `BehaviorTelemetry`. Importing `reinforce.py` adds `MLP`, `Adam`, `NeuralBrain` and `softmax`. Importing `runs.py` adds `research.experiments` and `research.plots`, which import pandas and matplotlib. So `import hunger_games.training` loads most of the simulator plus the plotting stack. It is not free, but it is a one-time cost.

### `__all__`

`["GeneticTrainer", "TrainingConfig", "GenerationStats", "ReinforceTrainer", "RLConfig", "EpochStats", "save_run"]`.

| Name | What it is | You use it for |
| --- | --- | --- |
| `TrainingConfig` | Dataclass of every genetic-algorithm knob (population, generations, rounds, elites, tournament, crossover, mutation, workers, seed, fitness weights, validation games, telemetry) | Describing a GA run |
| `GeneticTrainer` | Builds a population, evaluates it by playing games, validates the champion on fixed seeds, breeds the next generation | Running a GA run, saving and loading champions |
| `GenerationStats` | One record per generation (best, mean, worst and validation fitness, champion genome, seconds, cumulative seconds, telemetry) | Plotting progress, picking the best generation |
| `RLConfig` | Dataclass of every policy-gradient knob (epochs, episodes, learners, learning rates, entropy bonus, value network, validation, gradient clipping, workers, seed) | Describing an RL run |
| `ReinforceTrainer` | Collects episodes, takes a gradient step on the policy and value networks, validates greedily on fixed seeds | Running an RL run, saving the policy |
| `EpochStats` | One record per epoch (losses, entropy, returns, survival, win and kill rates, seconds, genome, telemetry) | Plotting progress |
| `save_run` | Writes `config.json`, `history.json`, `champion.json` and a `plots/` folder for either trainer | Keeping a run's results |

Each is documented in full on its own page.

### How a run flows

```
SimulationConfig + Scenario --> GeneticTrainer / ReinforceTrainer --> Game (one per job)
TrainingConfig / RLConfig        population or policy                 brain_factory places the genome
                                 evaluate() / _collect() --------->   Game.run()
                                 fitness_of() / reward per tick <--   placement, kills, days / hooks
                                 history: GenerationStats / EpochStats
                                 champion --> save_champion() / save_policy() --> JSON
                                 save_run(trainer, method, name) --> results/<name>_<timestamp>/
```

`GeneticTrainer.load_champion(path)` reads either kind of champion file back (the RL file has the same keys plus a few extras). Its dictionary feeds `create_brain` or a `TributeSpec.genome` in a scenario roster.

## How to use it / experiment

**The genetic algorithm in three lines.**

```python
from hunger_games.config import SimulationConfig
from hunger_games.training import GeneticTrainer, TrainingConfig

trainer = GeneticTrainer(SimulationConfig(width=60, height=60, max_days=6), TrainingConfig(brain_name="voting", generations=5, seed=0))
history = trainer.run()
print(history[-1].best_fitness, history[-1].val_fitness)
```

**Policy gradient in three lines.**

```python
from hunger_games.training import ReinforceTrainer, RLConfig

trainer = ReinforceTrainer(SimulationConfig(width=60, height=60, max_days=6), RLConfig(epochs=5, episodes_per_epoch=2, seed=0))
history = trainer.run()
print(history[-1].val_return, history[-1].entropy)
```

**Keep the results.** `save_run(trainer, "genetic", "my_ga")` or `save_run(trainer, "reinforce", "my_rl")` writes a folder under `results/`. See [runs.md](runs.md).

**From the command line.** `python experiments/run_ga.py` and `python experiments/run_rl.py` wrap the two snippets above with flags. See [../experiments/run_ga.md](../experiments/run_ga.md) and [../experiments/run_rl.md](../experiments/run_rl.md).

**From the dashboard.** `python -m hunger_games.ui`, open the Train tab, pick the method, set the knobs and press Train. `Session.start_training` builds the matching trainer on the painted map in a background thread; `Session.save_training_run` calls `save_run`.

**Add a third trainer.** Create `hunger_games/training/my_trainer.py`, import its class here, and append its name to `__all__`. Give it the shared shape (`config`, `history`, `history_rows()`, `champion`, `champion_brain()`, `run()`, `stop()`) plus a settings dataclass stored on the trainer, and `save_run` will need only a small change to know which save method to call.

## Gotchas

- `TrainingConfig(brain_name="random")` raises `ValueError("The 'random' brain has no genome to train")` inside `GeneticTrainer.__init__`, because the random brain's genome is empty. Only `"voting"` and `"neural"` can be evolved. `ReinforceTrainer` always trains the neural brain and has no `brain_name` at all.
- `workers > 1` in either config uses `multiprocessing`. On macOS the start method is `spawn`, so your script needs an `if __name__ == "__main__":` guard and must be a file, not the interactive prompt. Details in [genetic.md](genetic.md).
- `from hunger_games.training import *` gives only the seven names in `__all__`. `fitness_of`, `play_evaluation_game`, `play_validation_game` and `play_rl_episode` must be imported from their submodules.
- Nothing in this package is imported by `hunger_games/__init__.py`, so `import hunger_games` alone does not give you `hunger_games.training`. Import it explicitly.
- Importing this package imports matplotlib through `runs.py`. On a headless machine, set `MPLBACKEND=Agg` if you see backend errors.
