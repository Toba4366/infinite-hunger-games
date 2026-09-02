# `__init__.py`

**Source:** [hunger_games/training/__init__.py](../../hunger_games/training/__init__.py)
**Depends on:** [training/genetic.py](genetic.md) (`GenerationStats`, `GeneticTrainer`, `TrainingConfig`); [training/imitation.py](imitation.md) (`ImitationConfig`, `ImitationStats`, `ImitationTrainer`); [training/reinforce.py](reinforce.md) (`EpochStats`, `ReinforceTrainer`, `RLConfig`); [training/runs.py](runs.md) (`save_run`)
**Used by:** [hunger_games/ui/session.py](../ui/session.md) (`GeneticTrainer`, `ImitationConfig`, `ImitationTrainer`, `ReinforceTrainer`, `RLConfig`, `TrainingConfig`, `save_run`); [hunger_games/ui/app.py](../ui/app.md) (`ImitationConfig`, `RLConfig`, `TrainingConfig` back the Train tab's controls); [experiments/run_ga.py](../experiments/run_ga.md) (`GeneticTrainer`, `TrainingConfig`, `save_run`); [experiments/run_rl.py](../experiments/run_rl.md) (`ReinforceTrainer`, `RLConfig`, `save_run`); `tests/test_recorder_training.py`; `tests/test_research.py`; `tests/test_ui_session.py`; `tests/test_imitation.py`

## Purpose

The `training` package is where brains get better. It holds three trainers and one writer:

- [imitation.py](imitation.md): behaviour cloning. It records the voting brain's decisions and trains the neural brain to predict them, so the network starts with working instincts. Run this first.
- [genetic.py](genetic.md): a genetic algorithm. It keeps a population of genomes, plays them against each other, and breeds the winners. It scores whole games.
- [reinforce.py](reinforce.md): REINFORCE with a value baseline. It trains the neural brain by policy gradient, scoring every single action with the reward weights from [../config.md](../config.md).
- [runs.py](runs.md): `save_run`, which writes any trainer's results to a timestamped folder with one PNG per chart.

Both the genetic and the policy-gradient trainer accept an `initial_genome` and can warm-start from an imitation champion. That is the recommended flow: imitation first, then evolve or reinforce from the result.

This `__init__.py` is a thin front door. It re-exports the ten public names so callers can write

```python
from hunger_games.training import ImitationTrainer, ImitationConfig, GeneticTrainer, TrainingConfig, ReinforceTrainer, RLConfig, save_run
```

instead of reaching into the four submodules. The docstring states who drives the package: the dashboard's Train tab and the scripts in `experiments/`.

## Concepts you need

**Package front door.** Python runs `__init__.py` when you import the package. Anything imported here becomes an attribute of `hunger_games.training`. The same pattern is used in [../brain/init.md](../brain/init.md) and [../init.md](../init.md).

**`__all__`.** The list of names that `from hunger_games.training import *` brings in. It doubles as a statement of the public API.

**Three ways to learn.** Imitation needs a teacher to copy and works only for the neural brain, because it trains by backpropagation. A genetic algorithm needs only a score per game, so it works for any brain with a `genome()` / `set_genome()` pair (the voting brain's eight genes or the neural brain's 5872 weights). Policy gradient needs a probability for each action and a reward after each tick, so it also works only for the neural brain. All three share the same outward shape: `config`, `settings`, `run`, `stop`, `history`, `history_rows`, `champion`, `champion_brain` and `save_champion`. That shared shape is what lets the dashboard and `save_run` treat them alike.

**Warm start.** Every trainer's constructor takes `initial_genome`. The imitation trainer loads it into the student, the REINFORCE trainer loads it into the policy, and the genetic trainer seeds its population with it plus close relatives.

**Where training sits.** The simulator is layered: `config` describes settings, `game` plays one game, `runner` plays batches and writes CSVs, `research` measures behaviour and draws charts, and `training` plays batches *to improve a brain*. Training depends on `game` and `research`; nothing in `game` depends on training. A trained genome is loaded back through the ordinary roster and config routes with no special cases.

## Walkthrough

### Imports

Four lines, one per submodule, in this order:

1. `from hunger_games.training.genetic import GenerationStats, GeneticTrainer, TrainingConfig`
2. `from hunger_games.training.imitation import ImitationConfig, ImitationStats, ImitationTrainer`
3. `from hunger_games.training.reinforce import EpochStats, ReinforceTrainer, RLConfig`
4. `from hunger_games.training.runs import save_run`

Importing `genetic.py` pulls in `Game`, `create_brain`, `SimulationConfig` and `BehaviorTelemetry`. Importing `imitation.py` adds `Adam`, `NeuralBrain`, `softmax` and, through it, `reinforce.py` (for `play_rl_episode`). Importing `runs.py` adds `research.experiments` and `research.plots`, which import pandas and matplotlib. So `import hunger_games.training` loads most of the simulator plus the plotting stack. It is not free, but it is a one-time cost.

### `__all__`

`["ImitationTrainer", "ImitationConfig", "ImitationStats", "GeneticTrainer", "TrainingConfig", "GenerationStats", "ReinforceTrainer", "RLConfig", "EpochStats", "save_run"]`.

| Name | What it is | You use it for |
| --- | --- | --- |
| `ImitationConfig` | Dataclass of every imitation knob (teacher, teacher chaos, demonstration games, epochs, batch size, learning rate, validation split and games, learners, workers, seed, showcase) | Describing a pretraining run |
| `ImitationTrainer` | Records teacher games, trains the student by cross-entropy, validates on held-out demonstrations and greedy games | Pretraining a network before the other trainers |
| `ImitationStats` | One record per epoch (train and validation loss and accuracy, validation survival and win rate, seconds, genome, telemetry, showcase) | Plotting progress, picking the champion |
| `TrainingConfig` | Dataclass of every genetic-algorithm knob (population, generations, rounds, elites, tournament, crossover, mutation, workers, seed, fitness weights, validation games, telemetry, showcase) | Describing a GA run |
| `GeneticTrainer` | Builds a population (random or warm-started), evaluates it by playing games, validates the champion on fixed seeds, breeds the next generation | Running a GA run, saving and loading champions |
| `GenerationStats` | One record per generation (best, mean, worst and validation fitness, champion genome, seconds, cumulative seconds, telemetry, showcase) | Plotting progress, picking the best generation |
| `RLConfig` | Dataclass of every policy-gradient knob (epochs, episodes, learners, learning rates, entropy bonus, value network, validation, gradient clipping, workers, seed, showcase) | Describing an RL run |
| `ReinforceTrainer` | Collects episodes, takes a gradient step on the policy and value networks, validates greedily on fixed seeds | Running an RL run, saving the policy |
| `EpochStats` | One record per epoch (losses, entropy, returns, survival, win and kill rates, seconds, genome, telemetry, showcase) | Plotting progress |
| `save_run` | Writes `config.json`, `history.json`, `champion.json` and a `plots/` folder for any trainer | Keeping a run's results |

Each is documented in full on its own page.

### How a run flows

```
SimulationConfig + Scenario --> ImitationTrainer / GeneticTrainer / ReinforceTrainer --> Game (one per job)
ImitationConfig / TrainingConfig / RLConfig      student, population or policy       brain_factory places the genome
(optional initial_genome: a warm start)          collect() / evaluate() / _collect() -->   Game.run()
                                                 labels / fitness_of() / reward per tick <--   decision hooks, placement, tick hooks
                                                 history: ImitationStats / GenerationStats / EpochStats
                                                 champion --> save_champion() --> JSON
                                                 save_run(trainer, method, name) --> results/<name>_<timestamp>/
```

`GeneticTrainer.load_champion(path)` reads any of the three champion files back (they share the same keys; the RL and imitation files add a few extras). Its `genome` feeds `initial_genome`, `create_brain` or a `TributeSpec.genome` in a scenario roster.

## How to use it / experiment

**The recommended flow in a few lines.**

```python
from hunger_games.config import SimulationConfig
from hunger_games.training import GeneticTrainer, ImitationConfig, ImitationTrainer, TrainingConfig

config = SimulationConfig(width=60, height=60, max_days=6)
student = ImitationTrainer(config, ImitationConfig(demonstration_games=4, epochs=10, seed=0))
student.run()
trainer = GeneticTrainer(config, TrainingConfig(brain_name="neural", generations=5, mutation_scale=0.02, seed=0), initial_genome=student.champion)
history = trainer.run()
print(history[-1].best_fitness, history[-1].val_fitness)
```

**The genetic algorithm alone.**

```python
trainer = GeneticTrainer(config, TrainingConfig(brain_name="voting", generations=5, seed=0))
history = trainer.run()
```

**Policy gradient alone, or from the student.**

```python
from hunger_games.training import ReinforceTrainer, RLConfig

trainer = ReinforceTrainer(config, RLConfig(epochs=5, episodes_per_epoch=2, seed=0), initial_genome=student.champion)
history = trainer.run()
print(history[-1].val_return, history[-1].entropy)
```

**Keep the results.** `save_run(trainer, "imitation", "my_student")`, `save_run(trainer, "genetic", "my_ga")` or `save_run(trainer, "reinforce", "my_rl")` writes a folder under `results/`. See [runs.md](runs.md).

**From the command line.** `python experiments/run_ga.py` and `python experiments/run_rl.py` wrap the GA and RL snippets with flags. See [../experiments/run_ga.md](../experiments/run_ga.md) and [../experiments/run_rl.md](../experiments/run_rl.md). There is no imitation script yet; use the snippet above or the dashboard.

**From the dashboard.** `python -m hunger_games.ui`, open the Train tab, pick the method (imitation, genetic or reinforce), set the knobs and press Train. `Session.start_training` builds the matching trainer on the painted map in a background thread, passing the last champion as `initial_genome` when "start from the current champion" is ticked; `Session.save_training_run` calls `save_run`.

**Add a fourth trainer.** Create `hunger_games/training/my_trainer.py`, import its class here, and append its names to `__all__`. Give it the shared shape (`config`, a `settings` property returning its dataclass, `history`, `history_rows()`, `champion`, `champion_brain()`, `save_champion()`, `run()`, `stop()`), then add a branch for its method name in `training_run_plots` so `save_run` knows which charts to draw.

## Gotchas

- `TrainingConfig(brain_name="random")` raises `ValueError("The 'random' brain has no genome to train")` inside `GeneticTrainer.__init__`, because the random brain's genome is empty. Only `"voting"` and `"neural"` can be evolved. `ReinforceTrainer` and `ImitationTrainer` always train the neural brain and have no `brain_name` at all.
- A warm start must match the brain. An imitation champion is a neural genome; passing it to a `GeneticTrainer` that evolves the voting brain raises `ValueError` from `set_genome`. The dashboard drops the warm start in that case.
- `workers > 1` in any config uses `multiprocessing`. On macOS the start method is `spawn`, so your script needs an `if __name__ == "__main__":` guard and must be a file, not the interactive prompt. Details in [genetic.md](genetic.md).
- `from hunger_games.training import *` gives only the ten names in `__all__`. `fitness_of`, `play_evaluation_game`, `play_validation_game`, `play_rl_episode` and `collect_demonstration_game` must be imported from their submodules.
- Nothing in this package is imported by `hunger_games/__init__.py`, so `import hunger_games` alone does not give you `hunger_games.training`. Import it explicitly.
- Importing this package imports matplotlib through `runs.py`. On a headless machine, set `MPLBACKEND=Agg` if you see backend errors.
