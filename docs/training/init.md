# `__init__.py`

**Source:** [hunger_games/training/__init__.py](../../hunger_games/training/__init__.py)
**Depends on:** [training/common.py](common.md) (`Curriculum`, `CurriculumConfig`, `EventLog`, `IterationStats`, `LearnerSpec`, `SystemMonitor`); [training/genetic.py](genetic.md) (`GenerationStats`, `GeneticTrainer`, `TrainingConfig`); [training/imitation.py](imitation.md) (`ImitationConfig`, `ImitationStats`, `ImitationTrainer`); [training/neat.py](neat.md) (`NeatTrainer`, `NeatTrainerConfig`); [training/ppo.py](ppo.md) (`PPOConfig`, `PPOTrainer`); [training/reinforce.py](reinforce.md) (`EpochStats`, `ReinforceTrainer`, `RLConfig`); [training/runs.py](runs.md) (`save_run`)
**Used by:** [hunger_games/ui/session.py](../ui/session.md) (every trainer class, `Curriculum`, `CurriculumConfig`, `SystemMonitor`, `save_run`); [hunger_games/ui/app.py](../ui/app.md) (`CurriculumConfig`, `ImitationConfig`, `NeatTrainerConfig`, `PPOConfig`, `RLConfig`, `TrainingConfig` back the Train tab's controls); [experiments/run_ga.py](../experiments/run_ga.md) (`GeneticTrainer`, `TrainingConfig`, `save_run`); [experiments/run_rl.py](../experiments/run_rl.md) (`ReinforceTrainer`, `RLConfig`, `save_run`); `tests/test_methods.py`; `tests/test_recorder_training.py`; `tests/test_research.py`; `tests/test_ui_session.py`; `tests/test_imitation.py`; `tests/test_feed.py`; `tests/test_comparison.py`

## Purpose

The `training` package is where brains get better. It holds five trainers, the pieces they share, and one writer:

- [common.py](common.md): the shared `IterationStats`, the `EventLog`, the opponent `Curriculum`, the `SystemMonitor`, and `LearnerSpec` for worker processes.
- [imitation.py](imitation.md): behaviour cloning. It records the voting brain's decisions and trains the neural brain to predict them, so the network starts with working instincts. Run this first.
- [genetic.py](genetic.md): a genetic algorithm over flat genomes. By default each genome plays as the learner against voting opponents and is scored by episode return; `opponents="self"` restores the population tournament.
- [neat.py](neat.md): NEAT, which evolves the shape of the network as well as its weights, in species.
- [reinforce.py](reinforce.md): REINFORCE with a value baseline. It trains the neural brain by policy gradient, scoring every action with the reward weights from [../config.md](../config.md).
- [ppo.py](ppo.md): PPO, REINFORCE's machinery with a clipped, multi-pass update and GAE.
- [runs.py](runs.md): `save_run`, which writes any trainer's results to a timestamped folder with one PNG per chart.

Every trainer trains one learner against voting opponents, accepts `initial_genome` (a warm start) and `curriculum`, and records the same `IterationStats` per iteration. That is what lets the dashboard, `save_run` and the method comparison in [../research/comparison.md](../research/comparison.md) treat all five alike.

This `__init__.py` is a thin front door. It re-exports the public names so callers can write

```python
from hunger_games.training import ImitationTrainer, PPOTrainer, NeatTrainer, Curriculum, CurriculumConfig, save_run
```

instead of reaching into the submodules. The docstring names who drives the package: the dashboard's Train tab and the scripts in `experiments/`.

## Concepts you need

**Package front door.** Python runs `__init__.py` when you import the package. Anything imported here becomes an attribute of `hunger_games.training`. The same pattern is used in [../brain/init.md](../brain/init.md) and [../init.md](../init.md).

**`__all__`.** The list of names that `from hunger_games.training import *` brings in. It doubles as a statement of the public API.

**Five ways to learn.** Imitation needs a teacher to copy and works only for the neural brain, because it trains by backpropagation. The genetic algorithm needs only a score per game, so it works for any brain with a `genome()` / `set_genome()` pair (the voting brain's eight genes or the neural brain's weights). NEAT evolves its own genome type and its own brain. REINFORCE and PPO need a probability for each action and a reward after each tick, so they work only for the neural brain.

**The shared shape.** Every trainer has `config`, a `settings` property returning its own dataclass, `run()`, `stop()`, `step()` returning an `IterationStats`, `history` (its own record type) and `history_rows()`, `learning_history` (the shared records), `events` (an `EventLog`), `champion`, `champion_brain()`, `learner_spec()`, `champion_spec()` and `save_champion()`. `save_run` and `Session.start_training` rely on exactly this.

**Warm start.** Every trainer's constructor takes `initial_genome`. Imitation loads it into the student, REINFORCE and PPO into the policy, the genetic trainer seeds its population with it plus close relatives, and NEAT clones it (a dictionary genome) into a population.

**Where training sits.** The simulator is layered: `config` describes settings, `game` plays one game, `runner` plays batches and writes CSVs, `research` measures behaviour and draws charts, and `training` plays batches *to improve a brain*. Training depends on `game` and `research`; nothing in `game` depends on training, except that `Game` knows how to build a `NeatBrain` from a dictionary genome on a roster.

## Walkthrough

### Imports

Seven statements, in this order:

1. `from hunger_games.training.common import Curriculum, CurriculumConfig, EventLog, IterationStats, LearnerSpec, SystemMonitor`
2. `from hunger_games.training.genetic import GenerationStats, GeneticTrainer, TrainingConfig`
3. `from hunger_games.training.imitation import ImitationConfig, ImitationStats, ImitationTrainer`
4. `from hunger_games.training.neat import NeatTrainer, NeatTrainerConfig`
5. `from hunger_games.training.ppo import PPOConfig, PPOTrainer`
6. `from hunger_games.training.reinforce import EpochStats, ReinforceTrainer, RLConfig`
7. `from hunger_games.training.runs import save_run`

Importing `genetic.py` pulls in `Game`, `create_brain`, `SimulationConfig` and `BehaviorTelemetry`. `imitation.py`, `neat.py` and `ppo.py` all import `reinforce.py`. `runs.py` adds `research.experiments` and `research.plots`, which import pandas and matplotlib. So `import hunger_games.training` loads most of the simulator plus the plotting stack. It is not free, but it is a one-time cost. `build_learner` and `learner_ids` from `common.py` are not re-exported.

### `__all__`

Twenty-one names:

| Name | What it is | You use it for |
| --- | --- | --- |
| `Curriculum` | Tracks the opponent stage and promotes on a score threshold or a timeout | Growing the opposition during a run |
| `CurriculumConfig` | Dataclass of the ladder (`opponents`, `threshold`, `window`, `max_iterations_per_stage`, `enabled`) | Describing a curriculum |
| `EventLog` | Timestamped one-line events with `add` and `tail` | Reading what a trainer did |
| `IterationStats` | The per-iteration record every trainer fills | Plotting any method on the same axes |
| `LearnerSpec` | A learner's kind and genome, rebuildable in a worker | Tournaments, warm starts across methods |
| `SystemMonitor` | CPU and memory readings via psutil | The dashboard's system panel |
| `NeatTrainer` | Evolves NEAT genomes in species against voting opponents | Running NEAT |
| `NeatTrainerConfig` | Dataclass of every NEAT trainer knob, including a nested `NeatConfig` | Describing a NEAT run |
| `PPOTrainer` | REINFORCE's machinery with the clipped, multi-pass update | Running PPO |
| `PPOConfig` | `RLConfig` plus `clip_ratio`, `update_epochs`, `minibatch_size`, `gae_lambda` | Describing a PPO run |
| `ImitationTrainer` | Records teacher games, trains the student by cross-entropy, validates greedily | Pretraining before the other trainers |
| `ImitationConfig` | Dataclass of every imitation knob, including `winners_top` | Describing a pretraining run |
| `ImitationStats` | One record per epoch (losses, accuracies, survival, win rate, genome, telemetry, showcase) | Reading an imitation run's own history |
| `GeneticTrainer` | Builds a population, scores it against voting opponents or itself, breeds the next | Running a GA |
| `TrainingConfig` | Dataclass of every GA knob, including `opponents` and `learners_per_game` | Describing a GA run |
| `GenerationStats` | One record per generation (best, mean, worst and validation fitness, champion, timings, telemetry, showcase) | Reading a GA run's own history |
| `ReinforceTrainer` | Collects episodes, one gradient step on the policy and value networks, validates greedily | Running REINFORCE |
| `RLConfig` | Dataclass of every policy-gradient knob | Describing an RL run |
| `EpochStats` | One record per epoch (losses, entropy, returns, survival, win and kill rates, genome, telemetry, showcase) | Reading an RL run's own history |
| `save_run` | Writes `config.json`, `history.json`, `learning.json`, `events.txt`, `champion.json` and `plots/` | Keeping a run's results |

Each is documented in full on its own page.

### How a run flows

```
SimulationConfig + Scenario + Curriculum --> one of five trainers --> Game (one per job)
ImitationConfig / TrainingConfig / NeatTrainerConfig / RLConfig / PPOConfig
(optional initial_genome: a warm start)      collect / evaluate / _collect --> play_rl_episode with a LearnerSpec
                                             labels / fitness / reward per tick <-- decision hooks, tick hooks
                                             history: the trainer's own record; learning_history: IterationStats
                                             events: EventLog; curriculum.observe(mean_score)
                                             champion --> save_champion() --> JSON
                                             save_run(trainer, method, name) --> results/<name>_<timestamp>/
```

`GeneticTrainer.load_champion(path)` reads any champion file back. Array genomes become numpy arrays; a NEAT dictionary stays a dictionary. Its `genome` feeds `initial_genome` of a trainer of the matching kind, or a roster entry.

## How to use it / experiment

**The recommended flow.**

```python
from hunger_games.config import SimulationConfig
from hunger_games.training import Curriculum, CurriculumConfig, ImitationConfig, ImitationTrainer, PPOConfig, PPOTrainer

config = SimulationConfig(width=60, height=60, max_days=6)
student = ImitationTrainer(config, ImitationConfig(demonstration_games=4, epochs=10, seed=0))
student.run()
trainer = PPOTrainer(
    config, PPOConfig(epochs=20, seed=0), initial_genome=student.champion, curriculum=Curriculum(CurriculumConfig())
)
trainer.run()
last = trainer.learning_history[-1]
print(last.mean_score, last.val_score, last.opponents)
```

**Any trainer, the same loop.**

```python
from hunger_games.training import GeneticTrainer, NeatTrainer, NeatTrainerConfig, ReinforceTrainer, RLConfig, TrainingConfig

for trainer in (
    GeneticTrainer(config, TrainingConfig(generations=3, seed=0)),
    NeatTrainer(config, NeatTrainerConfig(population_size=12, generations=3, seed=0)),
    ReinforceTrainer(config, RLConfig(epochs=3, seed=0)),
):
    stats = trainer.step()
    print(type(trainer).__name__, stats.mean_score, trainer.events.tail(1))
```

**Keep the results.** `save_run(trainer, "imitation" | "genetic" | "neat" | "reinforce" | "ppo", "my_run")` writes a folder under `results/`. See [runs.md](runs.md).

**From the command line.** `python experiments/run_ga.py` and `python experiments/run_rl.py` wrap the GA and REINFORCE trainers with flags. `python experiments/run_comparison.py` trains every method under one budget and runs a tournament. See [../experiments/run_ga.md](../experiments/run_ga.md) and [../experiments/run_rl.md](../experiments/run_rl.md).

**From the dashboard.** `python -m hunger_games.ui`, open the Train tab, pick the method (imitation, genetic, neat, reinforce or ppo), tick the curriculum if you want it, and press Start. `Session.start_training` builds the matching trainer on the painted map in a background thread and calls `step()` until the iteration budget is used up or Stop is pressed.

**Add a sixth trainer.** Create `hunger_games/training/my_trainer.py`, give it the shared shape (including `learning_history`, `events`, `step()`, `learner_spec()`, `champion_spec()`), import its classes here, append the names to `__all__`, and add a branch for its method name in `training_run_plots` if it needs charts of its own. The shared learning curves come for free from `learning_history`.

## Gotchas

- `TrainingConfig(brain_name="random")` raises `ValueError("The 'random' brain has no genome to train")` inside `GeneticTrainer.__init__`, because the random brain's genome is empty. Only `"voting"` and `"neural"` can be evolved by the GA. NEAT has its own genome and cannot be evolved by the GA.
- A warm start must match the trainer. A neural genome fits imitation, REINFORCE, PPO and a neural GA; a NEAT dictionary fits only NEAT. The dashboard and the method comparison both drop a mismatched warm start.
- `workers > 1` in any config uses `multiprocessing`. On macOS the start method is `spawn`, so your script needs an `if __name__ == "__main__":` guard and must be a file, not the interactive prompt. Details in [genetic.md](genetic.md).
- `from hunger_games.training import *` gives only the names in `__all__`. `fitness_of`, `play_evaluation_game`, `play_validation_game`, `play_rl_episode`, `collect_demonstration_game`, `build_learner` and `learner_ids` must be imported from their submodules.
- The module docstring still describes the package as it was in an earlier release ("both trainers", three classes). The code exports five trainers; trust `__all__`.
- Nothing in this package is imported by `hunger_games/__init__.py`, so `import hunger_games` alone does not give you `hunger_games.training`. Import it explicitly.
- Importing this package imports matplotlib through `runs.py`. On a headless machine, set `MPLBACKEND=Agg` if you see backend errors.
