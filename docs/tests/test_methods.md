# `test_methods.py`

**Source:** [tests/test_methods.py](../../tests/test_methods.py)
**Tests:** [../brain/neat.md](../brain/neat.md) (`InnovationTracker`, `NeatBrain`, `NeatConfig`, `NeatGenome`), [../training/common.md](../training/common.md) (`Curriculum`, `CurriculumConfig`, `SystemMonitor`, the shared `IterationStats`), [../training/init.md](../training/init.md) (every trainer and settings class), [../training/runs.md](../training/runs.md) (`save_run`), with [../brain/neural.md](../brain/neural.md) (`MENU_SIZE`), [../perception.md](../perception.md) (`VECTOR_SIZE`), [../config.md](../config.md) (`SimulationConfig`) and [../game.md](../game.md) (`Game`)

## Purpose

Version 0.6.0 gave every trainer the same outward shape: `step()` returns an `IterationStats`, `learning_history` collects them, `events` logs what happened, and `learner_spec()` describes the learner for a worker process. It also added NEAT, PPO and the curriculum. This file checks those additions with the smallest games that still run.

The first test exercises a NEAT genome on its own: it evaluates, grows, stays feed-forward, crosses over, measures distance, survives JSON, and drives a brain in a real game. The second pins down the curriculum's promotion rule. The third is the contract test: all five trainers produce one well-formed `IterationStats` per step and save a run folder with the shared curves. The fourth proves the genetic algorithm follows the curriculum's roster sizes and that imitation can keep winners' decisions only. The fifth checks the system monitor returns the keys the dashboard shows.

Without these, a method that filled `IterationStats` with a different shape would only break when the dashboard or the comparison tried to plot it.

## Concepts you need

**Test discovery.** Five `test_*` functions. `small` is a helper.

**The `tmp_path` fixture.** The third test takes `tmp_path`; pytest gives it a fresh temporary folder for the run folders.

**NEAT genome.** Nodes and connections, each connection with an innovation number. `minimal` wires every input and a bias to every output. `mutate` can nudge weights, add a connection or add a node; `crossover` lines genes up by innovation number; `distance` is the compatibility measure species use. See [../brain/neat.md](../brain/neat.md).

**Feed-forward by depth.** `depths()` assigns every node a depth; a valid genome has every enabled connection going from a shallower node to a deeper one, so evaluation never loops.

**The curriculum.** `observe(mean_score)` returns `True` on promotion. Promotion happens when the last `window` scores average at least `threshold`, or when `max_iterations_per_stage` iterations have passed in the stage. See [../training/common.md](../training/common.md).

**Running a subset.** `python -m pytest tests/test_methods.py -k curriculum` runs the two curriculum tests.

## Walkthrough

### `small() -> SimulationConfig`

```python
def small() -> SimulationConfig:
    return SimulationConfig(seed=9, width=40, height=40, max_days=3)
```

A 40 by 40 arena capped at 3 days. Everything else is the default: 24 tributes, voting opponents, a 64 by 32 network with 5872 weights.

### `test_neat_genome_grows_and_round_trips()`

**Setup.** `rng = np.random.default_rng(0)`, a fresh `InnovationTracker`, and `NeatConfig(add_node_rate=1.0, add_connection_rate=1.0)` so every mutation is guaranteed to add a node and try a connection. `genome = NeatGenome.minimal(VECTOR_SIZE, MENU_SIZE, rng, config, tracker)`: 50 inputs, 16 outputs.

**`assert genome.forward(np.zeros(VECTOR_SIZE)).shape == (MENU_SIZE,)`.** A minimal genome evaluates to one score per menu item. A failure means the forward pass does not produce the output layer in order.

**`assert genome.enabled_count == (VECTOR_SIZE + 1) * MENU_SIZE`.** 51 sources (50 inputs plus the bias) times 16 outputs is 816 enabled connections. A failure means `minimal` is not fully connected or the bias is missing.

**Five mutations, then `assert genome.hidden_count >= 1`.** With `add_node_rate=1.0` every `mutate` splits a connection with a new hidden node. A failure means add-node is not firing or the node is not counted as hidden.

**`assert all(depth[c.src] < depth[c.dst] for c in genome.connections if c.enabled)`.** After the mutations the genome is still feed-forward. A failure means a mutation created a cycle or a backward connection, which would make `forward` wrong.

**`child = genome.crossover(other, rng)` then `assert child.forward(np.ones(VECTOR_SIZE)).shape == (MENU_SIZE,)`.** Crossing the grown genome with a fresh minimal one still gives a network that evaluates. A failure means crossover mixes genes into something that cannot be evaluated.

**`assert genome.distance(other, config) > 0.0`.** The grown genome and the minimal one differ by excess and disjoint genes and by weights, so their compatibility distance is positive. Zero would mean species could never form.

**`rebuilt = NeatGenome.from_dict(genome.to_dict())` then `assert np.allclose(rebuilt.forward(ones), genome.forward(ones))`.** The JSON round trip keeps every node, connection, weight and enabled flag. A failure means champion files or `LearnerSpec` dictionaries would rebuild a different network.

**`brain = NeatBrain(genome, chaos=0.0)`, a `Game(small())`, `perception = game.players[0].perceive(...)`, then `assert brain.decide(perception, rng) is not None`.** The brain turns a real perception of a real arena into an action. A failure means the brain and the perception vector disagree about their size.

**`assert "NEAT" in brain.describe()`.** The description names the brain kind, which the dashboard's inspector shows.

### `test_curriculum_promotes_on_score_or_timeout()`

**Setup.** `Curriculum(CurriculumConfig(opponents=(1, 3, 7), threshold=1.0, window=2, max_iterations_per_stage=4))`.

**`assert curriculum.opponents == 1`.** Stage 0 is the first entry.

**`assert not curriculum.observe(0.0)`.** One score is fewer than `window`, so no promotion.

**`assert curriculum.observe(2.0) is True and curriculum.opponents == 3`.** The window is now `[0.0, 2.0]`, mean 1.0, which reaches the threshold. Promotion to stage 1. A failure means the mean test or the window length test is wrong.

**`assert not any(curriculum.observe(0.0) for _ in range(3))`.** Three poor iterations in stage 1: the mean never reaches 1.0 and three is fewer than `max_iterations_per_stage`.

**`assert curriculum.observe(0.0) is True and curriculum.opponents == 7 and curriculum.finished`.** The fourth iteration times the stage out. Stage 2 is the last, so `finished` is true. A failure means the timeout is off by one or `finished` does not recognise the last stage.

**`assert Curriculum(CurriculumConfig(enabled=False)).opponents == 23`.** A disabled curriculum sits on the hardest stage, the last of the default `(1, 3, 7, 11, 23)`.

### `test_every_method_produces_the_shared_iteration_stats(tmp_path)`

**Setup.** Five trainers on `small()`, each with the smallest settings that still play a game and validate:

| Method | Settings |
| --- | --- |
| imitation | `ImitationConfig(demonstration_games=1, epochs=1, validation_games=1, learners_per_game=4, seed=0)` |
| genetic | `TrainingConfig(brain_name="neural", population_size=6, generations=1, rounds_per_generation=1, validation_games=1, seed=0)` |
| neat | `NeatTrainerConfig(population_size=6, generations=1, validation_games=1, learners_per_game=4, seed=0)` |
| reinforce | `RLConfig(epochs=1, episodes_per_epoch=1, learners_per_game=4, validation_games=1, seed=0)` |
| ppo | `PPOConfig(epochs=1, episodes_per_epoch=1, learners_per_game=4, validation_games=1, seed=0, update_epochs=2)` |

Every assertion carries `, method` so a failure names the trainer.

**`stats = trainer.step()` then `assert stats.iteration == 0 and len(trainer.learning_history) == 1`.** The first step is iteration 0 and is recorded exactly once. A failure means a trainer numbers from 1, records twice, or does not append to `learning_history`.

**`assert np.isfinite(stats.mean_score) and stats.mean_length >= 0 and 0.0 <= stats.win_rate <= 1.0`.** The three headline numbers are sane. A `NaN` mean score would poison every overlay chart; a negative length or a win rate outside 0 to 1 means the wrong quantity was stored.

**`assert trainer.events.events`.** The step logged at least one event for the dashboard's event monitor.

**`assert trainer.learner_spec().kind in ("neural", "neat")`.** The learner can be described for a worker process. A failure means the spec kind is misspelled and `build_learner` would fall through to the neural branch with the wrong genome.

**`folder = save_run(trainer, method, f"t_{method}", tmp_path)`.** The run folder for this method.

**`assert (folder / "learning.json").exists() and (folder / "events.txt").exists()`.** The shared history and the event log were written.

**`assert (folder / "plots" / "score.png").exists()`.** The first shared learning-curve chart exists, so `learning_curve_plots` received rows with the expected keys. A failure usually means a key was renamed in `IterationStats.to_row()`.

### `test_genetic_curriculum_and_winners_only_demonstrations()`

**Setup, part one.** `Curriculum(CurriculumConfig(opponents=(2, 5), threshold=99.0, window=1, max_iterations_per_stage=1))`: an unreachable threshold and a one-iteration timeout, so every iteration promotes. `GeneticTrainer(small(), TrainingConfig(brain_name="neural", population_size=4, generations=2, rounds_per_generation=1, validation_games=0, learners_per_game=2, seed=0), curriculum=curriculum)`, then `trainer.run()`.

**`assert trainer.learning_history[0].opponents == 2 and trainer.learning_history[1].opponents == 5`.** Generation 0 played against 2 opponents, generation 1 against 5. A failure means the genetic trainer does not record the curriculum stage, or promotes at the wrong moment.

**`assert trainer.config.num_players == 2 + 5`.** The roster was resized to learners plus opponents. A failure means `_apply_curriculum` did not rebuild the config.

**Setup, part two.** Two imitation trainers on `small()` with `demonstration_games=1, epochs=1, validation_games=0, seed=0`; `winners` adds `winners_top=3`. The same seed gives both the same teacher game.

**`assert winners.collect() < everyone.collect()`.** Keeping only the decisions of the top three finishers gives fewer demonstrations than keeping everyone's. A failure means the placement filter is not applied or keeps every tribute.

### `test_system_monitor_reads()`

**`reading = SystemMonitor().read()` then `assert set(reading) == {"cpu_percent", "memory_mb", "memory_percent", "gpu"}`.** The dictionary has exactly the keys the dashboard's system line formats, with or without `psutil` installed. A failure means a key was renamed on one branch of `read` but not the other.

## How to use it / experiment

- Run the file alone: `python -m pytest tests/test_methods.py -q`. The contract test is the slow one because it plays five methods' games.
- Adding a sixth method: add it to the `trainers` dictionary in the third test with its smallest settings. If it fills `IterationStats` and saves a run folder, it will work in the dashboard and in `MethodComparison`.
- To see the genome grow, print `genome.hidden_count` and `genome.enabled_count` after each of the five mutations.

## Gotchas

- **The mutation rates are 1.0 on purpose.** With the default `add_node_rate=0.03`, five mutations would rarely add a node and the `hidden_count` assertion would be flaky.
- **`learners_per_game=4` on a 24-tribute roster** puts learners in slots 0, 6, 12 and 18. The genetic curriculum test uses 2 learners, so its rosters are 4 and 7 tributes, not 24.
- **`validation_games=0`** in the fourth test means `val_score` is 0.0 and no showcase is recorded; that is fine because only `opponents` and `num_players` are checked.
- **The system monitor test passes without `psutil`**, because the fallback branch returns the same four keys with zero readings.
