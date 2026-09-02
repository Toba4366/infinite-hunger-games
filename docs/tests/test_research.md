# `test_research.py`

**Source:** [tests/test_research.py](../../tests/test_research.py)
**Tests:** `hunger_games/brain/mlp.py` (`MLP`, `Adam`), `hunger_games/research/telemetry.py` (`BehaviorTelemetry`, `ACTION_NAMES`), `hunger_games/research/plots.py` (`behaviour_plots`), `hunger_games/research/experiments.py` (`Sweep`, `SweepConfig`, `set_field`), `hunger_games/training/reinforce.py` (`ReinforceTrainer`, `RLConfig`), `hunger_games/training/genetic.py` (`GeneticTrainer`, `TrainingConfig`), `hunger_games/training/runs.py` (`save_run`), with [../game.md](../game.md) (`Game`) and [../config.md](../config.md) (`SimulationConfig`) used to play the games

## Purpose

The research layer is everything that measures *how* tributes behave rather than just who won. It has four parts. `MLP` is a plain numpy neural network with backpropagation, and `Adam` is the optimiser that updates it. `BehaviorTelemetry` plugs into a game's hooks and tallies every decision against thirst, hunger, health, danger and position. `plots.py` turns those tallies into one PNG per chart. `Sweep` changes one config field across several values and plays a batch of games per value. The two trainers, genetic and REINFORCE, use all of it and `save_run` writes their results to a run folder.

This file checks that each part works on its own and that they fit together. The gradient test is the one that matters most: if `backward` is wrong, reinforcement learning still runs and still prints numbers, but it learns nothing. The other tests are integration tests. They run a tiny trainer or sweep end to end and check the folder it wrote.

## Concepts you need

**Finite differences.** The derivative of a function at a point is the slope of the line through two nearby points. Nudge one weight up by `eps`, measure the loss, nudge it down by `eps`, measure again, and `(plus - minus) / (2 * eps)` is the numeric gradient. It is slow, one forward pass per weight, so it is only for checking. Backpropagation computes every gradient in one pass and must agree with it.

**Why divide by `out.size`.** The test's loss is `np.mean((out - target) ** 2)`. The derivative of a mean of squares with respect to one output is `2 * (out - target) / N`, where `N` is the number of outputs summed over the batch. `MLP.backward` sums gradients over the batch and never divides, so the caller passes the already-divided gradient. That makes the analytic gradient match the numeric gradient of the *mean* loss.

**Game hooks.** `Game.decision_hooks` is a list of functions called as `hook(player, perception, action)` after every decision. `Game.tick_hooks` is called as `hook(game)` at the end of every tick. Telemetry registers one of each with `attach`. Tests can append their own lambdas to the same lists.

**The `tmp_path` fixture.** Four tests write folders. pytest gives each a fresh temporary directory that is deleted later.

**Run folders.** `make_run_dir(base, name)` creates `base/<name>_<timestamp>/plots`. Both `save_run` and `Sweep.run` use it, so the tests look for files inside the returned `Path`.

**Running a subset.** `python -m pytest tests/test_research.py -k mlp` runs the two network tests.

## Walkthrough

### `small() -> SimulationConfig`

```python
def small() -> SimulationConfig:
    return SimulationConfig(seed=7, width=50, height=50, max_days=4)
```

The same helper as in `test_recorder_training.py`: a 50 by 50 arena capped at 96 ticks. With seed 7 a plain game runs to the cap with 16 eliminations, which gives the telemetry something to count.

### `test_mlp_backward_matches_finite_differences()`

**Setup.** A batch of six inputs with five features and six targets with three values, both from `default_rng(0)`. Then, for each of the five activations, a fresh `MLP([5, 4, 3], activation, rng=default_rng(1))`: one hidden layer of four, so two weight matrices. `forward_cached` returns the outputs and the cache backprop needs. `net.backward(cache, 2.0 * (out - target) / out.size)` returns one `(grad_w, grad_b)` pair per layer.

**The loop.** For each of the two layers, weight `[1, 2]` is nudged by `eps = 1e-6` up, then down, restoring it afterwards. `plus` and `minus` are the mean squared losses at each nudge.

**`assert abs((plus - minus) / (2 * eps) - grads[layer][0][i, j]) < 1e-5, activation`.** The numeric slope must match the analytic gradient for that weight to five decimal places. The message after the comma is the activation name, so a failure names the culprit. A failure would mean an activation derivative in `initializers.py` is wrong (the classic bug is using `z` where `a` was meant, or the wrong slope for `leaky_relu`), or the chain rule in `backward` multiplies at the wrong layer.

**Why these numbers.** `eps = 1e-6` is small enough that the slope is nearly exact and large enough that floating-point noise stays below `1e-5`. Weight `[1, 2]` exists in both a 5 by 4 and a 4 by 3 matrix. Checking every activation matters because each has its own derivative function. `relu` and `leaky_relu` have a kink at zero, but with random inputs no pre-activation lands exactly there.

### `test_adam_reduces_a_simple_loss()`

**Setup.** 32 inputs with four features and a target that is a fixed linear map of them (`x @ W`). An `MLP([4, 8, 2], "tanh")` and `Adam(net, 0.01)`. Two hundred steps of forward, record the loss, backward with the same divided gradient, `optimizer.step`.

**`assert losses[-1] < losses[0] * 0.5`.** After 200 steps the loss must be under half its starting value. A linear target is easy for a tanh network, so this is a generous bar. A failure would mean Adam's update has the wrong sign, the bias correction divides by zero at `t = 0`, or the running averages are stored back into the wrong slot.

**`net.set_genome(genome * 0 + 1.0); assert (net.genome() == 1.0).all()`.** The genome round trip: write a vector of ones into every weight and bias, read it back, and every value is one. `genome * 0 + 1.0` is just a ones vector of the right length. This proves `set_genome` fills every slot, which the genetic trainer and `save_policy` rely on.

### `test_telemetry_counts_every_decision_and_merges()`

**Setup.** `Game(small())`, `BehaviorTelemetry(50, 50).attach(game)`. The telemetry needs the arena size for its 30 by 30 position heatmap. The test then appends its own hook that pushes a `1` into `decisions` on every decision, and runs the game.

**`assert sum(summary["action_counts"]) == len(decisions)`.** The telemetry saw every decision the test's hook saw. Both hooks sit in the same list, so a decision missed by one would be missed by the other; what this really checks is that `on_decision` never returns early for a tracked tribute and never drops an action kind. `ACTION_INDEX` maps every `ActionType` to a column.

**`assert summary["games"] == 1`.** `on_game_end` ran once, from `on_tick` when `game.is_over` became true.

**`assert len(summary["survival_ticks"]) == 24`.** One entry per tribute, because `tracked_ids` is `None` and the default roster has 24.

**`assert summary["death_count"] == len(game.eliminations)`.** Every death was noticed exactly once. `on_tick` walks the players each tick and uses `_dead_seen` so a corpse is counted the tick it appears and never again.

**`assert set(summary["action_names"]) == set(ACTION_NAMES)`.** The summary carries the nine action names in `ActionType` order, so the plots can label columns.

**`merged = BehaviorTelemetry.merge([summary, summary])`.** Merging a summary with itself doubles the counts. `merged["games"] == 2`, the action counts double, and `survival_ticks` is 48 long because list keys concatenate rather than add. A failure would mean a key was missing from `array_keys` or `list_keys` in `merge`, which is the bug you get when adding a new tally and forgetting the merge step.

**`json.dumps(merged)`.** No assertion, but it raises if anything in the merged dictionary is a numpy array or numpy scalar. The sweep and the run folders write summaries as JSON, so this line guards them.

### `test_reinforce_trainer_runs_and_saves(tmp_path)`

**Setup.** `ReinforceTrainer(small(), RLConfig(epochs=2, episodes_per_epoch=1, learners_per_game=4, validation_games=1, seed=0))`. Each epoch plays one training game with four learner tributes (slots 0, 6, 12 and 18, spread across the podiums) and one greedy validation game on seed 90000. `history = trainer.run()`.

**`assert len(history) == 2`.** One `EpochStats` per epoch.

**The key loop.** `history[-1].to_row()` drops the genome and telemetry and keeps the numbers. For each of `policy_loss`, `value_loss`, `entropy`, `train_return`, `val_return`, `train_survival`, `win_rate`, `seconds` and `cumulative_seconds`, the key must exist and `np.isfinite` must be true. A `nan` here usually means a division by zero in `_update` when no learner made a decision, or a `log(0)` in the entropy.

**`assert 0.0 < history[-1].entropy <= np.log(16) + 1e-6`.** Entropy is measured in nats over the 16-item action menu. The maximum for 16 equally likely items is `log(16)`, about 2.77. A fresh policy sits near that; it must be above zero because the policy samples at temperature 1 and never collapses in two epochs.

**`assert trainer.champion.size == trainer.policy.parameter_count`.** The champion is the genome with the best validation return and must be the full 1088-number policy (50 inputs, 16 hidden, 16 outputs).

**`folder = save_run(trainer, "reinforce", "test_rl", tmp_path)`.** Writes `config.json`, `history.json`, `champion.json` (via `save_policy`, which also stores the value network) and `plots/`. The test checks `history.json` and `champion.json` exist, that at least 10 PNGs landed in `plots/` (reward, losses, entropy, survival, win and kill rate, timing, three over-training charts and the twelve behaviour charts), and that `reward.gif`, the growing-curve animation, exists.

**Why two epochs.** The `over training` charts need at least two points to draw a line, and `previous` genome comparisons need two entries. Two is the smallest history that exercises everything.

### `test_genetic_trainer_validation_telemetry_and_run_folder(tmp_path)`

**Setup.** `GeneticTrainer(small(), TrainingConfig(brain_name="voting", population_size=24, generations=2, rounds_per_generation=1, validation_games=1, seed=0))`. Eight-gene voting genomes, one evaluation game per generation, plus one validation game where the champion drives six tributes against the config's default brain.

**`assert all("val_fitness" in row and "cumulative_seconds" in row for row in rows)`.** `history_rows()` includes the two newer columns. The run-folder plots read `val_fitness` for the validation line, so a missing key would crash `training_run_plots`.

**`assert trainer.history[-1].telemetry["games"] >= 1`.** `collect_telemetry` defaults to `True`, so each evaluation game returns a summary and the generation merges them. A failure would mean `play_evaluation_game` stopped attaching telemetry.

**`assert trainer.previous_champion() is not None`.** With two generations there is a generation before the latest. The dashboard uses this to highlight which genes changed.

**`folder = save_run(trainer, "genetic", "test_ga", tmp_path)`.** For the genetic method the plots are `fitness.png` (best, mean and validation lines) and `fitness.gif`. Both must exist.

### `test_sweep_sets_nested_fields_and_writes_results(tmp_path)`

**`assert set_field(config, "terrain.water_threshold", 0.4).terrain.water_threshold == 0.4`.** A dotted path walks into the nested `TerrainConfig` and sets the last part.

**`assert config.terrain.water_threshold == 0.25`.** The original was deep-copied, not edited. Without the copy, every value in a sweep would leak into the next.

**Setup.** `Sweep(config, SweepConfig(name="t", parameter="chaos", values=[0.0, 0.5], games_per_value=2, results_dir=tmp_path))`. Two values, two games each, all seeded from `seed=1000` so both values play the same games. `folder = sweep.run()`.

**`assert (folder / "results.csv").exists() and (folder / "summary.json").exists()`.** The table and the JSON with telemetry.

**`assert len(sweep.rows) == 2 and all("victor_rate" in row for row in sweep.rows)`.** One row per value, each carrying the batch metrics from `batch_metrics`.

**`assert len(list((folder / "plots").glob("*.png"))) >= 5`.** One PNG per numeric column (`games`, `victor_rate`, `mean_days`, the three method shares, `eliminations_per_point`, `mean_interventions`, `eliminations_per_game`, `entropy`, `mean_survival_ticks`, `kill_rate`) plus `action_distribution_by_value.png`. The behaviour charts go into a `behaviour/` subfolder, which the non-recursive glob does not count.

### `test_plots_write_one_png_each(tmp_path)`

**Setup.** One recorded game with telemetry attached, then `plots.behaviour_plots(telemetry.summary(), tmp_path)`.

**`assert len(written) == 12 and all(path.exists() for path in written)`.** The twelve behaviour charts, in order: action distribution, actions by thirst, by hunger, by health, instinct curves, consumption timing, fight or flight, proximity versus remaining, actions by remaining, the position heatmap, the armed versus unarmed heatmaps, and deaths by cause. Each function returns the path it wrote. A failure would mean a plot function crashed on a real summary, which is what happens when a summary key is renamed in `telemetry.py` but not in `plots.py`.

## How to run and extend

```bash
MPLBACKEND=Agg python -m pytest tests/test_research.py
python -m pytest tests/test_research.py -k "mlp or adam"
python -m pytest tests/test_research.py::test_sweep_sets_nested_fields_and_writes_results
python -m pytest tests/test_research.py -v --durations=0
```

**1. Bias gradients too.** The finite-difference test only checks weights. The same loop on `net.biases[layer][j]` against `grads[layer][1][j]` covers the other half of `backward`.

```python
def test_bias_gradients_match():
    rng = np.random.default_rng(0)
    x, target = rng.normal(size=(6, 5)), rng.normal(size=(6, 3))
    net = MLP([5, 4, 3], "tanh", rng=np.random.default_rng(1))
    out, cache = net.forward_cached(x)
    grads = net.backward(cache, 2.0 * (out - target) / out.size)
    eps, b = 1e-6, net.biases[0]
    b[1] += eps; plus = np.mean((net.forward(x) - target) ** 2)
    b[1] -= 2 * eps; minus = np.mean((net.forward(x) - target) ** 2)
    b[1] += eps
    assert abs((plus - minus) / (2 * eps) - grads[0][1][1]) < 1e-5
```

**2. Tracked subsets.** `BehaviorTelemetry(50, 50, {0, 1})` should only count two tributes.

```python
def test_telemetry_tracks_a_subset():
    game = Game(small())
    telemetry = BehaviorTelemetry(50, 50, {0, 1}).attach(game)
    game.run()
    assert len(telemetry.summary()["survival_ticks"]) == 2
```

**3. `stop()` ends a sweep early.**

```python
def test_sweep_stops_after_first_value(tmp_path):
    sweep = Sweep(small(), SweepConfig(name="s", parameter="chaos", values=[0.0, 1.0], games_per_value=1, results_dir=tmp_path))
    sweep.run(on_value=lambda row: sweep.stop())
    assert len(sweep.rows) == 1
```

**4. Gradient clipping caps the norm.** Feed `_clip` a huge gradient and check the combined length equals `max_grad_norm`.

## Gotchas

**Always set `MPLBACKEND=Agg` or let `plots.py` do it.** `plots.py` calls `matplotlib.use("Agg")` at import, so these tests never open a window. Importing `hunger_games.renderer` first in the same process can pick a different backend; the run folder tests still work because `Agg` is forced before any figure is made.

**These are the slow tests.** The three trainer and sweep tests take about three seconds each because every chart is a separate matplotlib figure and the GIFs are animated frame by frame. Keep `epochs`, `generations` and `games_per_value` at the minimum that still draws a line.

**The gradient passed to `backward` is already a mean.** If you write your own training loop and pass `2.0 * (out - target)` without the `/ out.size`, the gradients are `N` times too large. Adam mostly hides that, plain `apply_gradients` does not.

**`MLP.backward` returns a list, not a dict.** `grads[layer]` is `(grad_w, grad_b)`; the test indexes `[0]` for the weight matrix.

**Telemetry hooks must be attached before the game runs.** `attach` appends to `game.decision_hooks` and `game.tick_hooks`. Attaching after `run()` records nothing.

**Merging is not deduplication.** `merge([summary, summary])` doubles everything on purpose. Pass each game's summary once.

**Entropy bounds depend on `MENU_SIZE`.** The `log(16)` in the REINFORCE test is the menu length from `brain/neural.py`. Adding a menu item changes the bound.

**Run folders are timestamped to the second.** Two `save_run` calls in the same second with the same name share a folder because `mkdir(exist_ok=True)` does not complain. The tests use different names.

**The sweep writes `batches/<value>/` too.** `Runner` is given an `output_dir` under the run folder, so a sweep leaves CSVs per value as well as the summary.
