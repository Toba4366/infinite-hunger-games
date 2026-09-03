# `test_imitation.py`

**Source:** [tests/test_imitation.py](../../tests/test_imitation.py)
**Tests:** [../training/imitation.md](../training/imitation.md) (`ImitationTrainer`, `ImitationConfig`), [../brain/neural.md](../brain/neural.md) (`NeuralBrain.action_to_menu_index`, `ATTACK_INDEX`, `FLEE_INDEX`, `FIRST_MOVE_INDEX`), [../training/genetic.md](../training/genetic.md) (`GeneticTrainer`, `TrainingConfig`, the `initial_genome` warm start), [../training/reinforce.md](../training/reinforce.md) (`ReinforceTrainer`, `RLConfig`, the `initial_genome` warm start), [../training/runs.md](../training/runs.md) (`save_run` with method `"imitation"`), with [../actions.md](../actions.md) (`Action`, `ActionType`) and [../config.md](../config.md) (`SimulationConfig`)

## Purpose

Imitation pretraining is the first step of the recommended training flow: copy the voting brain, then evolve or reinforce from the copy. This file checks the three pieces that flow depends on.

The first test pins down the label function. `action_to_menu_index` turns a teacher's `Action` into a menu index, and if it disagreed with `menu_to_action` the student would learn the wrong item for every move. The second test runs a tiny imitation trainer end to end and checks that it learns (loss falls or accuracy rises), that it beats chance on held-out data, that the showcase recording and the champion exist, and that `save_run` writes the imitation charts and the champion file. The third test proves the warm starts: a genetic population seeded from a genome starts with that genome in slot 0 and a relative in slot 1, and a REINFORCE policy seeded from a genome starts as exactly that genome.

Without these tests the imitation method would only be exercised by clicking Train in the dashboard, and a broken label mapping would show up as a student that never learns rather than as a failing test.

## Concepts you need

**Test discovery.** Three `test_*` functions. `small` is a helper.

**The `tmp_path` fixture.** The second test takes `tmp_path` and pytest gives it a fresh temporary folder for the run folder.

**Menu indices.** The neural brain chooses from a 16-item menu: six simple actions (`rest`, `drink`, `eat`, `hunt`, `pick_up`, `heal`), then `attack` at index 6 (`ATTACK_INDEX`), `flee` at 7 (`FLEE_INDEX`), and eight moves from index 8 (`FIRST_MOVE_INDEX`) in `DIRECTIONS` order. See [../brain/neural.md](../brain/neural.md).

**`np.array_equal`.** True when two arrays have the same shape and every element is equal. The warm-start test uses it both ways: equal for the copied genome, not equal for the mutated relative.

**Chance accuracy.** With 16 menu items a random guess is right 1/16 of the time. Beating that on held-out data is the weakest possible proof of learning, chosen so the test passes on two tiny games.

**Running a subset.** `python -m pytest tests/test_imitation.py -k warm` runs the warm-start test alone.

## Walkthrough

### `small() -> SimulationConfig`

```python
def small() -> SimulationConfig:
    return SimulationConfig(seed=5, width=50, height=50, max_days=4)
```

A 50 by 50 arena capped at 4 days, which is 96 ticks. Every other field is the default, so games have 24 tributes and the neural config is `(64, 32)` hidden layers, 5872 parameters. The trainer seeds override the config seed for every game the trainer plays.

### `test_action_to_menu_index_is_the_inverse_of_menu_to_action()`

Five direct checks of the static method, with no game involved.

**`assert NeuralBrain.action_to_menu_index(Action(ActionType.DRINK)) == 1`.** Simple actions map to their position in `SIMPLE_ACTIONS`; `drink` is second.

**`assert NeuralBrain.action_to_menu_index(Action.attack(3)) == ATTACK_INDEX`.** Any attack, whatever the target, is menu item 6. The target is filled back in from the perception by `menu_to_action` at decision time.

**`assert NeuralBrain.action_to_menu_index(Action.flee(1, 0)) == FLEE_INDEX`.** Any flee, whatever the direction, is item 7.

**`assert NeuralBrain.action_to_menu_index(Action.move(1, 0)) == FIRST_MOVE_INDEX + 4`.** `(1, 0)` is the fifth entry of `DIRECTIONS`, so the move "right" is item 12.

**`assert NeuralBrain.action_to_menu_index(Action.move(0, 0)) == 0`.** A zero move is not in `DIRECTIONS`, so it counts as `rest`.

### `test_imitation_learns_the_teacher_and_saves(tmp_path)`

**Setup.** `ImitationTrainer(small(), ImitationConfig(demonstration_games=2, epochs=6, validation_games=1, learners_per_game=4, seed=0))`. Two teacher games give a few thousand demonstrations, 20 percent held out. Each epoch plays one greedy validation game with the student driving four tributes on slots `[0, 6, 12, 18]`. `history = trainer.run()`. This is the slowest test in the suite, about 16 seconds: six epochs over the demonstrations with the default `(64, 32)` network, six validation games, and a showcase recording each epoch.

**`assert len(history) == 6`.** One `ImitationStats` per epoch.

**`assert history[-1].train_accuracy > history[0].train_accuracy or history[-1].train_loss < history[0].train_loss`.** Learning happened. Either form of progress is accepted because accuracy can stall for a few epochs while the loss keeps falling.

**`assert history[-1].val_accuracy > 1.0 / 16.0`.** Better than chance on the held-out demonstrations.

**`assert history[-1].showcase is not None`.** `record_showcase` defaults to `True` and `validation_games` is 1, so the first validation game was recorded.

**`assert trainer.champion.size == trainer.policy.parameter_count`.** The champion is the best-validation-loss genome and must be the full policy, 5872 numbers for the default architecture.

**`folder = save_run(trainer, "imitation", "test_im", tmp_path)`.** The run folder: `config.json`, `history.json`, `learning.json` (the unified rows every method shares), `events.txt`, `champion.json` and `plots/`.

**`assert (folder / "plots" / "accuracy.png").exists() and (folder / "champion.json").exists()`.** `accuracy.png` is written only on the imitation branch of `training_run_plots`, so this checks that `save_run` passed the method through. `champion.json` is written because `trainer.champion` is not `None` and `save_champion` exists on the imitation trainer.

### `test_warm_starts_begin_from_the_given_genome()`

**Setup.** `student = ImitationTrainer(config, ImitationConfig(demonstration_games=1, epochs=1, validation_games=0, seed=0))`, then `student.run()` and `genome = student.champion`. With no validation games the epoch is one pass over one game's demonstrations; the validation loss on the held-out 20 percent still picks `best_genome`, so `champion` is the genome after that epoch.

**The genetic trainer.** `GeneticTrainer(config, TrainingConfig(brain_name="neural", population_size=8, generations=1, rounds_per_generation=1, validation_games=0, seed=0), initial_genome=genome)`. Nothing is run; only the constructor is under test.

**`assert np.array_equal(ga.population[0], genome)`.** The first member of the population is an exact copy of the seed genome.

**`assert not np.array_equal(ga.population[1], genome)`.** The second is a relative: the seed plus Gaussian noise with standard deviation `0.25 * mutation_scale`, which is 0.025 with the default scale.

**The REINFORCE trainer.** `ReinforceTrainer(config, RLConfig(epochs=1, episodes_per_epoch=1, validation_games=0, seed=0), initial_genome=genome)`.

**`assert np.array_equal(rl.policy.genome(), genome)`.** The policy network starts as exactly the seed genome. The value network is still fresh.

## How to run and extend

```bash
python -m pytest tests/test_imitation.py
python -m pytest tests/test_imitation.py -v
python -m pytest tests/test_imitation.py -k menu_index
python -m pytest tests/test_imitation.py::test_imitation_learns_the_teacher_and_saves
```

**1. Every menu index round-trips.** Build a perception from a real game and check both directions for the eight moves and the six simple actions. Attack and flee are excluded because `menu_to_action` may turn them into a move or a rest when nobody is in reach.

```python
from hunger_games.brain.neural import FIRST_MOVE_INDEX, NeuralBrain
from hunger_games.game import Game

def test_menu_round_trip_for_moves_and_simple_actions():
    game = Game(small())
    player = game.players[0]
    perception = player.perceive(game.arena, game.players, False, 0.0, 1.0, game.config.vision_radius)
    for index in list(range(6)) + list(range(FIRST_MOVE_INDEX, 16)):
        assert NeuralBrain.action_to_menu_index(NeuralBrain.menu_to_action(index, perception)) == index
```

**2. The champion file says where it came from.**

```python
import json

def test_champion_file_records_the_method(tmp_path):
    trainer = ImitationTrainer(small(), ImitationConfig(demonstration_games=1, epochs=1, validation_games=0, seed=0))
    trainer.run()
    trainer.save_champion(tmp_path / "c.json")
    data = json.loads((tmp_path / "c.json").read_text())
    assert data["method"] == "imitation" and data["teacher"] == "voting"
```

**3. `stop()` ends a run early.**

```python
def test_stop_after_first_epoch():
    trainer = ImitationTrainer(small(), ImitationConfig(demonstration_games=1, epochs=5, validation_games=0, seed=0))
    trainer.run(on_epoch=lambda stats: trainer.stop())
    assert len(trainer.history) == 1
```

**4. A warm-started student starts from the genome.**

```python
def test_student_warm_start():
    config = small()
    first = ImitationTrainer(config, ImitationConfig(demonstration_games=1, epochs=1, validation_games=0, seed=0))
    first.run()
    second = ImitationTrainer(config, ImitationConfig(seed=1), initial_genome=first.champion)
    assert np.array_equal(second.policy.genome(), first.champion)
```

**5. Learning from winners only.** `winners_top` keeps only the decisions of tributes that placed that well, so fewer demonstrations come out of the same games.

```python
def test_winners_top_keeps_fewer_demonstrations():
    everyone = ImitationTrainer(small(), ImitationConfig(demonstration_games=1, seed=0))
    winners = ImitationTrainer(small(), ImitationConfig(demonstration_games=1, seed=0, winners_top=3))
    assert winners.collect() < everyone.collect()
```

## Gotchas

**The tests play real games.** Two demonstration games plus six validation games on a 50 by 50 map, with a showcase recording per epoch, take about 16 seconds. Keep `demonstration_games` and `epochs` small in new tests, and set `record_showcase=False` when the feed is not under test.

**`validation_games=0` does not switch off validation loss.** The held-out demonstrations are still scored every epoch, so `champion` is still chosen by validation loss. Only the greedy games, telemetry and showcase are skipped.

**The GA relative is random.** `population[1]` differs from the seed because noise was added; with `seed=0` the noise is fixed. A `mutation_scale` of 0 would make the relative equal to the seed and fail the inequality.

**Architecture must match.** All three trainers in the warm-start test share one `config`, so their networks are the same shape. Passing a genome from a different `NeuralConfig` raises `ValueError` in `set_genome`.

**`workers` stays at 1.** Multi-process collection works but needs the `__main__` guard on macOS, and pytest already runs in one process.
