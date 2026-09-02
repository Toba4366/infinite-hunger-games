# `test_recorder_training.py`

**Source:** [tests/test_recorder_training.py](../../tests/test_recorder_training.py)
**Tests:** [../recorder.md](../recorder.md) (`Recorder`, `Recording`, `Frame`), [../training/genetic.md](../training/genetic.md) (`GeneticTrainer`, `TrainingConfig`, `GenerationStats`), [../game.md](../game.md) (`Game`), [../config.md](../config.md) (`SimulationConfig`), [../brain/voting.md](../brain/voting.md) and [../brain/neural.md](../brain/neural.md) (the genomes being evolved)

## Purpose

Two features share this file because both wrap a `Game` and drive it from outside. The `Recorder` steps a game and copies its state after every tick into a `Frame`, so the dashboard can scrub back and forth and export GIFs. The `GeneticTrainer` plays populations of genomes against each other and breeds the winners, which is how the dashboard's Train tab makes brains better.

The first test records a whole game and proves the bookkeeping: one frame per tick plus the starting frame, every elimination landing on exactly one frame, the result filled in, and a pickle round trip that preserves positions and the roster. The second runs two generations of voting-brain evolution and checks the statistics are sane, the champion has the right shape, and the champion file round-trips through JSON. The third evolves neural brains with a population that does not divide evenly into games, proving the padding logic scores every genome.

Both features are only exercised through the dashboard in normal use, so without these tests a broken recorder or trainer would only be noticed by someone clicking around a window. The trainer's newer features, validation games, telemetry and run folders, are covered in [test_research.md](test_research.md).

## Concepts you need

**Test discovery.** Three `test_*` functions. `small` is a helper.

**The `tmp_path` fixture.** Two tests take `tmp_path` and pytest gives each a fresh temporary folder. The replay file and the champion JSON go there and are cleaned up afterwards.

**Pickle.** `Recording.save` uses `pickle.dumps` to write the whole object, numpy arrays included, and `Recording.load` reads it back. Pickle can run code when loading, so only open replay files you made yourself.

**Generator expressions inside `sum` and `all`.** `sum(len(f.eliminations) for f in recording.frames)` adds up one number per frame without building a list. `all(... for f in frames)` stops at the first `False`.

**`np.isfinite`.** True for ordinary numbers, False for `inf` and `nan`. A fitness of `nan` would silently break sorting, so the test rules it out.

**Chained comparisons.** `a >= b >= c` is Python for `a >= b and b >= c`. The trainer test uses it to check best, mean and worst in one line.

**Running a subset.** `python -m pytest tests/test_recorder_training.py -k trainer` runs the two trainer tests.

## Walkthrough

### `small() -> SimulationConfig`

```python
def small() -> SimulationConfig:
    return SimulationConfig(seed=7, width=50, height=50, max_days=4)
```

A 50 by 50 arena capped at 4 days, which is 96 ticks. Games this size finish in well under a tenth of a second, so the trainer tests, which play several games each, stay fast. Seed 7 makes every game reproducible; the trainer draws its own game seeds from its own generator, but the config seed still fixes the arena used for the recorder test.

### `test_recorder_captures_every_tick_and_round_trips(tmp_path)`

**Setup.** `game = Game(small())`, `recorder = Recorder(game)`. The constructor captures frame 0 straight away. `recording = recorder.record_all()` then steps to the end, capturing after each tick.

**`assert recording.length == game.tick + 1`.** With seed 7 the game reaches the 96-tick cap, so there are 97 frames: frame 0 plus one per tick. A failure would mean a tick was skipped or captured twice.

**`assert sum(len(f.eliminations) for f in recording.frames) == len(game.eliminations)`.** `Recorder.capture` records only the eliminations that appeared since the previous capture, using `_eliminations_seen` as a cursor. Summing across frames must give the game's full list of 16 deaths: none dropped, none duplicated.

**`assert recording.result is not None and recording.result.ticks == game.tick`.** `Recorder.step` fills in `recording.result` when the game ends, and `record_all` fills it in as a fallback. The result's tick count must match the game's.

**`assert all(len(f.players) == len(game.players) for f in recording.frames)`.** Every frame snapshots every tribute, dead or alive. The dashboard relies on this to draw greyed-out corpses.

**`recording.save(path); loaded = Recording.load(path)`.** The pickle round trip.

**`assert loaded.length == recording.length`.** Same frame count.

**`assert loaded.frames[-1].players[0].x == recording.frames[-1].players[0].x`.** A concrete value from the last frame survived.

**`assert loaded.roster[0].name == game.players[0].name`.** The roster, stored once per recording, survived too.

### `test_trainer_improves_or_holds_and_saves_a_champion(tmp_path)`

**Setup.** `GeneticTrainer(small(), TrainingConfig(brain_name="voting", population_size=24, generations=2, rounds_per_generation=1, seed=0))`. The population equals the default player count, so each round is exactly one evaluation game with no padding. The `TrainingConfig` defaults also switch on two validation games per generation (the generation's champion drives six tributes against the config's voting brain on seeds 90000 and 90001) and behaviour telemetry. `history = trainer.run()`.

**`assert len(history) == 2`.** One `GenerationStats` per generation.

**`assert all(np.isfinite(s.best_fitness) and s.best_fitness >= s.mean_fitness >= s.worst_fitness for s in history)`.** Fitness is a real number, and the best is at least the mean, which is at least the worst. This catches a sign flip in `np.argsort(fitness)[::-1]`, which would swap best and worst. Fitness itself is `fitness_of`: a placing term from 0 to 1 plus `kills_weight` (0.05) per kill and `days_weight` (0.01) per day survived.

**`assert trainer.champion.shape == (8,)`.** The voting brain has eight genes, so its genome is a length-8 vector. `trainer.champion` is the champion of whichever generation had the best fitness.

**`trainer.save_champion(path); data = GeneticTrainer.load_champion(path)`.** The JSON round trip. The file holds the brain name, the neural config, the genome as a list, the best fitness and the generation count. `load_champion` turns the genome back into a numpy array and the neural config back into a `NeuralConfig`.

**`assert data["brain_name"] == "voting"`.** The kind was saved.

**`assert np.allclose(data["genome"], trainer.champion)`.** The genome came back as a numpy array equal to the champion. `allclose` because JSON floats can lose a few bits.

**`assert trainer.champion_brain().name == "voting"`.** `champion_brain` builds a brain of the trained kind with the champion loaded, using the config's chaos unless you pass one.

The test name says "improves or holds" but no assertion compares generation 1 with generation 0. With one round per generation, fitness is too noisy for that to be a safe check.

### `test_trainer_handles_neural_genomes_and_padding()`

**Setup.** `TrainingConfig(brain_name="neural", population_size=30, generations=1, rounds_per_generation=1, seed=1)` on the same small config. 30 genomes do not divide into games of 24, so `_make_jobs` pads the shuffled order with 18 random repeats to make two full games.

**`assert trainer.genome_size == trainer.champion.size`.** A default neural brain has 1088 parameters (50 inputs, one hidden layer of 16, 16 outputs: 800 + 16 + 256 + 16). The champion must be that long.

**`assert (trainer.fitness > 0).all() or (trainer.fitness >= 0).all()`.** Every genome has a fitness of at least zero. Because `fitness_of` returns a placing term from 0 to 1 plus non-negative bonuses, the second half of the `or` is always true, so this line really only checks that `fitness` is a numeric array of the right shape with no `nan` (a `nan` compares False to everything). The `> 0` half is a stricter version that passes when every genome earned at least something.

**`assert len(trainer.population) == 30`.** After breeding, the population is the same size: 3 elites (10 percent of 30) plus 27 children.

## How to run and extend

```bash
python -m pytest tests/test_recorder_training.py
python -m pytest tests/test_recorder_training.py -v
python -m pytest tests/test_recorder_training.py -k recorder
python -m pytest tests/test_recorder_training.py::test_trainer_handles_neural_genomes_and_padding
```

**1. Frame 0 is the starting line.** Nobody has moved and nobody is dead.

```python
def test_frame_zero_is_untouched():
    game = Game(small())
    recording = Recorder(game).recording
    first = recording.frames[0]
    assert first.tick == 0 and first.day == 1
    assert all(p.alive for p in first.players)
    assert first.eliminations == [] and first.gifts == []
```

**2. Gifts land on frames too.** Same cursor logic as eliminations.

```python
def test_recorder_captures_gifts():
    game = Game(SimulationConfig(seed=7, width=50, height=50, max_days=4, sponsor_gift_chance=1.0))
    recording = Recorder(game).record_all()
    assert sum(len(f.gifts) for f in recording.frames) == len(game.gifts)
```

**3. A brain with no genome cannot be trained.** `RandomBrain.genome()` is empty.

```python
import pytest

def test_random_brain_is_not_trainable():
    with pytest.raises(ValueError):
        GeneticTrainer(small(), TrainingConfig(brain_name="random", population_size=4))
```

**4. `stop()` ends a run early.**

```python
def test_stop_after_first_generation():
    trainer = GeneticTrainer(small(), TrainingConfig(brain_name="voting", population_size=24, generations=5, rounds_per_generation=1, seed=0))
    trainer.run(on_generation=lambda stats: trainer.stop())
    assert len(trainer.history) == 1
```

**5. Validation and telemetry can be switched off** for a faster, quieter trainer.

```python
def test_trainer_without_validation():
    trainer = GeneticTrainer(small(), TrainingConfig(brain_name="voting", population_size=24, generations=1, rounds_per_generation=1, validation_games=0, collect_telemetry=False, seed=0))
    stats = trainer.run()[0]
    assert stats.val_fitness == 0.0 and stats.telemetry == {}
```

## Gotchas

**`Recorder` captures on construction.** `Recorder(game)` already appends frame 0. If you also call `capture()` yourself you get two copies of the start.

**`record_all` plays the game.** After it returns, `game.is_over` is True and `game.tick` is final. Do not expect to step the same game afterwards.

**Pickle files are not portable across code changes.** A `.replay` written before a dataclass gained a field may fail to load. Regenerate rather than migrate.

**Trainer seeds and config seeds are different generators.** `TrainingConfig.seed` drives population init, shuffling, breeding and per-game seeds. `SimulationConfig.seed` is overwritten for every evaluation game by `play_evaluation_game`, so it does not make training games repeatable on its own; the trainer seed does. Validation games use `validation_seed + i` instead, so they are the same every generation.

**Padding double-counts.** With 30 genomes and 24 per game, 18 genomes play twice in one round. Their fitness is a mean over their games, so they are not favoured, but `counts` differs per genome.

**Each generation plays more than the evaluation games.** With the defaults, two validation games and the telemetry bookkeeping run as well. That is why the two trainer tests take about 0.5 to 0.7 seconds each rather than the 0.2 seconds the evaluation games alone would need. Set `validation_games=0` and `collect_telemetry=False` when speed matters.

**`workers` stays at 1 in these tests.** Multi-process evaluation works but needs the `__main__` guard on macOS, and pytest already runs in one process. See [../runner.md](../runner.md) for the spawn rules.
