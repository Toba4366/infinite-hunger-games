# `test_feed.py`

**Source:** [tests/test_feed.py](../../tests/test_feed.py)
**Tests:** [../training/genetic.md](../training/genetic.md) (`GeneticTrainer`, `TrainingConfig`, `GenerationStats.showcase`), [../training/reinforce.md](../training/reinforce.md) (`ReinforceTrainer`, `RLConfig`, `EpochStats.showcase`, `play_rl_episode`), [../ui/session.md](../ui/session.md) (`Session.feed_mode`, `update`, `_advance_feed`, `load_recording`, `start_champion_game`, `network_evolution`), and through them [../recorder.md](../recorder.md) (`Recorder`, `Recording`), [../game.md](../game.md), [../config.md](../config.md) (`SimulationConfig`)

## Purpose

The dashboard has a training feed. While a trainer runs in the background, the feed can show what training looks like instead of a table of numbers. It has three modes. `"off"` shows nothing. `"replay"` loads the real training game that each generation (or epoch) recorded and plays it back. `"live"` hands the newest champion to the trainer's learner slots and starts a fresh game so the Network tab shows real activations.

For the replay mode to work, both trainers must record one game per training step. That is the `record_showcase` switch on `TrainingConfig` and `RLConfig`, and the `showcase` field on `GenerationStats` and `EpochStats`. For the live mode to work, the session has to notice that a new step has arrived, wait until the arena is free, and then start the champion game.

This file checks all of that. The first test proves the genetic trainer records a full game every generation, keeps it out of the JSON rows, and records nothing when the switch is off. The second proves the REINFORCE trainer records its first training game each epoch, with the result filled in. The third drives a whole feed cycle through a `Session`: train in a thread, let the feed replay the newest generation, pretend to watch it to the end, switch to live, and check that a champion game starts and that the Network tab's evolution chart has data.

None of this needs a window. `Session` has no GUI code in it, so the test runs in CI in under ten seconds.

## Concepts you need

**Test discovery.** Three `test_*` functions. `small` is a helper.

**Polling a thread.** `Session.start_training` runs the trainer in a daemon thread. The third test waits with `while session.training_running: time.sleep(0.05)`, checking every 50 milliseconds. This is the same thing the dashboard does every frame, rather than the `join` that `test_ui_session.py` uses.

**Recordings.** `Recorder(game).record_all()` plays a game to the end and returns a `Recording` with one `Frame` per tick, starting with tick 0. `recording.length` is the frame count and `recording.result` is the `GameResult`, filled in when the game ends. A recording of a game that ran at least one tick has `length > 1`.

**Learner slots.** Both trainers put the learner into a few tribute slots and fill the rest with the config's voting brain. `learner_ids(num_players, learners)` spreads the slots evenly: `[0, 4, 8, 12, 16, 20]` for six learners in 24, `[0, 6, 12, 18]` for four. The genetic trainer plays one such game per genome per round and scores the genome by its copies' mean episode return.

**`all(...)` with a generator.** `all(stats.showcase is not None and stats.showcase.length > 1 for stats in trainer.history)` checks every generation without building a list, and stops at the first failure.

**`in` on strings and dicts.** `"showcase" not in trainer.history_rows()[0]` asks whether a dict has that key. `"replaying" in session.feed_label` asks whether a string contains that word.

**Private attributes.** `session._feed_steps_seen` starts with an underscore, which means the session's own business. The test resets it on purpose to make the feed think a new step has arrived. Tests are allowed to peek and poke.

**numpy shape.** `evolution["genes"].shape[0]` is the number of rows of a 2-D array. Here one row per training step.

**Running a subset.** `python -m pytest tests/test_feed.py -k session` runs only the third test.

## Walkthrough

### `small() -> SimulationConfig`

```python
def small() -> SimulationConfig:
    return SimulationConfig(seed=3, width=40, height=40, max_days=3)
```

A 40 by 40 arena capped at 3 days, which is 72 ticks at the default 24 ticks per day. The player count stays at the default 24. Games this size finish in a few hundredths of a second, so a few dozen of them per test stay fast. Seed 3 fixes the arena; the trainers draw their own game seeds from their own generators.

### `test_genetic_trainer_records_one_showcase_game_per_generation()`

**Setup.** `GeneticTrainer(small(), TrainingConfig(brain_name="voting", population_size=24, generations=2, rounds_per_generation=1, validation_games=0, seed=0))`. `opponents` keeps its default `"voting"`, so each generation `evaluate_against_voting` plays one game per genome: 24 games, each with the genome's six copies against 18 voting tributes. The record flag for a job is `index == 0 and r == 0 and self.training.record_showcase`, so only genome 0's first-round game is recorded, and `record_showcase` defaults to `True`. `validation_games=0` skips the validation games to keep the test quick. `trainer.run()` plays both generations, 48 games in about 5 seconds.

**`assert all(stats.showcase is not None and stats.showcase.length > 1 for stats in trainer.history)`.** Both `GenerationStats` carry a `Recording`, and each has more than one frame: frame 0 plus at least one tick. A failure would mean `evaluate_against_voting` forgot to pass the record flag, dropped the `"recording"` entry of the first job's result, or `step_generation` did not put `_last_showcase` into the stats.

**`assert "showcase" not in trainer.history_rows()[0]`.** `to_row()` walks `__dict__` and skips `champion`, `telemetry` and `showcase`. If the recording leaked into a row, `history.json` would fail to serialise, because a `Recording` is not JSON. This line catches that before `save_run` does.

**The quiet trainer.** A second `GeneticTrainer` with the same settings, `generations=1`, and `record_showcase=False`. After `quiet.run()`, `assert quiet.history[0].showcase is None`. With the flag off, every job's record flag is `False`, so `play_rl_episode` skips the `Recorder` and returns `"recording": None` for every game, and `_last_showcase` stays `None`. This is the off switch that a long headless run needs, because every recording otherwise stays in memory until the trainer is dropped.

### `test_reinforce_trainer_records_one_showcase_game_per_epoch()`

**Setup.** `ReinforceTrainer(small(), RLConfig(epochs=1, episodes_per_epoch=2, learners_per_game=4, validation_games=0, seed=0))`. One epoch of two training games, four learners per game (slots `[0, 6, 12, 18]` of 24), no validation. `record_showcase` defaults to `True`. `trainer.run()` plays the epoch: `step_epoch` calls `_collect(..., record_first=True)`, which sets `record=True` on the first job only.

**`assert trainer.history[0].showcase is not None`.** `step_epoch` stored `episodes[0].get("recording")` on the `EpochStats`. The first episode's dict carries the `Recording` that `play_rl_episode` made with `Recorder(game).record_all()`. Only the first episode is recorded; the second one has `"recording": None`, which the test does not check.

**`assert trainer.history[0].showcase.result is not None`.** `record_all` fills in `recording.result` when the game ends. This proves the recording is a finished game and not a recorder that captured frame 0 and stopped. The `game.run()` call that follows the recorder in `play_rl_episode` finds the game already over and does not play it twice.

### `test_session_feed_replays_and_then_plays_the_champion_live()`

**Setup.** `Session(small())` builds a Perlin arena into the painter and a 24-tribute roster. `session.feed_mode = "replay"` turns the feed on before training starts. `session.start_training(TrainingConfig(brain_name="voting", population_size=24, generations=2, rounds_per_generation=1, validation_games=0, seed=0))` builds a `GeneticTrainer` on the painted map (a config copy with the roster's 24 players, and `Scenario(terrain=...)`), resets `_feed_steps_seen` and `feed_label`, and starts a daemon thread. The default `method` is `"genetic"`. The thread runs the session's own loop: `trainer.step(progress)` until `learning_history` has `generations` entries or `stop()` was called. Same settings as the first test, so every generation records genome 0's game.

**`while session.training_running: time.sleep(0.05)`.** Poll until the thread ends. Two generations of 24 games each take about 3 seconds.

**`session.update(0.01)`.** One UI frame of 10 milliseconds. `update` first calls `_advance_feed`. The feed is on, the trainer exists, and `len(trainer.history)` is 2 while `_feed_steps_seen` is 0, so there is something new. `_feed_ready_for_next()` returns `True` because nothing is loaded. The feed catches up (`_feed_steps_seen = 2`), takes `history[-1].showcase` (generation 1's game) and calls `load_recording` on it, which adopts the recording, drops any live game, loads the recording's terrain into the painter, adopts its config, rebuilds the roster from the recording, and rewinds. Then `feed_label` becomes `"training feed: replaying a real generation 1 game"` and `playing` is set to `True`. Back in `update`, `playing` is now `True`, but 0.01 seconds at the default 8 ticks per second is 0.08 of a tick, so the playhead does not move.

**`assert session.game is None and session.recording is not None`.** A replay: a recording to scrub through but no live `Game`. This is how the session tells a replay from a live game everywhere else.

**`assert "replaying" in session.feed_label`.** The label names the replay mode. The full text also says which generation.

**`assert session.playing`.** The feed started playback at the current speed, so the dashboard would animate the replay without another click.

**`session.playhead = session.recording.length - 1`.** Pretend the whole replay has been watched by jumping to the last frame. `_feed_ready_for_next` treats a replay whose playhead is on its last frame as finished.

**`session.feed_mode = "live"` and `session._feed_steps_seen = 0`.** Switch modes and forget that generation 1 was already shown. Without the reset, `len(history) <= _feed_steps_seen` would be true and the feed would wait for a generation that is never coming, because training has finished.

**`session.update(0.01)`.** `_advance_feed` again. Something is "new" (2 steps, 0 seen) and the arena is free (replay at its last frame). The live branch calls `start_champion_game(all_slots=False)`. That gives the champion genome to the trainer's `_learner_ids()` slots, `[0, 4, 8, 12, 16, 20]` for six learners in 24 players, through `give_champion`, then calls `new_game()` to build a fresh recorded `Game` on the painted map, and sets `playing`. The label becomes `"training feed: generation 1 champion playing live"`.

**`assert session.game is not None and "live" in session.feed_label`.** There is a live game now, and the label says so.

**`evolution = session.network_evolution()`.** Stacks the learner after every training step into a matrix and reports how it changed. `genome_history()` reads `stats.learner` from each entry of the trainer's `learning_history`; for a genetic trainer that is the generation's champion.

**`assert evolution is not None and len(evolution["steps"]) == 2 and evolution["genes"].shape[0] == 2`.** Two generations give two steps and a two-row matrix. The voting brain has 8 genes, so `genes` is 2 by 8, well under the `max_genes` cap of 200.

**`assert evolution["change"][0] == 0.0`.** The change entry for the first step is defined as zero, because there is no previous champion to compare with. The second entry is the Euclidean distance between the two champions and is not checked, because with elitism the same genome can win twice and the distance would then be zero too.

## How to run and extend

```bash
python -m pytest tests/test_feed.py
python -m pytest tests/test_feed.py -v
python -m pytest tests/test_feed.py -k genetic
python -m pytest tests/test_feed.py::test_session_feed_replays_and_then_plays_the_champion_live
```

**1. The RL off switch.** Mirror the genetic test for the other trainer.

```python
def test_reinforce_trainer_can_skip_showcases():
    trainer = ReinforceTrainer(
        small(), RLConfig(epochs=1, episodes_per_epoch=1, learners_per_game=4, validation_games=0, seed=0, record_showcase=False)
    )
    trainer.run()
    assert trainer.history[0].showcase is None
    assert "showcase" not in trainer.history_rows()[0]
```

**2. The feed does nothing when off.** The default mode is `"off"`, so `update` never touches the arena.

```python
def test_feed_off_leaves_the_arena_alone():
    session = Session(small())
    session.start_training(
        TrainingConfig(brain_name="voting", population_size=24, generations=1, rounds_per_generation=1, validation_games=0, seed=0)
    )
    while session.training_running:
        time.sleep(0.05)
    session.update(0.01)
    assert session.recording is None and session.feed_label == ""
```

**3. A missing recording gets a label, not a crash.** With `record_showcase=False`, replay mode explains itself.

```python
def test_feed_replay_without_a_recording():
    session = Session(small())
    session.feed_mode = "replay"
    session.start_training(
        TrainingConfig(
            brain_name="voting", population_size=24, generations=1, rounds_per_generation=1, validation_games=0, seed=0, record_showcase=False
        )
    )
    while session.training_running:
        time.sleep(0.05)
    session.update(0.01)
    assert session.recording is None and "no recording" in session.feed_label
```

**4. The feed waits for the arena.** A replay that has not been watched to the end blocks the next step.

```python
def test_feed_waits_until_the_replay_is_watched():
    session = Session(small())
    session.feed_mode = "replay"
    session.start_training(
        TrainingConfig(brain_name="voting", population_size=24, generations=2, rounds_per_generation=1, validation_games=0, seed=0)
    )
    while session.training_running:
        time.sleep(0.05)
    session.update(0.01)
    first = session.recording
    session._feed_steps_seen = 0
    session.update(0.01)
    assert session.recording is first
```

**5. The RL feed.** The same session path with `method="reinforce"`; the label then says "epoch".

```python
def test_session_feed_with_reinforce():
    session = Session(small())
    session.feed_mode = "replay"
    session.start_training(RLConfig(epochs=1, episodes_per_epoch=1, learners_per_game=4, validation_games=0, seed=0), method="reinforce")
    while session.training_running:
        time.sleep(0.05)
    session.update(0.01)
    assert session.game is None and "epoch 0" in session.feed_label
```

## Gotchas

**The feed only checks on `update`.** Nothing happens when training finishes. The session notices a new step the next time the dashboard calls `update`, which is every frame in the real app and by hand in a test.

**`_feed_steps_seen` catches up in one jump.** When two generations arrive between frames, the feed sets `_feed_steps_seen` to 2 and shows only the newest. Generation 0's game is never shown in the third test. That is by design: the feed shows the latest state of training, not a queue.

**The reset of `_feed_steps_seen` is a test trick.** In the dashboard, switching to live mode while training is still running picks up the next generation naturally. The test resets the counter because training has already finished and there will be no next generation.

**`load_recording` rewrites the session.** It replaces `session.config` with the recording's config, reloads the painter from the recording's terrain, and rebuilds the roster from the recording's roster. After the replay step, the session's config is the trainer's copy of it, with `num_players` set from the roster and `seed` set to whatever random seed that training game drew. The live champion game is then built from that config and the rebuilt roster.

**`start_champion_game(all_slots=False)` reaches into the trainer.** It calls `trainer._learner_ids()`, a private method, to find the learner slots. Every trainer has one, and they all use the shared `learner_ids` rule with their own `learners_per_game`: 6 by default for both the genetic and the REINFORCE trainer, 4 in this file's RL test.

**Showcases are big.** Every step's recording stays in `trainer.history` for the life of the trainer. The tests run one or two steps on a 40 by 40 map, which is nothing. A long run on a big map should set `record_showcase=False` unless the feed is being watched.

**Only the first game of a step is recorded.** The genetic trainer plays one game per genome per round and records only genome 0's first game, so with 24 genomes the other 23 games are never recorded and the champion of the generation is usually not in the recorded one. The REINFORCE trainer records the first of its `episodes_per_epoch` games.

**`showcase.length > 1` needs at least one tick.** A game that was over at tick 0 would give a recording of length 1 with `result` filled in. That cannot happen with 24 living tributes, but if you shrink the test to two players and a lethal start, the first assertion is the one that would fail.

**Daemon threads and polling.** If a test forgets the `while session.training_running` loop, `update` sees an empty history, the feed does nothing, and the assertions about `recording` fail in a confusing way. Always wait first. The poll loop has no timeout; a hung trainer would hang the test.
