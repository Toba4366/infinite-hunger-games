# `test_ui_session.py`

**Source:** [tests/test_ui_session.py](../../tests/test_ui_session.py)
**Tests:** [../ui/painter.md](../ui/painter.md) (`MapPainter`), [../ui/session.md](../ui/session.md) (`Session`), and through them [../game.md](../game.md), [../recorder.md](../recorder.md), [../scenario.md](../scenario.md), [../training/genetic.md](../training/genetic.md), `hunger_games/research/telemetry.py`, [../config.md](../config.md) (`SimulationConfig`), [../resources.md](../resources.md) (`ResourceKind`), [../terrain.md](../terrain.md) (`TerrainType`)

## Purpose

The dashboard (`python -m hunger_games ui`) is split so that the interesting logic has no GUI in it. `MapPainter` is a terrain grid with brushes, stamps and presets. `Session` owns everything the dashboard is doing: the config, the painted map, the roster, the current game and its recording, the playback clock, the behaviour telemetry, the trainer and the sweep. The Dear PyGui code in `app.py` only reads and writes these two objects. That split is what makes this file possible: every test here runs without a window, without Dear PyGui installed, and in CI.

The first test drives the painter: brush, rectangle stamp, circular carve, every preset, and the coverage summary. The second walks through a whole dashboard session: edit loot and a podium, rename a tribute, start a game, play it at high speed, finish it, read the event log, scrub, and then save and reload the scenario, the replay and the config into a second session. The third starts genetic training in a background thread, waits for it, hands the champion genome to two tributes, and proves the next game really uses it.

If any of this broke, the dashboard would still open, and the failure would only show up as a button that does nothing or a wrong picture. These tests catch that in under a second.

## Concepts you need

**Test discovery.** Three `test_*` functions, no helpers.

**The `tmp_path` fixture.** The session test takes `tmp_path` and writes three files there: `s.json`, `r.replay` and `c.json`. pytest deletes the folder later.

**Grid indexing.** `painter.terrain` is a numpy array indexed `[y, x]`, but every painter and session method takes `(x, y)`. `painter.terrain[10, 13]` is column 13 of row 10.

**Threads.** `Session.start_training` runs the trainer in a `threading.Thread` so the window stays responsive. The test calls `session._training_thread.join(timeout=60)` to wait for it. The leading underscore means the attribute is private; tests are allowed to peek.

**`or` with `None`.** `assert x is None or x != 0` passes if there is no tribute near the cell, or if the nearest one is not tribute 0.

**Running a subset.** `python -m pytest tests/test_ui_session.py -k painter` runs one test.

## Walkthrough

### `test_painter_brush_stamps_and_presets()`

**Setup.** `MapPainter(60, 60)`: a 60 by 60 grid of grass with flat heights and `version = 0`.

**`painter.paint(10, 10, TerrainType.WATER, 2)`.** A round brush of radius 2 centred on `(10, 10)`.

**`assert painter.terrain[10, 10] == int(TerrainType.WATER)`.** The centre cell was painted.

**`assert painter.terrain[10, 13] == int(TerrainType.GRASS)`.** Column 13 is three cells away, outside a radius-2 brush, so it is untouched. A failure would mean the brush radius was too big or the circle test used `<` on the wrong value.

**`assert painter.version > before`.** Every edit bumps `version`. The canvas uses that number to know when to redraw, so a stamp that forgot `_changed()` would leave the screen stale.

**`painter.stamp_rectangle(50, 50, 40, 40, TerrainType.ROCK)` then `assert painter.terrain[45, 45] == int(TerrainType.ROCK)`.** The corners are given in the wrong order on purpose: the method sorts them, so the rectangle spans 40 to 50 on both axes and `(45, 45)` is inside.

**`painter.carve_round()` then `assert painter.terrain[0, 0] == int(TerrainType.VOID)`.** Voids everything outside the inscribed circle of radius 29. The corner is 42 cells from the centre.

**`for name in MapPainter.PRESETS: painter.apply_preset(name, config)`.** All five presets: `perlin`, `flat_field`, `flat_round`, `quarter_quell`, `lake_island`. For each, `assert painter.terrain.shape == (60, 60)` proves the size is kept, and `assert painter.heights.min() >= 0.0 and painter.heights.max() <= 1.0` proves the derived heights are in range. The `perlin` preset builds a real `Arena` from the config, so this also covers `MapPainter.load`.

**`painter.apply_preset("quarter_quell", config); coverage = painter.coverage()`.** The 75th-games island: a sea from the centre out to 45 percent of the radius, a sand island, a sand beach, rocky spokes. `coverage` returns each terrain's share of the non-void cells.

**`assert coverage["water"] > 0.1 and coverage["sand"] > 0.02`.** With a 60 by 60 grid the sea is about 17 percent and the sand about 8 percent. The thresholds are loose so the preset can be tweaked without breaking the test, but a preset that forgot the sea or the beach would fail.

### `test_session_edit_play_and_files(tmp_path)`

**Setup.** `Session(SimulationConfig(seed=3, width=50, height=50, max_days=6, start_thirst_min=0.3))`. The constructor generates a Perlin arena into the painter and a 24-tribute roster by building a throwaway `Game` on that map. `start_thirst_min=0.3` makes early deaths likely, so the event log has something in it.

**`assert len(session.tributes) == 24`.** The roster matches the default `num_players`.

**`session.place_loot(20, 20, ResourceKind.WEAPON, 1, 0.9)`.** Adds a `LootSpec` to the scenario. Not asserted directly; it is exercised when the game is built.

**`session.move_tribute(0, 20, 21)` then `session.tribute(0).name = "Rue"`.** Drags tribute 0's podium to `(20, 21)` and renames them on the spec.

**`assert session.tribute_at(20, 21) == 0`.** With no game loaded, `positions()` reads podiums from the roster, and the nearest within Chebyshev distance 1.5 of `(20, 21)` is tribute 0.

**`assert session.tribute_at(0, 0) is None or session.tribute_at(0, 0) != 0`.** Tribute 0 is nowhere near the corner. Either nobody is there or it is somebody else. This guards against `tribute_at` always returning the first id.

**`session.new_game()`.** Builds a `Game` from the config plus a frozen copy of the scenario (painted map, loot, roster), attaches a `BehaviorTelemetry` to it, and starts a `Recorder` on it. `playhead` is 0.

**`session.playing = True; session.ticks_per_second = 1000; session.update(0.05)`.** One UI frame of 0.05 seconds at 1000 ticks per second asks for 50 ticks. `update` caps a single call at a budget of 50, so at most 50 ticks run.

**`assert 0 < session.playhead <= 50`.** Some ticks ran and the playhead followed the live edge, but no more than the budget. A failure would mean the accumulator or the budget cap was wrong, which in the real dashboard shows up as a frozen window.

**`session.run_to_end()` then `assert session.recording.result is not None`.** The rest of the game is simulated instantly, the result is filled in, and the game's telemetry summary is stored for the Research tab.

**`events = len(result.eliminations) + len(result.gifts)` then `assert len(session.event_log(3)) == min(3, events)`.** After `run_to_end` the playhead is on the last frame, so the log covers the whole game. Asking for the last 3 lines returns 3, or fewer if the game had fewer events.

**`session.seek(3)` then `assert session.current_frame.tick == 3`.** Scrubbing to frame index 3 shows tick 3, because frames are stored one per tick from tick 0.

**`assert len(session.event_log()) <= events`.** With the playhead at tick 3, the log only includes frames 0 to 3, so it can never exceed the whole game's event count. The default `last` is 12.

**`session.save_scenario(...)`, `session.save_replay(...)`, `session.save_config(...)`.** Three files: the scenario as JSON (with the painted terrain), the recording as pickle, and the config as JSON.

**`other = Session(SimulationConfig(seed=9, width=50, height=50))` and the three `load_*` calls.** A fresh session with a different seed adopts everything. `load_scenario` loads the map and roster. `load_replay` then drops any live game, loads the replay's terrain into the painter, adopts the replay's config, and rebuilds the roster from the recording's own roster so names match the dots. `load_config` finally replaces the config with `c.json`.

**`assert other.tribute(0).name == "Rue"`.** The rename survived both the scenario JSON and the recording's roster, because the game was built after the rename.

**`assert other.recording.length == session.recording.length`.** The replay loaded with every frame.

**`assert other.config.max_days == 6`.** The config JSON round trip restored the first session's `max_days`, replacing the second session's default of 24. The value 6 is the one passed to the first `Session`.

### `test_session_training_and_champion()`

**Setup.** `Session(SimulationConfig(seed=5, width=40, height=40, max_days=3))`, the smallest and shortest game in the file.

**`session.start_training(TrainingConfig(brain_name="voting", population_size=24, generations=1, rounds_per_generation=1, seed=0))`.** The second argument, `method`, defaults to `"genetic"`, so this builds a `GeneticTrainer` on the painted map and starts a daemon thread running `trainer.run`. One generation, one round, 24 genomes: one evaluation game, plus the trainer's default two validation games.

**`session._training_thread.join(timeout=60)`.** Wait for the thread. Sixty seconds is a generous ceiling; it actually takes well under a second.

**`assert not session.training_running`.** The thread finished.

**`assert len(session.training_history()) == 1`.** One `GenerationStats`.

**`assert session.give_champion([0, 1]) == 2`.** The champion genome is written into the specs for tributes 0 and 1, their `brain_name` is set to the trained kind, and the method returns how many it touched.

**`assert session.tribute(0).genome is not None and session.tribute(2).genome is None`.** Only the named tributes were changed.

**`session.new_game()` then `assert session.game.players[0].brain.name == "voting"`.** The next game builds tribute 0's brain from the spec's `brain_name`, which `give_champion` set to the trained kind.

**`assert np.allclose(session.game.players[0].brain.genome(), session.trainer.champion)`.** The brain in the running game carries the champion's genes. This is the whole point of the Train tab: evolve, hand over, watch.

## How to run and extend

```bash
python -m pytest tests/test_ui_session.py
python -m pytest tests/test_ui_session.py -v
python -m pytest tests/test_ui_session.py -k "session and not training"
python -m pytest tests/test_ui_session.py::test_painter_brush_stamps_and_presets
```

**1. Unknown presets raise.**

```python
import pytest

def test_unknown_preset_raises():
    with pytest.raises(KeyError):
        MapPainter(20, 20).apply_preset("mordor", SimulationConfig(width=20, height=20))
```

**2. Adding and removing tributes keeps `num_players` in step.**

```python
def test_add_and_remove_tribute():
    session = Session(SimulationConfig(seed=1, width=40, height=40))
    spec = session.add_tribute()
    assert spec.player_id == 24 and session.config.num_players == 25
    session.remove_tribute(spec.player_id)
    assert session.config.num_players == 24
```

**3. Loot cannot be placed in the void.**

```python
def test_loot_in_void_is_refused():
    session = Session(SimulationConfig(seed=1, width=40, height=40))
    session.painter.carve_round()
    session.place_loot(0, 0, ResourceKind.FOOD, 1, 0.5)
    assert session.scenario.loot == []
```

**4. REINFORCE training through the session.** The same thread path with `method="reinforce"` and an `RLConfig`.

```python
from hunger_games.training import RLConfig

def test_session_reinforce_training():
    session = Session(SimulationConfig(seed=5, width=40, height=40, max_days=3))
    session.start_training(RLConfig(epochs=1, episodes_per_epoch=1, validation_games=0, seed=0), method="reinforce")
    session._training_thread.join(timeout=120)
    assert len(session.training_history()) == 1 and session.give_champion([0]) == 1
    assert session.tribute(0).brain_name == "neural"
```

## Gotchas

**`(x, y)` in, `[y, x]` out.** `painter.paint(10, 10, ...)` and `painter.terrain[10, 13]` look similar but the second is row 10, column 13. Mixing them up makes a test pass on square symmetric shapes and fail on real maps.

**`Session()` builds a `Game` on construction.** `generate_roster` runs a throwaway game to roll names, scores and podiums. Constructing a session is therefore as expensive as building an arena.

**The playback budget is 50 ticks per `update`.** `ticks_per_second = 1000` with `update(0.05)` asks for 50 exactly; a bigger `seconds` value would still stop at 50 and carry the rest in `_accumulator`.

**`event_log` depends on the playhead.** It walks frames up to the playhead only. Seek first, then read.

**`load_replay` rewrites the roster and the config.** It rebuilds `scenario.tributes` from the recording and sets `session.config` to the recording's config. In the test that is harmless because the replay came from the same session, but loading a stranger's replay replaces your edits.

**`load_config` does not rebuild the roster or the map.** It only replaces `session.config`.

**Training runs on a daemon thread.** If a test forgets to `join`, pytest may finish while the thread is still working, and assertions about `training_history()` become flaky. Always join or poll `training_running`.

**`give_champion` also changes `config.neural`.** For a neural champion the architecture must match, so the session copies the trainer's neural config into its own. With a voting champion this is harmless. For a REINFORCE trainer the brain kind is always `"neural"`.

**No Dear PyGui needed.** Neither `painter.py` nor `session.py` imports it. `renderer.py` is imported for GIF export, so matplotlib is loaded, but no window is opened.
