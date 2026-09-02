# `session.py`

**Source:** [hunger_games/ui/session.py](../../hunger_games/ui/session.py)
**Depends on:** `json`, `threading`, `pathlib.Path`, `numpy`; project modules [../arena.md](../arena.md) (`Arena`), [../brain/neural.md](../brain/neural.md) (`MENU_NAMES`, `NeuralBrain`), [../config.md](../config.md) (`SimulationConfig`), [../districts.md](../districts.md) (`SEXES`, `default_tribute_name`), [../game.md](../game.md) (`Game`), [../recorder.md](../recorder.md) (`Frame`, `Recorder`, `Recording`), [../renderer.md](../renderer.md) (`export_recording_gif`), [../research/experiments.md](../research/experiments.md) (`Sweep`, `SweepConfig`), [../research/plots.md](../research/plots.md) (`behaviour_plots`), [../research/telemetry.md](../research/telemetry.md) (`BehaviorTelemetry`), [../resources.md](../resources.md) (`CornucopiaLayout`, `ResourceKind`, `RingLayout`), [../scenario.md](../scenario.md) (`LootSpec`, `Scenario`, `TributeSpec`), [../terrain.md](../terrain.md) (`TerrainType`), [../training/init.md](../training/init.md) (`GeneticTrainer`, `ReinforceTrainer`, `RLConfig`, `TrainingConfig`, `save_run`), [painter.md](painter.md) (`MapPainter`)
**Used by:** [app.md](app.md) (every widget callback), [canvas.md](canvas.md) (reads `painter`, `positions()`, `current_frame`, `tributes`, `selected_id`, `recording`, `scenario.loot`), [screenshots.md](screenshots.md) (paints the demo stroke and waits for training), `tests/test_ui_session.py`, `tests/test_feed.py`

## Purpose

`Session` is everything the dashboard is doing, with no GUI code in it. It owns the config, the painted map, the roster, the hand-placed loot, the current game and its recording, the playback clock, the selected tribute, the behaviour telemetry of watched games, the trainer (genetic or reinforce), the training feed that shows training as it happens, and the parameter sweep. Every button in [app.md](app.md) calls one method here; every frame the panels are redrawn from this object. Because there is no Dear PyGui in it, the whole workflow can be run and tested from a script.

## Concepts you need

- **Scenario as the editing buffer.** The roster and the hand-placed loot live in `self.scenario`, a [`Scenario`](../scenario.md). The painted map lives in `self.painter`. `_scenario_for_game` combines them into a frozen copy for each game.
- **Recording and playhead.** A game is not played directly; a `Recorder` steps it and keeps every tick as a `Frame`. `playhead` is the frame on screen. Playing at the newest frame simulates a new tick; playing behind it just moves the playhead. See [../recorder.md](../recorder.md).
- **Ticks per second and the accumulator.** The UI calls `update(seconds)` every frame. Fractional ticks are carried in `_accumulator` so slow speeds like 0.5 ticks per second work.
- **Telemetry.** A `BehaviorTelemetry` hooks into each game and tallies what tributes chose against how thirsty, hungry, hurt and threatened they were. Its `summary()` is a plain dictionary that can be merged across games and drawn by [../research/plots.md](../research/plots.md).
- **Threads.** Training and sweeps run in a `threading.Thread` marked `daemon=True`, so closing the window ends them. The GUI polls `training_progress`, `sweep_progress`, `status` and the trainer's `history` each frame. Trainers with more than one worker also start worker processes.
- **Genomes.** Both trainers produce a champion as a flat numpy vector. A `TributeSpec.genome` stores it as a list, and the game rebuilds the brain from it.
- **Showcase recordings and the training feed.** Both trainers keep a `showcase` on every step's stats object: a full `Recording` of one real game from that step (the first evaluation job's game for genetic, the first training episode for reinforce), made when `record_showcase` is on, which it is by default in `TrainingConfig` and `RLConfig`. The training feed replays those recordings, or lets the newest champion play live, one step at a time, without the GUI knowing anything about trainers.
- **Learner slots.** A trainer does not drive every tribute. `trainer._learner_ids()` names the roster slots the champion (genetic: a quarter of the roster, spread out) or the policy (reinforce: `learners_per_game` slots, spread out) plays in; the rest use the config's default brain. The feed's `live` mode gives the champion to those same slots.

## Walkthrough

### `class Session`

"The dashboard's state and every operation the buttons perform."

#### `Session.__init__`

`def __init__(self, config: SimulationConfig | None = None) -> None`

Takes a config or builds the default `SimulationConfig()`. Creates the `MapPainter` at the config's size and a `Scenario(title="Dashboard scenario")`. Initial state: `game = recorder = recording = None`, `playhead = 0`, `playing = False`, `ticks_per_second = 8.0`, `_accumulator = 0.0`, `selected_id = None`, `trainer = None`, `training_method = "genetic"`, `telemetry = None`, `watched_summaries = []`, `_summary_stored = False`, `sweep = None`, `_sweep_thread = None`, `sweep_progress = (0, 0)`, `feed_mode = "off"`, `_feed_steps_seen = 0`, `feed_label = ""`, `_training_thread = None`, `training_progress = (0, 0)`, `status = "Ready"`. Then it calls `generate_arena()` and `generate_roster()`, so a fresh session already has a Perlin map and 24 tributes on podiums.

#### `Session.generate_arena`

`def generate_arena(self, seed: int | None = None) -> None`

Resizes the painter if its size differs from `config.width` and `config.height`, then loads the `perlin` preset with the given seed (or the config's). Status: "Generated a new arena".

#### `Session.apply_preset`

`def apply_preset(self, name: str) -> None`

Loads one of `MapPainter.PRESETS` into the painter at its current size. Status: "Loaded preset '<name>'".

#### `Session.paint`

`def paint(self, x: int, y: int, terrain: TerrainType, radius: int) -> None`

One brush dab, only if the cell is on the grid. Heights are not recomputed here.

#### `Session.finish_painting`

`def finish_painting(self) -> None`

Calls `painter.finish()` to recompute heights after a stroke. The dashboard calls it on mouse release.

#### `Session.generate_roster`

`def generate_roster(self) -> None`

Builds a throwaway `Game` on the painted terrain so it rolls names, scores, brains and podiums, then turns each player into a `TributeSpec` with `podium=(x, y)`. Clears the selection. Status: "Generated a new roster".

#### `Session.tributes`

`@property def tributes(self) -> list[TributeSpec]`

`scenario.tributes` or an empty list.

#### `Session.tribute`

`def tribute(self, player_id: int) -> TributeSpec | None`

One roster entry by id.

#### `Session.add_tribute`

`def add_tribute(self) -> TributeSpec`

Appends a tribute with the next id, district `(id // 2) % 12 + 1`, sex alternating from `SEXES`, the default name for that district and sex, training score 6, survival 0.5, the config's default brain, and a podium at the map centre. Sets `config.num_players` to the roster size.

#### `Session.remove_tribute`

`def remove_tribute(self, player_id: int) -> None`

Removes them, keeps `config.num_players` at least 2, and deselects if they were selected.

#### `Session.move_tribute`

`def move_tribute(self, player_id: int, x: int, y: int) -> None`

Sets the podium to `(x, y)` if the tribute exists, the cell is on the grid and it is not void.

#### `Session.PODIUM_PRESETS`

```python
PODIUM_PRESETS = ("edge ring", "around cornucopia", "random", "two sides")
```

The names accepted by `arrange_podiums` and shown in the Tributes tab's combo.

#### `Session._arena_for_podiums`

`def _arena_for_podiums(self) -> Arena`

A throwaway `Arena(config, rng(0), terrain=painter.terrain)` used for snapping and layout podiums on the painted map.

#### `Session.arrange_podiums`

`def arrange_podiums(self, preset: str) -> None`

Places every podium by a preset, then spreads the strongest tributes apart. Positions per preset:

| Preset | Positions |
| --- | --- |
| `edge ring` | `RingLayout().spawn_positions(arena, count)`: along the outer edge |
| `around cornucopia` | `CornucopiaLayout().spawn_positions(arena, count)`: a tight circle round the middle |
| `random` | Random cells (unseeded), each snapped with `arena.snap_to_podium` |
| `two sides` | Even indices at `x = 3`, odd at `x = width - 4`, `y` spread from 3 to `height - 3`, snapped |

Any other name raises `KeyError`. Then the roster is ranked by training score; the top third (at least one) is placed at evenly spaced slots and the rest fill the gaps in order. Slot `i` gets position `i`. Status: "Podiums arranged: <preset>". Does nothing for an empty roster.

#### `Session.reposition_off_void`

`def reposition_off_void(self) -> None`

For every tribute whose podium is `None` or not `arena.is_walkable`, snaps it to the nearest podium cell (from the centre if there was no podium). Called after painting, carving, stamping, presets and shape changes.

#### `Session.apply_config_change`

`def apply_config_change(self, what: str) -> None`

Reacts to a setting change so the arena on screen matches:

| `what` | Effect |
| --- | --- |
| `"size"` or `"shape"` | `generate_arena()` then `reposition_off_void()` |
| `"layout"` | `arrange_podiums("edge ring")` if the layout is `ring`, else `"around cornucopia"` |
| `"players"` | `generate_roster()` |

Other strings do nothing. The dashboard's `_setter(..., react=...)` passes these.

#### `Session.tribute_at`

`def tribute_at(self, x: int, y: int, radius: float = 1.5) -> int | None`

The id of the tribute nearest to a cell (Chebyshev distance) within `radius`, using `positions()`. Ties go to the last one checked.

#### `Session.positions`

`def positions(self) -> dict[int, tuple[int, int]]`

Living tributes' positions from the current frame if one is on screen, else every podium from the roster.

#### `Session.place_loot`

`def place_loot(self, x: int, y: int, kind: ResourceKind, quantity: int, quality: float) -> None`

Replaces any hand-placed stack at the cell with a new `LootSpec`. Ignored off the grid or on void.

#### `Session.remove_loot`

`def remove_loot(self, x: int, y: int) -> None`

Removes the hand-placed stack at a cell, if any.

#### `Session.clear_loot`

`def clear_loot(self) -> None`

Empties `scenario.loot`.

#### `Session._scenario_for_game`

`def _scenario_for_game(self) -> Scenario`

A frozen copy for a game: the painted terrain as lists, `use_layout_loot`, a copy of the loot list, a copy of each `TributeSpec` (or `None` if the roster is empty), and the title.

#### `Session.new_game`

`def new_game(self, seed: int | None = None) -> Game`

Sets `config.num_players` to the roster size, builds a config copy with `seed` (or the config's seed), builds the `Game` from `_scenario_for_game()`, attaches a fresh `BehaviorTelemetry`, creates the `Recorder` (frame 0 is captured at once), points `recording` at it, rewinds, stops playback and resets the accumulator. Status: "New game, seed <n>".

#### `Session.current_frame`

`@property def current_frame(self) -> Frame | None`

The frame at the playhead, after clamping the playhead into the recording. `None` when nothing is loaded.

#### `Session.at_live_edge`

`@property def at_live_edge(self) -> bool`

True when a recording exists and the playhead is on its last frame. True for finished games and loaded replays too.

#### `Session.step_once`

`def step_once(self) -> None`

At the live edge of a game that is still running: `recorder.step()`, follow the new frame, and `_store_summary()`. At the live edge of a finished game or replay: `playing = False`. Behind the edge: `playhead += 1`.

#### `Session.FEED_MODES`

```python
FEED_MODES = ("off", "replay", "live")
```

The values `feed_mode` may take, and the choices in the Train tab's "Training feed" radio button.

| Mode | What the arena shows after every training step |
| --- | --- |
| `off` | Nothing changes; training only fills the plots |
| `replay` | The step's showcase recording: one real evaluation game from that step, the population playing itself |
| `live` | A fresh game in which the newest champion drives the trainer's learner slots, so the Network tab shows real activations |

#### `Session._feed_ready_for_next`

`def _feed_ready_for_next(self) -> bool`

Whether the arena is free for the next step. True with nothing loaded; for a loaded recording without a live game (a replay), true once `playhead >= recording.length - 1`; for a live game, true once `game.is_over` and the playhead is at the live edge. So a step is never shown while the previous one is still being watched.

#### `Session._advance_feed`

`def _advance_feed(self) -> None`

Called at the start of every `update`. Returns at once when `feed_mode == "off"` or there is no trainer. Otherwise it looks at `trainer.history`: if it holds no more steps than `_feed_steps_seen`, or `_feed_ready_for_next()` is false, nothing happens. Otherwise `_feed_steps_seen` catches up to the newest step (intermediate steps that finished while a game was being watched are skipped, not queued) and the newest step is shown. `step_name` is "generation" for genetic and "epoch" for reinforce; the step number in the labels is `len(history) - 1`.

| Mode | Action | `feed_label` |
| --- | --- | --- |
| `replay`, showcase present | `load_recording(history[-1].showcase)` | "training feed: replaying a real generation 3 game" |
| `replay`, showcase is `None` (`record_showcase` off) | Nothing loaded; returns before playing | "training feed: generation 3 has no recording" |
| `live` | `start_champion_game(all_slots=False)` | "training feed: generation 3 champion playing live" |

Then `playing = True`, at the current `ticks_per_second`. The dashboard prints `feed_label` in front of the headline while the feed is not `off`.

#### `Session.load_recording`

`def load_recording(self, recording: Recording) -> None`

Watches a `Recording` object. Sets `recording`, drops the live game (`game = recorder = telemetry = None`), shows the map it was played on (`painter.load(recording.terrain, recording.heights)`), adopts `recording.config`, and rebuilds `scenario.tributes` from `recording.roster` so names, districts, sexes, scores and brain names match the markers, with each podium set to the tribute's position in frame 0. Deselects, rewinds, stops. The hand-placed loot list is left as it was. Used by `load_replay` and by the feed's `replay` mode.

#### `Session.start_champion_game`

`def start_champion_game(self, all_slots: bool = True) -> bool`

Gives the trainer's champion to the roster and starts a game. Without a trainer or a champion: status "No champion yet: train first", returns `False`. With `all_slots=True` every tribute gets it (`give_champion(None)`); with `all_slots=False` only `trainer._learner_ids()` do, and the other tributes keep the brains and genomes they already have. Then `new_game()`, `playing = True`, returns `True`. Note that `give_champion` also sets `config.neural` to the trainer's architecture.

#### `Session.genome_history`

`def genome_history(self) -> list[np.ndarray]`

The champion genome after every step so far: `stats.champion` per generation for genetic (that generation's best, not the best ever), `stats.genome` per epoch for reinforce (the policy after that epoch's update). `[]` without a trainer.

#### `Session.network_evolution`

`def network_evolution(self, max_genes: int = 200) -> dict | None`

Stacks `genome_history()` into a steps-by-genes matrix and returns `None` when it is empty. Otherwise:

| Key | Value |
| --- | --- |
| `steps` | `[0, 1, ..., n - 1]` |
| `change` | The L2 norm of `genome[i] - genome[i - 1]` per step; `0.0` for step 0 |
| `mean_abs` | The mean absolute gene value per step |
| `genes` | The matrix cut to its first `max_genes` columns, as a numpy array |
| `gene_count` | The full number of genes |

The Network tab's evolution plots draw `change` and `mean_abs` as lines and `genes` as a heat map.

#### `Session.update`

`def update(self, seconds: float) -> None`

Called every UI frame. First `_advance_feed()`, whether or not playback is running. Then, while playing, adds `seconds * ticks_per_second` to the accumulator and runs whole ticks with `step_once`, at most 50 per call so the window stays responsive.

#### `Session.run_to_end`

`def run_to_end(self) -> None`

If there is a live game, `recorder.record_all()`, jump to the last frame, stop, store the summary. Status: "Game finished". Does nothing for a loaded replay.

#### `Session._store_summary`

`def _store_summary(self) -> None`

Appends `telemetry.summary()` to `watched_summaries` once, only when the game is over.

#### `Session.watched_summary`

`def watched_summary(self) -> dict | None`

`BehaviorTelemetry.merge` of every stored summary plus the current game's tallies so far (if not yet stored). `None` before any game. The Charts tab reads this twice a second.

#### `Session.export_behaviour_plots`

`def export_behaviour_plots(self, folder: str | Path) -> int`

Writes `behaviour_plots(summary, folder)` and returns how many files. The files are `action_distribution.png`, `actions_by_thirst.png`, `actions_by_hunger.png`, `actions_by_health.png`, `instinct_curves.png`, `consumption_timing.png`, `fight_or_flight.png`, `proximity_vs_remaining.png`, `actions_by_remaining.png`, `position_heatmap.png`, `armed_vs_unarmed_heatmaps.png` and `deaths_by_cause.png`. Status reports the count, or "No games watched yet".

#### `Session.network_snapshot`

`def network_snapshot(self, player_id: int | None) -> dict | None`

What the selected tribute's network is doing now, for [visualizer.md](visualizer.md). Needs a live `game`, the id in `game.player_by_id`, a `NeuralBrain` and a `last_perception`. Returns `{"layer_sizes", "inputs", "activations", "weights", "probabilities", "chosen", "menu"}`: the perception vector, every layer's activation from `network.hidden_activations`, the weight matrices, the softmax of the last activation, `brain.last_index` and `MENU_NAMES`. Otherwise `None`.

#### `Session.seek`

`def seek(self, frame_index: int) -> None`

Moves the playhead, clamped into the recording.

#### `Session.rewind`

`def rewind(self) -> None`

`seek(0)`.

#### `Session.event_log`

`def event_log(self, last: int = 12) -> list[str]`

Lines for every elimination ("Day 3: Name out (spear by Killer), placed 12") and parachute ("Day 2: parachute for Name (medicine)") up to the playhead, newest last, trimmed to the last `last`.

#### `Session.save_scenario`

`def save_scenario(self, path: str | Path) -> None`

Saves `_scenario_for_game()` as JSON, so the painted terrain is included.

#### `Session.load_scenario`

`def load_scenario(self, path: str | Path) -> None`

Loads a scenario; if it has terrain, loads it into the painter and sets `config.width` and `config.height` to match. Adopts the loot and roster, sets `config.num_players`, deselects.

#### `Session.save_config`

`def save_config(self, path: str | Path) -> None`

Writes `config.to_dict()` as indented JSON.

#### `Session.load_config`

`def load_config(self, path: str | Path) -> None`

Replaces `config` with `SimulationConfig.from_dict(...)`. Nothing else changes; the dashboard follows up with `apply_config_change("size")`.

#### `Session.save_replay`

`def save_replay(self, path: str | Path) -> None`

Pickles the recording, if there is one.

#### `Session.load_replay`

`def load_replay(self, path: str | Path) -> None`

`load_recording(Recording.load(path))`, then status "Loaded replay from <path>". Everything it does to the map, config and roster is described under `load_recording`.

#### `Session.export_gif`

`def export_gif(self, path: str | Path, step: int = 2, fps: int = 15) -> None`

Finishes the live game with `run_to_end()` first, then `export_recording_gif(recording, path, fps=fps, step=step)`. `step` is ticks per GIF frame.

#### `Session.training_running`

`@property def training_running(self) -> bool`

Whether the trainer thread is alive.

#### `Session.start_training`

`def start_training(self, settings: TrainingConfig | RLConfig, method: str = "genetic") -> None`

Ignored if already running. Builds a config copy with `num_players = max(2, len(tributes) or config.num_players)` and a `Scenario` holding only the painted terrain. Sets `training_method` and builds `GeneticTrainer(config, settings, scenario=scenario)` for `"genetic"`, else `ReinforceTrainer`. Starts a daemon thread that calls `trainer.run(on_progress=...)`, where the progress callback stores `(done, total)` games in `training_progress`. When the run returns, status becomes "Training stopped" or "Training finished"; an exception becomes "Training error: ...". Status while running: "Training (<method>)...". `_feed_steps_seen` is reset to 0 and `feed_label` cleared, so the feed starts from the new run's first step.

The trainers are documented in [../training/genetic.md](../training/genetic.md) and [../training/reinforce.md](../training/reinforce.md). The genetic trainer scores whole games by placement; reinforce rewards every action with `config.reward` and trains the neural brain only.

#### `Session.stop_training`

`def stop_training(self) -> None`

`trainer.stop()`: finish the current generation or epoch, then return.

#### `Session.training_history`

`def training_history(self) -> list`

The trainer's stats objects so far (`GenerationStats` or `EpochStats`), or `[]`.

#### `Session.training_rows`

`def training_rows(self) -> list[dict]`

`trainer.history_rows()`: plain dictionaries without genomes, telemetry or showcase recordings. Genetic rows have `generation`, `best_fitness`, `mean_fitness`, `worst_fitness`, `seconds`, `val_fitness`, `cumulative_seconds`. Reinforce rows have `epoch`, `policy_loss`, `value_loss`, `entropy`, `train_return`, `val_return`, `train_survival`, `val_survival`, `win_rate`, `val_win_rate`, `kill_rate`, `seconds`, `cumulative_seconds`. The Train tab plots these.

#### `Session.champion_genes`

`def champion_genes(self) -> tuple[np.ndarray, np.ndarray] | None`

The latest step's genome and a boolean mask of genes that changed since the step before. Genetic: `history[-1].champion` against `trainer.previous_champion()`. Reinforce: `history[-1].genome` against `history[-2].genome`. A gene counts as changed when it moved by more than `1e-9`; on the first step every gene is marked changed. `None` without history.

#### `Session.save_training_run`

`def save_training_run(self, name: str, results_dir: str | Path = "results") -> Path | None`

`save_run(trainer, training_method, name, results_dir)`: writes `results/<name>_<timestamp>/` with `config.json`, `history.json`, `champion.json` and a `plots/` folder. Genetic plots: `fitness.png`, `fitness.gif`, `timing.png`. Reinforce plots: `reward.png`, `losses.png`, `entropy.png`, `survival.png`, `win_kill_rate.png`, `reward.gif`, `timing.png`. When telemetry was collected, both add `action_distribution_over_training.png`, `death_needs_over_training.png`, `behaviour_over_training.png` and the twelve behaviour charts of the last step. Returns the folder, or `None` (status "Nothing trained yet").

#### `Session.sweep_running`

`@property def sweep_running(self) -> bool`

Whether the sweep thread is alive.

#### `Session.start_sweep`

`def start_sweep(self, settings: SweepConfig) -> None`

Ignored if running. Builds `Sweep(config, settings, scenario=<terrain only>)`, sets `sweep_progress = (0, len(values))` and runs it in a daemon thread. The sweep plays `games_per_value` seeded games per value, writes `results/<name>_<timestamp>/` with `config.json`, `results.csv`, `summary.json` and `plots/`, and reports each finished value into `sweep_progress`. Status on success: "Sweep saved to <folder>".

#### `Session.stop_sweep`

`def stop_sweep(self) -> None`

`sweep.stop()`: finish the current value, write the folder, return.

#### `Session.give_champion`

`def give_champion(self, player_ids: list[int] | None = None) -> int`

Loads `trainer.champion` into some tributes (all by default) and returns how many. The brain kind is `trainer.training.brain_name` for genetic, always `"neural"` for reinforce. Sets each spec's `brain_name` and `genome`, and sets `config.neural = trainer.config.neural` so the architecture matches. Returns 0 with no trainer or no champion.

#### `Session.save_champion`

`def save_champion(self, path: str | Path) -> None`

Genetic: `trainer.save_champion(path)`. Reinforce: `trainer.save_policy(path)`, which writes the same keys plus `value_genome`, `value_hidden`, `epochs` and `method`.

#### `Session.load_champion_into`

`def load_champion_into(self, path: str | Path, player_ids: list[int] | None = None) -> int`

Reads a champion file with `GeneticTrainer.load_champion` (works for both formats), sets `config.neural` from it, and gives its `brain_name` and genome to the targets.

## How to use it / experiment

**A whole game without a window.**

```python
from hunger_games.ui.session import Session
s = Session()
s.new_game(seed=1)
s.run_to_end()
print(s.status, s.recording.result.winner_name)
print(s.event_log(5))
print(s.watched_summary()["entropy"])
```

**Edit a roster and favour one tribute.**

```python
spec = s.tributes[0]
spec.name, spec.weapon_quality, spec.favor_bonus = "Katniss", 0.9, 0.5
s.move_tribute(spec.player_id, 60, 10)
s.arrange_podiums("two sides")     # or leave the drag in place
s.save_scenario("mine.json")
```

**Train and watch the champion.**

```python
from hunger_games.training import TrainingConfig, RLConfig
s.start_training(TrainingConfig(generations=3, population_size=24, workers=1), "genetic")
s._training_thread.join()
print(s.training_rows()[-1]["best_fitness"], s.champion_genes()[1].sum(), "genes changed")
s.give_champion(); s.new_game(); s.run_to_end()
s.save_training_run("demo")
s.start_training(RLConfig(epochs=2, episodes_per_epoch=2), "reinforce")
```

**Drive the training feed by hand.** The feed only moves inside `update`, so call it in a loop as the dashboard does.

```python
s = Session()
s.feed_mode = "replay"
s.start_training(TrainingConfig(brain_name="voting", population_size=24, generations=3,
                                rounds_per_generation=1, validation_games=0), "genetic")
while s.training_running or s.recording is None:
    s.update(0.05)                       # shows the first showcase when a step exists
print(s.feed_label)                      # training feed: replaying a real generation 0 game
s.feed_mode = "live"
s.playhead = s.recording.length - 1      # finish watching so the arena is free
s.update(0.05)                           # the champion now plays live in the learner slots
print(s.feed_label, s.game is not None)
print(s.network_evolution()["change"])   # how far the champion moved each step
```

**Add a podium preset.** Add the name to `PODIUM_PRESETS` and an `elif preset == ...` branch that builds a list of `count` positions, snapping each with `arena.snap_to_podium`. The strength spreading and the combo pick it up automatically.

**Add a config reaction.** Add a branch to `apply_config_change` and pass its name as `react=` in the dashboard's `_setter`.

## Gotchas

- Trainers and sweeps get the painted map only. The hand-placed loot and the edited roster (names, granted items, podiums) are not used during training; `new_game` uses all of them.
- `start_training` replaces `self.trainer`, so a second run drops the earlier history and champion unless you saved them.
- The feed's `replay` mode goes through `load_recording`, which replaces the roster with the training game's roster (the trainer's generated tributes, on their frame-0 positions), replaces `config` with the trainer's config copy, and drops the live game. Save your scenario before turning it on.
- The feed's `live` mode changes the roster too: `give_champion` writes `brain_name` and `genome` into the learner slots and sets `config.neural`.
- The feed skips steps. If three generations finish while one game is being watched, only the newest is shown next; `_feed_steps_seen` jumps to the newest.
- `feed_label` is never cleared. Switching the feed back to `off` hides it from the headline, but the string remains.
- Showcase games from the `replay` feed have no telemetry (`load_recording` sets `telemetry = None`), so they never reach `watched_summaries` or the Charts tab.
- `training_progress` counts games within the current step, not steps; the step count comes from `training_history()`.
- `champion_genes` compares the latest step to the step before, while `trainer.champion` (used by `give_champion` and `start_champion_game`) is the best step ever for genetic and the best validation return for reinforce. The bar chart and the brain you hand out can differ. `genome_history` follows the per-step champion, like the bar chart.
- `load_recording` (so `load_replay` and the replay feed) replaces `config` with the recording's, and `load_config` replaces it too. Any code holding the old config object keeps the old one.
- `at_live_edge` is also true for a loaded replay, where `recorder` is `None`; `step_once` then just stops playback at the last frame.
- A game abandoned midway (New game pressed before it ended) never reaches `watched_summaries`; only finished games count in the charts and exports. `watched_summary()` does include the unfinished current game while it plays.
- `network_snapshot` reads the live `Game`, not the frame at the playhead, so it always shows the newest decision even while scrubbing backwards, and returns `None` for loaded replays, including replay-feed games.
- `positions()` drops the dead when a frame is on screen, so clicking where someone fell selects nobody.
- `arrange_podiums("random")` uses an unseeded generator; the same button gives a different layout each press.
- The status string is written from the training and sweep threads. It is a plain assignment, which is safe in CPython, but the message can be overwritten by the next button press.
- `start_champion_game(all_slots=False)` calls the trainer's private `_learner_ids()`, which reads the trainer's `num_players`. If the roster grew or shrank since training started, the slots may not line up with the roster you see.
- Replays are pickles (`Recording.load`). Only open replay files you made yourself.
