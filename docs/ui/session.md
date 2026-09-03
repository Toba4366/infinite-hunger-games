# `session.py`

**Source:** [hunger_games/ui/session.py](../../hunger_games/ui/session.py)
**Depends on:** `json`, `threading`, `time`, `pathlib.Path`, `numpy`; project modules [../arena.md](../arena.md) (`Arena`), [../brain/neural.md](../brain/neural.md) (`MENU_NAMES`, `MENU_SIZE`, `NeuralBrain`), [hunger_games/brain/neat.py](../../hunger_games/brain/neat.py) (`NeatBrain`, imported inside `network_snapshot`), [../config.md](../config.md) (`SimulationConfig`), [../districts.md](../districts.md) (`SEXES`, `default_tribute_name`), [../game.md](../game.md) (`Game`), [../recorder.md](../recorder.md) (`Frame`, `Recorder`, `Recording`), [../renderer.md](../renderer.md) (`export_recording_gif`), [../research/experiments.md](../research/experiments.md) (`Sweep`, `SweepConfig`), [../research/plots.md](../research/plots.md) (`behaviour_plots`), [../research/telemetry.md](../research/telemetry.md) (`BehaviorTelemetry`), [../resources.md](../resources.md) (`CornucopiaLayout`, `ResourceKind`, `RingLayout`), [../scenario.md](../scenario.md) (`LootSpec`, `Scenario`, `TributeSpec`), [../terrain.md](../terrain.md) (`TerrainType`), [../training/init.md](../training/init.md) (`Curriculum`, `CurriculumConfig`, `GeneticTrainer`, `ImitationTrainer`, `NeatTrainer`, `PPOTrainer`, `ReinforceTrainer`, `SystemMonitor`, `save_run`), [painter.md](painter.md) (`MapPainter`)
**Used by:** [app.md](app.md) (every widget callback and every per-frame refresh), [canvas.md](canvas.md) (reads `painter`, `positions()`, `current_frame`, `tributes`, `selected_id`, `recording`, `scenario.loot`, `learner_ids_on_screen()`), [screenshots.md](screenshots.md) (paints the demo stroke and waits for training), `tests/test_ui_session.py`, `tests/test_feed.py`

## Purpose

`Session` is everything the dashboard is doing, with no GUI code in it. It owns the config, the painted map, the roster, the hand-placed loot, the current game and its recording, the playback clock, the selected tribute, the behaviour telemetry of watched games, the trainer (imitation, genetic, NEAT, reinforce or PPO), the training loop with its pause flag, the training feed that shows training as it happens, a system monitor, and the parameter sweep. Every button in [app.md](app.md) calls one method here; every frame the panels are redrawn from this object. Because there is no Dear PyGui in it, the whole workflow can be run and tested from a script.

## Concepts you need

- **Scenario as the editing buffer.** The roster and the hand-placed loot live in `self.scenario`, a [`Scenario`](../scenario.md). The painted map lives in `self.painter`. `_scenario_for_game` combines them into a frozen copy for each game.
- **Recording and playhead.** A game is not played directly; a `Recorder` steps it and keeps every tick as a `Frame`. `playhead` is the frame on screen. Playing at the newest frame simulates a new tick; playing behind it just moves the playhead. See [../recorder.md](../recorder.md).
- **Ticks per second and the accumulator.** The UI calls `update(seconds)` every frame. Fractional ticks are carried in `_accumulator` so slow speeds like 0.5 ticks per second work.
- **Telemetry.** A `BehaviorTelemetry` hooks into each game and tallies what tributes chose against how thirsty, hungry, hurt and threatened they were. Its `summary()` is a plain dictionary that can be merged across games and drawn by [../research/plots.md](../research/plots.md).
- **One learner, five methods.** Every trainer trains one learner network that plays a few roster slots against voting opponents. The five trainers (`ImitationTrainer`, `GeneticTrainer`, `NeatTrainer`, `ReinforceTrainer`, `PPOTrainer`) share one interface: a constructor `(config, settings, scenario=, initial_genome=, curriculum=)`, `step(on_progress)` for one iteration, `stop()`, a `_stop` flag, `history` (method-specific stats), `learning_history` (a list of `IterationStats`, the same shape for every method), `events` (an `EventLog`), `settings`, `champion`, `save_champion(path)` and `_learner_ids()`. The session only calls these, so most methods here do not care which trainer is running.
- **The loop lives here.** Trainers also have `run()`, but the session does not use it. `start_training` writes its own loop in the thread body: while the iteration count is below the target and `_stop` is clear, either sleep (paused) or call `trainer.step(progress)`. That is what makes Pause possible between iterations.
- **`IterationStats`.** Each iteration records `iteration`, `scores` (one per learner episode), `mean_score`, `best_score`, `entropy`, `mean_length`, `win_rate`, `val_score`, `seconds`, `cumulative_seconds`, `stage`, `opponents`, `extra` (method-specific numbers), `learner` (the network after the iteration), `telemetry` and `showcase` (a recording of one real episode). `to_row()` drops the big fields and flattens `extra` into `extra_*` keys. See [../training/init.md](../training/init.md).
- **Genomes: arrays and dictionaries.** Neural learners (imitation, genetic on `neural`, reinforce, PPO) have a flat numpy vector as their genome. The genetic trainer on `voting` has an eight-gene vector. A NEAT learner's genome is a dictionary (nodes and connections with innovation numbers), and its "weights" are the fourth field of each connection entry, `c[3]`. Everywhere the session needs one flat vector it checks `isinstance(genome, dict)` and, for NEAT, takes the connection weights. A `TributeSpec.genome` stores a list for arrays and the dictionary itself for NEAT.
- **Warm starts.** A trainer can begin from an existing genome (`initial_genome`). The genetic trainer builds its population as that genome plus close relatives; NEAT builds clones of the genome and mutates all but one; the reinforce, PPO and imitation trainers load it into their network. `warm_start_genome()` finds one; `start_training(..., warm_start=True)` passes it on when its kind matches the method. The intended order is imitation first, then PPO, reinforce, genetic or NEAT from the imitation champion.
- **Curriculum.** A `CurriculumConfig` names the opponents per stage (`(1, 3, 7, 11, 23)` by default), a promotion `threshold` on the mean score of the last `window` iterations, and `max_iterations_per_stage`. The session wraps it in a `Curriculum` only when `enabled` is true, and the trainer resizes its roster each iteration to `min(learners_per_game, 24) + opponents`. The genetic trainer applies it only in `voting` opponent mode.
- **Showcase recordings and the training feed.** Each step's stats object may carry a `showcase`, a full `Recording` of one real game from that step, made when `record_showcase` is on (the default in every settings dataclass). The training feed replays those recordings, or lets the newest champion play live, one step at a time.
- **Learner slots and stars.** `trainer._learner_ids()` names the roster slots the learner plays in: `learners_per_game` slots (6 by default for every method) spread evenly across the roster. The feed's `live` mode and Watch agent give the champion to those slots. The canvas draws a gold star on every tribute `learner_ids_on_screen()` reports.
- **Threads.** Training and sweeps run in a `threading.Thread` marked `daemon=True`, so closing the window ends them. The GUI polls `training_progress`, `sweep_progress`, `status`, the trainer's histories and events each frame. Trainers with more than one worker also start worker processes.

## Walkthrough

### `class Session`

"The dashboard's state and every operation the buttons perform."

#### `Session.__init__`

`def __init__(self, config: SimulationConfig | None = None) -> None`

Takes a config or builds the default `SimulationConfig()`. Creates the `MapPainter` at the config's size and a `Scenario(title="Dashboard scenario")`. Initial state: `game = recorder = recording = None`, `playhead = 0`, `playing = False`, `ticks_per_second = 8.0`, `_accumulator = 0.0`, `selected_id = None`, `trainer = None`, `training_method = "genetic"`, `telemetry = None`, `watched_summaries = []`, `_summary_stored = False`, `sweep = None`, `_sweep_thread = None`, `sweep_progress = (0, 0)`, `feed_mode = "off"`, `_paused = False`, `_max_iterations = 0`, `system = SystemMonitor()`, `_feed_steps_seen = 0`, `feed_label = ""`, `_training_thread = None`, `training_progress = (0, 0)`, `status = "Ready"`. Then it calls `generate_arena()` and `generate_roster()`, so a fresh session already has a Perlin map and 24 tributes on podiums. The `trainer` attribute is annotated `GeneticTrainer | ReinforceTrainer | None` but holds any of the five trainers.

#### `Session.generate_arena`

`def generate_arena(self, seed: int | None = None) -> None`

Resizes the painter if its size differs from `config.width` and `config.height`, then loads the `perlin` preset with the given seed (or the config's). Status: "Generated a new arena".

#### `Session.apply_preset`

`def apply_preset(self, name: str) -> None`

`painter.apply_preset(name, config)`; status "Loaded preset '<name>'".

#### `Session.paint`, `Session.finish_painting`

`def paint(self, x: int, y: int, terrain: TerrainType, radius: int) -> None`, `def finish_painting(self) -> None`

Paint with the brush when the cell is on the grid; recompute heights after a stroke.

#### `Session.generate_roster`

`def generate_roster(self) -> None`

Builds a throwaway `Game` on the painted terrain so it rolls names, scores, brains and podiums, then turns each player into a `TributeSpec`. Clears the selection. Status: "Generated a new roster".

#### `Session.tributes`, `Session.tribute`

`@property def tributes(self) -> list[TributeSpec]`, `def tribute(self, player_id: int) -> TributeSpec | None`

The editable roster (always a list) and one entry by id.

#### `Session.add_tribute`, `Session.remove_tribute`, `Session.move_tribute`

`def add_tribute(self) -> TributeSpec`, `def remove_tribute(self, player_id: int) -> None`, `def move_tribute(self, player_id: int, x: int, y: int) -> None`

Append a middling tribute (score 6, survival 0.5, the config's brain) at the map centre with the next id, cycling districts and sexes; remove one and keep `num_players` at least 2; put a podium at a walkable cell.

#### `Session.PODIUM_PRESETS`

```python
PODIUM_PRESETS = ("edge ring", "around cornucopia", "random", "two sides")
```

#### `Session._arena_for_podiums`, `Session.arrange_podiums`, `Session.reposition_off_void`

`def _arena_for_podiums(self) -> Arena`, `def arrange_podiums(self, preset: str) -> None`, `def reposition_off_void(self) -> None`

A throwaway arena on the painted map; positions by preset with the strongest third of the roster spread apart (`KeyError` for an unknown preset); nudge every podium onto a walkable cell.

#### `Session.apply_config_change`

`def apply_config_change(self, what: str) -> None`

`size` or `shape`: regenerate and re-place. `layout`: arrange podiums (`edge ring` for `ring`, else `around cornucopia`). `players`: regenerate the roster.

#### `Session.tribute_at`, `Session.positions`

`def tribute_at(self, x: int, y: int, radius: float = 1.5) -> int | None`, `def positions(self) -> dict[int, tuple[int, int]]`

The nearest tribute within `radius` cells (Chebyshev distance); positions from the frame on screen (living only) or from podiums.

#### `Session.place_loot`, `Session.remove_loot`, `Session.clear_loot`

Hand-placed supplies: place (replacing any stack there; not on void), remove at a cell, clear all.

#### `Session._scenario_for_game`

`def _scenario_for_game(self) -> Scenario`

The painted terrain, `use_layout_loot`, a copy of the loot, copies of the tribute specs, and the title.

#### `Session.new_game`

`def new_game(self, seed: int | None = None) -> Game`

Keeps `num_players` equal to the roster size, builds a config copy with the seed, builds the `Game` on `_scenario_for_game()`, attaches a `BehaviorTelemetry`, starts a `Recorder`, watches its recording from frame 0, stops playback and resets the clock. Status: "New game, seed N".

#### `Session.current_frame`, `Session.at_live_edge`

`@property def current_frame(self) -> Frame | None`, `@property def at_live_edge(self) -> bool`

The frame at the (clamped) playhead; whether the playhead is on the newest frame.

#### `Session.step_once`

`def step_once(self) -> None`

At the live edge of a running game: record one tick, follow it, store the summary if the game just ended; at the live edge of a finished game or a replay: stop playing; otherwise move the playhead forward.

#### `Session.FEED_MODES`

```python
FEED_MODES = ("off", "replay", "live")
```

#### `Session._feed_ready_for_next`

`def _feed_ready_for_next(self) -> bool`

True when nothing is loaded, when a replay has reached its last frame, or when the live game is over and fully watched.

#### `Session._advance_feed`

`def _advance_feed(self) -> None`

Returns when the feed is off or there is no trainer. Reads `trainer.history`; returns when there is nothing newer than `_feed_steps_seen` or the arena is busy. Catches up to the newest step. The step name is `generation` when `training_method == "genetic"`, else `epoch`. In `replay` mode it loads the newest step's `showcase` (or sets the label "training feed: <step> N has no recording" and returns); in `live` mode it calls `start_champion_game(all_slots=False)`. Sets `feed_label` and starts playing.

#### `Session.load_recording`

`def load_recording(self, recording: Recording) -> None`

Adopts the recording, drops the live game, recorder and telemetry, loads its map into the painter, adopts its config, rebuilds the roster from `recording.roster` with frame-0 positions as podiums, clears the selection, rewinds and stops.

#### `Session.start_champion_game`

`def start_champion_game(self, all_slots: bool = True) -> bool`

Needs `trainer.champion`; otherwise status "No champion yet: train first" and False. Gives the champion to every tribute, or only to `trainer._learner_ids()` when `all_slots` is False, starts a new game and plays it.

#### `Session.genome_history`

`def genome_history(self) -> list[np.ndarray]`

The learner after every iteration in `trainer.learning_history`, as flat vectors: a NEAT dictionary is reduced to its connection weights, anything else becomes a float array. Only vectors with the same length as the newest one are kept, because NEAT genomes grow and cannot be stacked with older, shorter ones.

#### `Session.network_evolution`

`def network_evolution(self, max_genes: int = 200) -> dict | None`

Stacks `genome_history()`; None when empty. Returns `steps`, `change` (the L2 norm of the difference from the previous step, 0 for the first), `mean_abs` per step, `genes` (the first `max_genes` columns) and `gene_count`.

#### `Session.update`

`def update(self, seconds: float) -> None`

`_advance_feed()`, then, while playing, accumulate `seconds * ticks_per_second` ticks and run whole ticks up to a budget of 50 per UI frame.

#### `Session.run_to_end`, `Session._store_summary`, `Session.watched_summary`, `Session.export_behaviour_plots`

Finish the game instantly; keep the finished game's summary once; merge every finished game plus the current one; write the behaviour charts (`behaviour_plots`) and return the count.

#### `Session.network_snapshot`

`def network_snapshot(self, player_id: int | None) -> dict | None`

What the selected tribute's network is doing, for the visualizer. Needs a live game and a selected id in it.

If the tribute's brain is a `NeatBrain` and it has a `last_perception`, the snapshot is in **graph mode**: `genome = brain.genome_data`, `inputs = last_perception.to_vector()`, `values = genome.activations(inputs)` (a value per node id), `depth = genome.depths()`, `logits = genome.forward(inputs)`. Probabilities come from `brain.probabilities_of(logits)` when the brain has that method, else `brain.last_probabilities`, else a uniform vector. The dictionary is:

| Key | Value |
| --- | --- |
| `graph` | `True` |
| `nodes` | `[(id, kind, depth, value), ...]` for every node in the genome |
| `edges` | `[(src, dst, weight), ...]` for enabled connections only |
| `probabilities` | the action probabilities |
| `chosen` | `brain.last_index` |
| `menu` | `MENU_NAMES` |

Otherwise the brain must be a `NeuralBrain` with a `last_perception`, and the snapshot is in **layer mode**: `layer_sizes`, `inputs`, `activations` (one array per layer from `network.hidden_activations`), `weights` (the matrices), `probabilities` (`brain.probabilities(logits)`), `chosen`, `menu`. Any other brain gives None.

#### `Session.seek`, `Session.rewind`, `Session.event_log`

Jump the playhead (clamped); back to frame 0; the last `last` (12) eliminations and gifts up to the playhead as text lines.

#### `Session.save_scenario` ... `Session.export_gif`

`save_scenario(path)`, `load_scenario(path)` (adopts the map, roster and loot; keeps `width`, `height` and `num_players` in step), `save_config(path)`, `load_config(path)`, `save_replay(path)`, `load_replay(path)` (through `load_recording`), `export_gif(path, step=2, fps=15)` (finishes the game first). Each sets a status line.

#### `Session.training_running`

`@property def training_running(self) -> bool`

Whether the training thread is alive.

#### `Session.LEARNER_KINDS`

```python
LEARNER_KINDS = ("neural", "neat")
```

The brain kinds a learner can have. Tributes with these brains get a star on the arena, and roster genomes of these kinds count for a warm start.

#### `Session.warm_start_genome`

`def warm_start_genome(self)`

A genome to start the next run from. First choice: the current trainer's `champion`, returned as the dictionary itself for NEAT or as a float array otherwise. Second choice: the first roster tribute whose `brain_name` is in `LEARNER_KINDS` and whose `genome` is not None (for example after Load champion into all), converted the same way. Else None.

#### `Session.start_training`

`def start_training(self, settings, method: str = "genetic", warm_start: bool = False, curriculum: CurriculumConfig | None = None) -> None`

Refuses while a run is alive. Then:

1. **Warm start.** `initial = warm_start_genome()` when asked. It is dropped (set back to None) when its kind does not fit the method: a dictionary is only accepted by `neat`, an array only by the other four; and `genetic` with `settings.brain_name` other than `neural` never warm-starts (a voting genome has 8 genes, a stored learner has thousands).
2. **Config and map.** A config copy whose `num_players` is the roster size (at least 2), and a `Scenario` holding only the painted terrain.
3. **Curriculum.** `Curriculum(curriculum)` when a config was given and `enabled` is true; else None.
4. **Trainer.** `training_method = method`, the feed counters and `_paused` reset, then `builders[method](config, settings, scenario=scenario, initial_genome=initial, curriculum=curriculum_object)` with `builders = {"imitation": ImitationTrainer, "genetic": GeneticTrainer, "neat": NeatTrainer, "reinforce": ReinforceTrainer, "ppo": PPOTrainer}`. An unknown method raises `KeyError` on the calling thread.
5. **Iterations.** `_max_iterations = int(getattr(settings, "epochs", getattr(settings, "generations", 0)))`: `epochs` for imitation, reinforce and PPO, `generations` for genetic and NEAT.
6. **The loop.** A `progress(done, total)` callback stores `training_progress`. The thread body sets `trainer._stop = False`, then `while len(trainer.learning_history) < _max_iterations and not trainer._stop:` sleep 0.1 s if `_paused`, else `trainer.step(progress)`. On exit the status is "Training stopped" (stopped) or "Training finished"; any exception becomes "Training error: ...".
7. **Status.** "Training (<method>[, warm start][, curriculum])...".

#### `Session.pause_training`, `Session.training_paused`

`def pause_training(self, paused: bool = True) -> None`, `@property def training_paused(self) -> bool`

Set or clear `_paused` (status "Training paused" or "Training resumed"); read it. Pausing takes effect between iterations: the current `step` runs to its end.

#### `Session.reset_training`

`def reset_training(self) -> None`

`stop_training()`, clear `_paused`, join the thread for up to 5 seconds, then forget the trainer, `training_progress`, `_feed_steps_seen` and `feed_label`. Status "Training reset". The roster keeps any champion genomes already handed out.

#### `Session.training_events`

`def training_events(self, count: int = 14) -> list[str]`

`trainer.events.tail(count)`: the newest timestamped lines such as `[   12.3s] rollout    iteration 4: mean score 2.10, length 310 ticks, win rate 0.17`, `record     new best mean score 2.10`, `curriculum promoted to stage 1: 3 opponents`, `evolution  generation 3: 5 species, best 4.20, mean 1.05`, `info       collected 40120 demonstrations from 12 teacher games`. Empty without a trainer.

#### `Session.learning_stats`

`def learning_stats(self) -> dict`

The learning statistics panel. With no history: `iteration 0`, `seed = config.seed`, and zeros for `seconds_per_iteration`, `max_score`, `learning_time`, `stage`, `opponents`, `mean_score`, `entropy`, `mean_length`. Otherwise from the last `IterationStats`: `iteration = last.iteration + 1`, `seed = trainer.settings.seed`, `seconds_per_iteration` = the mean of the last five `seconds`, `max_score` = the largest `best_score` so far, `learning_time = last.cumulative_seconds`, `stage`, `opponents`, `mean_score`, `entropy`, `mean_length`.

#### `Session.learner_ids_on_screen`

`def learner_ids_on_screen(self) -> set[int]`

Which tributes get a star. Watching a recording with no live game (a loaded replay or a replay-feed game): the ids in `recording.roster` whose `brain` is in `LEARNER_KINDS`. Otherwise: the roster tributes whose `brain_name` is in `LEARNER_KINDS` and whose `genome` is not None.

#### `Session.stop_training`

`def stop_training(self) -> None`

`trainer.stop()`: the loop ends after the current iteration.

#### `Session.training_history`

`def training_history(self) -> list`

A copy of `trainer.history`: the method-specific stats (`GenerationStats`, `EpochStats`, `ImitationStats`; for NEAT the same list as `learning_history`).

#### `Session.training_rows`

`def training_rows(self) -> list[dict]`

`[s.to_row() for s in trainer.learning_history]`: the unified rows every method shares (`iteration`, `mean_score`, `best_score`, `entropy`, `mean_length`, `win_rate`, `val_score`, `seconds`, `cumulative_seconds`, `stage`, `opponents`, and `extra_*` columns). The Train tab's plots read these.

#### `Session.latest_scores`

`def latest_scores(self) -> list[float]`

`learning_history[-1].scores`: one score per learner episode of the newest iteration (the "Latest scores" bars). For genetic and NEAT that is one fitness per genome in the population; for imitation the validation-game returns; for reinforce and PPO the returns of the collected episodes.

#### `Session.champion_genes`

`def champion_genes(self) -> tuple[np.ndarray, np.ndarray] | None`

None without a trainer or with an empty `trainer.history`. Otherwise the newest `learning_history` learner as a vector (NEAT: its connection weights) and a boolean mask of genes that moved by more than `1e-9` since the previous iteration. When there is no previous iteration, or its vector has a different length (NEAT after a structural mutation), every gene counts as changed.

#### `Session.save_training_run`

`def save_training_run(self, name: str, results_dir: str | Path = "results") -> Path | None`

`save_run(trainer, training_method, name, results_dir)`: writes `config.json`, `history.json`, `learning.json`, `events.txt`, `champion.json` and `plots/`. Status "Nothing trained yet" without history.

#### `Session.sweep_running`, `Session.start_sweep`, `Session.stop_sweep`

A `Sweep` on the painted map in a daemon thread with `sweep_progress` updates; stop after the current value.

#### `Session.give_champion`

`def give_champion(self, player_ids: list[int] | None = None) -> int`

Needs `trainer.champion`. The brain kind handed out is `neat` for the NEAT method, `trainer.training.brain_name` for genetic, else `neural`. Each target spec gets that `brain_name` and the champion as its `genome`: the dictionary itself for NEAT, else `np.asarray(champion).tolist()`. Then `config.neural = trainer.config.neural` so neural champions keep the architecture they were trained with. Returns the count.

#### `Session.save_champion`

`def save_champion(self, path: str | Path) -> None`

`trainer.save_champion(path)`. Every trainer writes the same file shape: `brain_name`, `genome`, `fitness`, plus method-specific fields (`neural` for neural champions; `method: "neat"` and `generations` for NEAT).

#### `Session.load_champion_into`

`def load_champion_into(self, path: str | Path, player_ids: list[int] | None = None) -> int`

Reads the file with `GeneticTrainer.load_champion` (which leaves a NEAT genome as a dictionary and turns anything else into an array). Sets `config.neural` when the file carries one. Each target spec gets the file's `brain_name` and the genome (the dictionary for NEAT, else a list). A file loaded this way is what `warm_start_genome` falls back to when there is no trainer.

## How to use it / experiment

**A whole game without a window.**

```python
from hunger_games.ui.session import Session
s = Session()
s.new_game(seed=1)
s.run_to_end()
print(s.status, s.recording.result.winner_name)
print(s.event_log(5))
```

**Imitation first, then PPO from it.** This is the order the Train tab recommends.

```python
from hunger_games.training import ImitationConfig, PPOConfig, CurriculumConfig
s = Session()
s.start_training(ImitationConfig(epochs=5, demonstration_games=4), "imitation")
s._training_thread.join()
print(s.training_rows()[-1]["extra_val_accuracy"])     # how often the student copies the teacher
print(s.warm_start_genome() is not None)               # True: the imitation champion
s.start_training(PPOConfig(epochs=3, episodes_per_epoch=2), "ppo",
                 warm_start=True, curriculum=CurriculumConfig(enabled=True))
print(s.status)                                        # Training (ppo, warm start, curriculum)...
s._training_thread.join()
print(s.learning_stats()["stage"], s.training_events(3))
s.save_champion("champion.json")                       # the same call for every trainer
```

**Pause and resume from a script.**

```python
from hunger_games.training import RLConfig
s.start_training(RLConfig(epochs=10, episodes_per_epoch=2), "reinforce")
s.pause_training(True)          # the current iteration finishes, then the loop sleeps
print(s.training_paused)        # True
s.pause_training(False)
s.stop_training()               # ends after the current iteration
s._training_thread.join()
s.reset_training()              # forget it
```

**NEAT and the graph snapshot.**

```python
from hunger_games.training import NeatTrainerConfig
s = Session()
s.start_training(NeatTrainerConfig(population_size=8, generations=2, validation_games=0), "neat")
s._training_thread.join()
print(isinstance(s.trainer.champion, dict))            # True: a NEAT genome dictionary
s.start_champion_game(all_slots=False)                 # the learner slots get brain_name "neat"
print(s.learner_ids_on_screen())                       # the starred tributes
s.step_once()
snap = s.network_snapshot(next(iter(s.learner_ids_on_screen())))
print(snap["graph"], len(snap["nodes"]), len(snap["edges"]))
print(len(s.champion_genes()[0]), "connection weights")
```

**Drive the training feed by hand.** The feed only moves inside `update`, so call it in a loop as the dashboard does.

```python
from hunger_games.training import TrainingConfig
s = Session()
s.feed_mode = "replay"
s.start_training(TrainingConfig(population_size=8, generations=3, rounds_per_generation=1,
                                validation_games=0), "genetic")
while s.training_running or s.recording is None:
    s.update(0.05)                       # shows the first showcase when a step exists
print(s.feed_label)                      # training feed: replaying a real generation 0 game
s.feed_mode = "live"
s.playhead = s.recording.length - 1      # finish watching so the arena is free
s.update(0.05)                           # the champion now plays live in the learner slots
print(s.feed_label, s.game is not None)
```

**Add a training method.** Add the trainer class to the `builders` dictionary in `start_training`, make sure it exposes the shared interface (see Concepts), and decide in `give_champion` which `brain_name` it hands out. The dashboard needs a matching entry in `Dashboard.METHODS`.

**Add a podium preset.** Add the name to `PODIUM_PRESETS` and an `elif preset == ...` branch that builds a list of `count` positions, snapping each with `arena.snap_to_podium`.

## Gotchas

- Trainers and sweeps get the painted map only. The hand-placed loot and the edited roster (names, granted items, podiums) are not used during training; `new_game` uses all of them.
- `start_training` replaces `self.trainer`, so a second run drops the earlier history and champion unless you saved them. A warm start copies the champion into the new trainer first, so the genome itself survives.
- `warm_start` defaults to `False` in `start_training`; the dashboard's checkbox defaults to on and passes its value every time.
- A warm start is dropped silently when the kinds do not match: a NEAT champion cannot seed a neural method and a neural champion cannot seed NEAT, and `genetic` with `brain_name="voting"` never warm-starts. The status line then lacks "warm start".
- `warm_start_genome` prefers `trainer.champion` over the roster. After Load champion into all, the loaded file is only used when no trainer exists yet (or the trainer has no champion); otherwise the previous run's champion wins. Reset training first to make the file win.
- The trainer is built on the calling thread, before the thread's `try` block. An error while building it (for example a warm-start genome whose length does not fit the current `config.neural` after the Brains tab was changed) is raised to the caller instead of becoming a status message.
- The loop counts `learning_history`, so a trainer whose `step` fails to append an iteration would loop forever; every shipped trainer appends one record per `step`.
- Pause is not instant. `trainer.step` is one whole iteration (all its games and the update); the pause flag is only read between iterations. Stop is the same.
- `reset_training` joins the thread for up to 5 seconds. A long iteration keeps the GUI waiting for that long, then the trainer is dropped while its thread may still be finishing the iteration in the background.
- `_advance_feed` labels steps `generation` only for the genetic method; NEAT steps are labelled `epoch` in the headline.
- The feed's `replay` mode goes through `load_recording`, which replaces the roster with the training game's roster, replaces `config` with the trainer's config copy (including its curriculum-sized `num_players`), and drops the live game. Save your scenario before turning it on.
- The feed's `live` mode changes the roster too: `give_champion` writes `brain_name` and `genome` into the learner slots and sets `config.neural`.
- The feed skips steps. If three iterations finish while one game is being watched, only the newest is shown next.
- `feed_label` is never cleared by the feed itself. Switching the feed back to `off` hides it from the headline, but the string remains until the next `start_training` or `reset_training`.
- Showcase games from the `replay` feed have no telemetry, so they never reach `watched_summaries` or the Charts tab.
- An imitation run with `validation_games=0` has no showcase and its scores, survival and win rate stay at 0.
- `training_progress` counts games within the current iteration, not iterations. For imitation the first epoch also counts the demonstration games.
- `champion_genes` and `genome_history` follow the learner after each iteration, while `trainer.champion` (used by `give_champion`, `start_champion_game` and `warm_start_genome`) is the best genome ever for genetic and NEAT, the best mean score for reinforce and PPO, and the lowest validation loss for imitation. The bar chart and the brain you hand out can differ.
- For NEAT, `genome_history` drops every earlier step whose genome has a different number of connections, so the Network tab's evolution plots restart after each structural change.
- `champion_genes` needs `trainer.history` to be non-empty but reads `learning_history`; for every shipped trainer both lists grow together.
- `learner_ids_on_screen` stars every `neural` or `neat` entry of a recording's roster, whether or not it was trained, because a `RosterEntry` has no genome field.
- `give_champion` with a curriculum-sized trainer config sets `config.neural` only; `num_players` stays the roster's. `start_champion_game(all_slots=False)` uses `trainer._learner_ids()`, which reads the trainer's `num_players`; with a curriculum the trainer's roster is smaller than yours, so the slots land in the low ids.
- `network_snapshot` reads the live `Game`, not the frame at the playhead, so it always shows the newest decision even while scrubbing backwards, and returns None for loaded replays, including replay-feed games.
- `load_recording` (so `load_replay` and the replay feed) replaces `config` with the recording's, and `load_config` replaces it too. Any code holding the old config object keeps the old one.
- The status string is written from the training and sweep threads. It is a plain assignment, which is safe in CPython, but the message can be overwritten by the next button press.
- Replays are pickles (`Recording.load`). Only open replay files you made yourself.
