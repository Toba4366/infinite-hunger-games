# `app.py`

**Source:** [hunger_games/ui/app.py](../../hunger_games/ui/app.py)
**Depends on:** `time`, `pathlib.Path`, `dearpygui.dearpygui as dpg`, `numpy`; project modules [../brain/init.md](../brain/init.md) (`BRAIN_REGISTRY`), [../brain/initializers.md](../brain/initializers.md) (`ACTIVATIONS`, `INITIALIZER_NOTES`, `INITIALIZERS`), [../brain/neural.md](../brain/neural.md) (`MENU_NAMES`, `MENU_SIZE`, `NeuralBrain`), [../brain/voting.md](../brain/voting.md) (`GENE_NAMES`), [../config.md](../config.md) (`ArenaShape`, `LayoutName`, `NeuralConfig`), [../districts.md](../districts.md) (`DISTRICT_INDUSTRIES`, `SEXES`), [../perception.md](../perception.md) (`VECTOR_NAMES`, `VECTOR_SIZE`), [../research/experiments.md](../research/experiments.md) (`SweepConfig`), [../research/telemetry.md](../research/telemetry.md) (`NEED_BIN_LABELS`), [../resources.md](../resources.md) (`ResourceKind`, `weapon_name`), [../terrain.md](../terrain.md) (`TerrainType`), [../training/init.md](../training/init.md) (`CurriculumConfig`, `ImitationConfig`, `NeatTrainerConfig`, `PPOConfig`, `RLConfig`, `TrainingConfig`), [canvas.md](canvas.md) (`ArenaCanvas`), [painter.md](painter.md) (`MapPainter.PRESETS`), [session.md](session.md) (`Session`), [visualizer.md](visualizer.md) (`NetworkVisualizer`)
**Used by:** [init.md](init.md) (`launch` builds a `Dashboard`), [screenshots.md](screenshots.md) (builds a `Dashboard` by hand and calls `_tutorial_action`)

## Purpose

`app.py` is the window. One primary window fills the viewport and holds three child panels that resize with it: control tabs on the left (Tutorial, Setup, Map, Loot, Tributes, Brains, Play, Train, Research), the arena with a transport bar in the centre, and Inspector, Network and Charts tabs on the right. Every widget callback changes the [`Session`](session.md) or a tool setting on the `Dashboard`; every frame `on_frame` advances playback and refreshes everything that changes. Nothing here simulates, paints or trains.

The Train tab is the largest part of the file. It is modelled on the training dashboard of the PPO zombie-arena video: a method picker with a one-line explanation, Start, Pause, Stop, Reset and Watch agent buttons, a bar plot of the newest iteration's scores, an event monitor, three line plots (average score, entropy, average game length), a learning-statistics block with a rollout bar and CPU and memory readings, the learner's genes as bars, the seconds per iteration, champion buttons, the run folder, and an Advanced settings section that shows only the current method's settings.

## Concepts you need

- **Context, viewport, frame loop.** `dpg.create_context()` initialises Dear PyGui, `dpg.create_viewport(...)` is the OS window, widgets are built before `dpg.setup_dearpygui()`, and `dpg.show_viewport()` displays it. The loop calls `dpg.render_dearpygui_frame()` until `dpg.is_dearpygui_running()` is false. It is written by hand so `on_frame()` runs before every render.
- **Primary window and child windows.** `dpg.set_primary_window("root", True)` makes one window fill the viewport. `dpg.child_window` is a resizable, scrollable box inside it; the three panels are child windows whose sizes `_layout` recomputes from `dpg.get_viewport_client_width()` and `..._height()` on every resize. The event monitor is a fixed-height child window too, so a long log scrolls inside it.
- **Tags.** Any widget can have `tag="..."`. `dpg.get_value`, `dpg.set_value` and `dpg.configure_item` address it later. Every widget this file updates has a tag; they are listed per tab below and in the tag index at the end. Tab bars and tabs are tagged too: `dpg.set_value("left_tabs", "tab_map")` switches the left panel to the Map tab. The tutorial and the screenshot tool use that.
- **Callbacks.** A callback is called as `callback(sender, app_data, user_data)`, but Dear PyGui passes only as many as the function accepts, so callbacks here take `()`, `(s, a)` or `(sender, value, player_id)`. `app_data` is the new value, or a dictionary for a file dialog.
- **Callback factories.** `_setter`, `setter`, `im`, `ga`, `ne`, `rl` and `rw` return callbacks bound to one attribute name; `lambda s, a, u=speed: ...` inside a loop freezes `speed` per button. `_tip` attaches a tooltip to the widget created just before it.
- **Settings objects on the Dashboard.** The Train tab edits five settings dataclasses held on the `Dashboard` (`imitation`, `ga`, `neat`, `rl`, `ppo`) plus a `curriculum`. Pressing Start copies the current method's dataclass and hands the copy to the session, so editing a field during a run does not change the run.
- **Collapsing headers, groups, tables, plots.** `dpg.collapsing_header` folds a section. `dpg.group(tag=...)` with `show=False` hides a block; `dpg.delete_item(tag, children_only=True)` empties a group so it can be refilled. A table has columns in slot 0 and rows in slot 1. A plot holds axes; line, bar and heat series are children of the Y axis; `dpg.set_value(series, [xs, ys])` replaces the data and `dpg.fit_axis_data(axis)` rescales.
- **Heat series.** `dpg.add_heat_series(values, rows, cols, ...)` draws a matrix as coloured cells; the plot's colormap (`dpg.bind_colormap`) maps `scale_min..scale_max` to colours. A heat series cannot change its row count after creation, so the evolution heat map is deleted and recreated each time a step is added.
- **Handler registry and threads.** `dpg.handler_registry()` catches mouse down (fires every frame while held), release and click events that are not tied to a widget. Training and sweeps run in session threads; this file only polls each frame and never blocks. The training loop itself lives in the session so that Pause can hold it between iterations.

## Walkthrough

### `TOOLS`

```python
TOOLS = ("Select", "Paint terrain", "Place loot", "Move tribute")
```

The mouse tools in the radio button at the top of the left panel.

### `SPEEDS`

```python
SPEEDS = {"Slow-mo 2/s": 2.0, "Normal 8/s": 8.0, "Fast 40/s": 40.0, "Max 400/s": 400.0}
```

Play tab speed buttons: label to ticks per second.

### `SWEEPABLE`

Config fields offered in the Research tab's parameter combo: `chaos`, `max_days`, `vision_radius`, `landmark_radius`, `thirst_days`, `hunger_days`, `sponsor_gift_chance`, `gamemaker_enabled`, `intervention_days`, `quiet_days_before_intervention`, `endgame_instinct`, `cannon_and_sky`, `start_thirst_min`, `start_hunger_min`, `start_health_min`, `num_players`, `terrain.water_threshold`, `terrain.sand_size`, `terrain.grass_size`, `noise.scale`, `noise.octaves`. Dotted names reach nested configs.

### `FONT_CANDIDATES`

Font files tried in order: Arial and Verdana from macOS's Supplemental folder, `/Library/Fonts/Arial.ttf`, Windows `segoeui.ttf`, Linux `DejaVuSans.ttf`. The first that exists is loaded at size 15; if none exists Dear PyGui's built-in font is used.

### `LEFT_FRACTION`, `RIGHT_FRACTION`

```python
LEFT_FRACTION, RIGHT_FRACTION = 0.27, 0.27
```

Left and right panel widths as fractions of the window; the centre takes the rest.

### `class Dashboard`

"Builds the window and wires every widget to the session."

#### `Dashboard.__init__`

`def __init__(self) -> None`

Creates the `Session`, the `ArenaCanvas(self.session)` and the `NetworkVisualizer()`, then the tool state: `tool = TOOLS[0]`, `brush_terrain = TerrainType.GRASS`, `brush_radius = 2`, `loot_kind = ResourceKind.WEAPON`, `loot_quantity = 1`, `loot_quality = 0.8`, `drag_id = None`, `painting = False`, `auto_next = False`, `_last_time = time.time()`, `_plotted_steps = -1` (how many training rows the plots have drawn), `_frame = 0`.

The training settings being edited are one dataclass per method, all at their defaults: `ga = TrainingConfig()`, `rl = RLConfig()`, `imitation = ImitationConfig()`, `neat = NeatTrainerConfig()`, `ppo = PPOConfig()`. The curriculum is `CurriculumConfig(enabled=False)`. `method = "imitation"` is the method the Train tab starts on. `_events_shown = 0` counts how many event lines the monitor has drawn. `brush_demo = None` is a brush ring the screenshot tool can park on the arena.

#### `Dashboard.run`

`def run(self) -> None`

Creates the context, loads the font, applies the theme, creates the viewport (title "Infinite Hunger Games - Game Makers' Dashboard", 1500 by 920, minimum 1100 by 700), calls `build()`, registers the three mouse handlers, sets the resize callback to `_layout`, then runs the manual frame loop: `on_frame()` then `dpg.render_dearpygui_frame()` until the window closes. Destroys the context at the end.

#### `Dashboard._load_font`

`def _load_font(self) -> None`

Binds the first existing `FONT_CANDIDATES` file at size 15.

#### `Dashboard._apply_theme`

`def _apply_theme(self) -> None`

A dark theme bound to everything: window background `(22, 24, 30)`, child background `(30, 33, 41)`, frames `(44, 48, 58)`, crimson buttons `(150, 28, 48)`, purple headers `(70, 52, 100)`, gold slider grabs, check marks and histogram bars `(242, 214, 72)`, text `(230, 230, 235)`, rounded corners (6 to 8 pixels) and padding.

#### `Dashboard._tip`

`def _tip(text: str) -> None` (static)

Attaches a tooltip (wrapped at 320 pixels) to `dpg.last_item()`.

#### `Dashboard.build`

`def build(self) -> None`

Builds the primary window `root`: a title row (`INFINITE HUNGER GAMES` in gold, the subtitle, and `status_text`), then three child windows side by side. `left_panel` holds the mouse tool radio button and the `left_tabs` tab bar with tabs `tab_tutorial`, `tab_setup`, `tab_map`, `tab_loot`, `tab_tributes`, `tab_brains`, `tab_play`, `tab_train`, `tab_research`. `center_panel` holds the canvas and the transport bar. `right_panel` holds `right_tabs` with `tab_inspector`, `tab_network`, `tab_charts`. Finally `dpg.set_primary_window("root", True)`.

#### `Dashboard._layout`

`def _layout(self) -> None`

Reads the viewport client size. Panel height is `max(400, height - 60)`. Left and right widths are 27 % of the window each; the centre is `max(300, width - left - right - 50)`. The canvas is resized to a square of `min(center - 24, panel_height - 190)` pixels; the visualizer to `right - 30` by `panel_height - 120`.

#### `Dashboard.on_frame`

`def on_frame(self) -> None`

Per frame: compute delta time (capped at 0.25 s), count the frame, call `session.update(seconds)`. If auto-next is on (`auto_next` and the `auto_next_box` checkbox) and the watched game is over, at the live edge and not playing, start a new game and play it. Set the canvas's brush preview to the cell under the mouse when the Paint terrain tool is active, else to `brush_demo`. Then `canvas.render()`, `_refresh_transport()`, `_refresh_inspector()`, `_refresh_network()`, `_refresh_training()`, `_refresh_research()`, and every 30th frame `_refresh_charts()`. Finally the status text.

#### `Dashboard.TUTORIAL_STEPS`

A list of `(title, text, action)` triples: "Welcome" (no action), "1. Build an arena" (`arena`), "2. Paint terrain" (`paint`), "3. Edit the tributes" (`tributes`), "4. Place loot" (`loot`), "5. Play a game" (`play`), "6. Inspect and watch a network think" (`network`), "7. Train and watch training" (`train`), "8. Research" (`research`), "9. Save and share" (`files`). Step 7's text describes the current Train tab: one network is trained and plays the starred tributes against voting opponents; start with `imitation`, then keep "start from the current champion" ticked and pick `ppo`, `reinforce`, `genetic` or `neat`; the curriculum grows the opposition 1, 3, 7, 11, 23; the feed replays a real training game or lets the newest learner play live; Show me starts a short imitation run with the live feed.

#### `Dashboard._build_tutorial`

`def _build_tutorial(self) -> None`

One collapsing header per step (the first two open), the text wrapped at 360, and a "Show me" button for every step with an action.

#### `Dashboard._tutorial_action`

`def _tutorial_action(self, name: str) -> None`

| Action | What it does |
| --- | --- |
| `arena` | `apply_preset("lake_island")`, `reposition_off_void()`, open Map |
| `paint` | Tool = Paint terrain, open Map |
| `tributes` | Select the first tribute, open Tributes |
| `loot` | Tool = Place loot, open Loot |
| `play` | Tool = Select, `new_game()`, speed 8, play, open Play |
| `network` | Default brain `neural` (config and `cfg_brain`), `_on_brain_all()`, `new_game()`, speed 4, play, select the first tribute, open Network on the right and Brains on the left |
| `train` | Set `train_method` to `imitation` and call `_on_method(None, "imitation")`; set the feed to `live` (session and `feed_mode`); reset `_plotted_steps` and `_events_shown`; call `session.start_training(ImitationConfig(demonstration_games=2, epochs=4, validation_games=1, learners_per_game=6), "imitation")` (no warm start, no curriculum); open Train |
| `research` | Open Research |
| `files` | Open Play |

The train step uses its own fixed `ImitationConfig` and leaves the Train tab's own fields alone. Because it sets the combo and calls `_on_method`, the method combo and the help text do change to `imitation`.

#### `Dashboard._setter`

`def _setter(self, name: str, convert=lambda v: v, react: str | None = None)`

Returns a callback that sets `session.config.<name> = convert(value)` and, when `react` is given, calls `session.apply_config_change(react)` and rebuilds the roster table.

#### `Dashboard._build_setup`

`def _build_setup(self) -> None`

Four collapsing headers and two button rows. Tags and defaults come from the session's config.

| Section | Widget | Tag | Callback |
| --- | --- | --- | --- |
| Arena | shape combo | `cfg_shape` | `_setter("shape", ArenaShape, "shape")` |
| | loot layout combo | `cfg_layout` | `_setter("layout", LayoutName, "layout")` |
| | size (cells), 30 to 300, on Enter | `cfg_size` | `_on_size` |
| | seed (-1 = random) | `cfg_seed` | `_setter("seed", ...)` (negative becomes `None`) |
| | chaos 0 to 1 | `cfg_chaos` | `_setter("chaos")` |
| | max days 1 to 60 | `cfg_days` | `_setter("max_days")` |
| | ticks per day 4 to 96 | `cfg_tpd` | `_setter("ticks_per_day")` |
| Tributes and starting bars | tributes 2 to 96, on Enter | `cfg_players` | `_setter("num_players", react="players")` |
| | min thirst, min hunger, min health 0.05 to 1 | `cfg_thirst`, `cfg_hunger`, `cfg_health` | `_setter(...)` |
| | Everyone starts full, Random above 0.5 | none | `_set_start_bars(1.0)`, `_set_start_bars(0.5)` |
| | vision radius 2 to 30, landmark radius 5 to 80 | `cfg_vision`, `cfg_landmark` | `_setter(...)` |
| | days to die of thirst 1 to 10, days to starve 2 to 30 | `cfg_thirst_days`, `cfg_hunger_days` | `_setter(...)` |
| What tributes know | cannon and nightly sky, endgame instinct | `cfg_cannon`, `cfg_endgame` | `_setter(...)` |
| Sponsors and game makers | sponsor gifts, gift chance / day | `cfg_sponsors`, `cfg_gift` | `_setter(...)` |
| | game maker circle, quiet days before it 0.25 to 5, days to close 1 to 20 | `cfg_gm`, `cfg_quiet`, `cfg_close` | `_setter(...)` |
| | podiums may stand in water | `cfg_water_podiums` | `_setter("allow_water_podiums")` |
| Buttons | Regenerate arena, New roster | none | `generate_arena()` + `reposition_off_void()`; `_on_generate_roster` |
| | Save config, Load config | none | `_file_dialog(self._save_config, ".json", "config.json")`, `_file_dialog(self._load_config, ".json")` |

#### `Dashboard._set_start_bars`

`def _set_start_bars(self, value: float) -> None`

Sets all three starting minimums in the config and the three sliders.

#### `Dashboard._on_size`

`def _on_size(self, sender, value) -> None`

Sets `config.width = config.height = int(value)` and calls `apply_config_change("size")`.

#### `Dashboard._on_generate_roster`

`def _on_generate_roster(self) -> None`

`session.generate_roster()` then `_rebuild_roster_table()`.

#### `Dashboard._build_map`

`def _build_map(self) -> None`

Brush header: a terrain radio button (`void`, `water`, `sand`, `grass`, `rock`, default `grass`), brush radius 0 to 20, and a hint. Presets and stamps header: the `map_preset` combo over `MapPainter.PRESETS` (default `perlin`), Load preset, Carve round, Fill grass, Fill water, `stamp_radius` (default 20, 2 to 100), Circle at centre, Square at centre, the show tribute labels checkbox (sets `canvas.show_labels`), and the `map_coverage` text. Scenario files header: Save scenario (`scenario.json`) and Load scenario.

#### `Dashboard._stamp_circle`, `Dashboard._stamp_square`

`def _stamp_circle(self) -> None`, `def _stamp_square(self) -> None`

Stamp the brush terrain at the centre with `stamp_radius`, then `painter.finish()` and `reposition_off_void()`.

#### `Dashboard._build_loot`

`def _build_loot(self) -> None`

Kind radio (`food`, `weapon`, `medicine`, default `weapon`), quantity 1 to 20, quality 0 to 1 with a tooltip naming the weapon tiers, the `loot_weapon_name` preview text, a hint, the "also scatter the layout's loot" checkbox (default on, sets `scenario.use_layout_loot`), Clear hand-placed loot, and the `loot_count` text.

#### `Dashboard._build_tributes`

`def _build_tributes(self) -> None`

The `podium_preset` combo (default `edge ring`) with Arrange podiums; the `roster_table` (columns name, district, sex, score, brain; height 200); Add tribute and Remove selected; and the `editor_header` collapsing header with one widget per `TributeSpec` field, each wired by the local `setter(name, convert)` factory: `ed_name`, `ed_district`, `ed_sex`, `ed_score` (1 to 12), `ed_survival` (0.05 to 0.95), `ed_brain` (the `BRAIN_REGISTRY` names: `voting`, `random`, `neural`), `ed_weapon`, `ed_food` (0 to 20), `ed_medicine` (0 to 5), `ed_favor`, `ed_thirst`, `ed_hunger`, `ed_health` (0 means use the config), and Forget trained genome (sets `genome = None`).

#### `Dashboard._rebuild_roster_table`

`def _rebuild_roster_table(self) -> None`

Deletes the table rows (slot 1) and adds one selectable row per tribute. The brain column shows the brain name plus ` *` when the tribute carries a genome.

#### `Dashboard._on_select_row`, `Dashboard._select`

`def _on_select_row(self, sender, value, player_id) -> None`, `def _select(self, player_id: int | None) -> None`

Set `session.selected_id`, rebuild the table, retitle the editor header and copy the spec's fields into the `ed_*` widgets.

#### `Dashboard._on_add_tribute`, `Dashboard._on_remove_tribute`

Add a tribute and select it; remove the selected tribute and clear the selection.

#### `Dashboard._build_brains`

`def _build_brains(self) -> None`

The `cfg_brain` default brain combo, Give this brain to every tribute (`_on_brain_all`), then the network editor: `nn_layer_count` (1 to 6), the `nn_widths_group` holding `nn_width_0`, `nn_width_1`, ... (1 to 512 each), `nn_activation`, `nn_init` (updates `nn_init_note`), `nn_scale`, `nn_sparsity` (0.01 to 1), Apply network settings (`_on_apply_neural`) and the `nn_summary` text. A closed header lists the inputs and outputs by name. `_on_apply_neural()` is called once at build time so the summary is filled.

#### `Dashboard._on_brain_all`

`def _on_brain_all(self) -> None`

Sets every roster spec to the config's brain name and drops its genome, then rebuilds the table.

#### `Dashboard._rebuild_width_fields`, `Dashboard._on_layer_count`, `Dashboard._read_widths`

Clear and refill `nn_widths_group` with one `nn_width_<i>` input per hidden layer; when the count changes the widths already typed are kept and new layers start at 16; `_read_widths` collects the fields until one is missing.

#### `Dashboard._on_apply_neural`

`def _on_apply_neural(self) -> None`

Builds a `NeuralConfig` from the widgets (an empty width list becomes `(16,)`), stores it in `session.config.neural`, and shows `"Network: " + NeuralBrain(...).describe()` in `nn_summary`.

#### `Dashboard._build_play`

`def _build_play(self) -> None`

A hint, the four `SPEEDS` buttons in two rows, the `auto_next_box` checkbox, Save replay (`game.replay`) and Load replay, `gif_step` (default 2, 1 to 12) and Export GIF of this game (`game.gif`).

#### `Dashboard._set_speed`

`def _set_speed(self, ticks_per_second: float) -> None`

Sets `session.ticks_per_second` and the `speed_slider`.

#### `Dashboard.METHODS`

```python
METHODS = ("imitation", "genetic", "neat", "reinforce", "ppo")
```

The methods offered in the combo, in the order a learner should try them.

#### `Dashboard.METHOD_HELP`

The one-line explanation shown under the combo for each method:

| Method | Help text |
| --- | --- |
| `imitation` | Copies the voting brain's decisions (supervised). Start here: it gives the network instincts. |
| `genetic` | Evolves the weights of a population of networks; each plays as the learner against voting opponents. |
| `neat` | Evolves weights and the shape of the network, in species (the Monopoly video's method). |
| `reinforce` | Policy gradient with a value baseline: every action is scored by the reward function. |
| `ppo` | Clipped policy gradient with several passes per batch (the zombie video's method). The most stable reward method. |

#### `Dashboard._build_train`

`def _build_train(self) -> None`

The training dashboard, top to bottom. Every widget the refresh code touches has a tag.

| Widget | Tag | Default | Notes |
| --- | --- | --- | --- |
| Intro text | none | | "One network is trained. It plays the starred tributes; every other tribute uses the voting brain." |
| method combo | `train_method` | `imitation` | Callback `_on_method` |
| help text | `method_help` | `METHOD_HELP["imitation"]` | Updated by `_on_method` |
| start from the current champion | `warm_start` | on | Read by `_on_start_training`; no callback |
| curriculum: opponents grow 1, 3, 7, 11, 23 | `curriculum_on` | off | Sets `curriculum.enabled` |
| Training feed radio | `feed_mode` | `off` | `off`, `replay`, `live` (`Session.FEED_MODES`); sets `session.feed_mode` |
| Start | `train_start` | | `_on_start_training`; disabled while a run is alive |
| Pause | `train_pause` | | `_on_pause_training`; relabelled Resume while paused |
| Stop | none | | `session.stop_training` |
| Reset | none | | `_on_reset_training` |
| Watch agent | none | | `_on_watch_champion` |
| progress bar | `train_progress` | 0 | Games done in the current iteration; overlay names the iteration |
| summary text | `train_summary` | empty | The last row's numbers |
| Latest scores (one bar per episode) plot | `score_plot`, axes `score_x` (no tick labels), `score_y`, series `score_bars` | | `latest_scores()` |
| Event monitor child window, 150 high | `event_monitor`, text `event_text` | empty | `training_events()` |
| Average score plot with legend | `perf_plot`, `perf_x`, `perf_y`; series `perf_train` (label "mean"), `perf_val` ("validation"), `perf_mean` ("best") | | `mean_score`, `val_score`, `best_score` per iteration |
| Entropy (lower = more confident) plot | `stab_plot`, `stab_x`, `stab_y`; series `stab_entropy` | | `entropy` in nats |
| Average game length (learner survival) plot | `len_plot`, `len_x`, `len_y`; series `len_series` | | `mean_length` in ticks |
| Learning statistics text | `learn_stats` | empty | Two lines, see `_refresh_training` |
| rollout bar | `rollout_bar` | 0, overlay "rollout" | The same fraction as `train_progress`, with a games count |
| system text | `system_stats` | empty | CPU, memory, GPU |
| Learner genes (gold = changed since last step) plot | `gene_plot`, `gene_x`, `gene_y`; series `gene_same` ("unchanged"), `gene_changed` ("changed") | | First 400 genes |
| Seconds per iteration plot | `time_plot`, `time_x`, `time_y`; series `time_bars` | | `seconds` per row |
| Champion to all, Champion to selected | none | | `give_champion()` + table rebuild; `_on_champion_selected` |
| run name, Save run folder | `run_name` | `run` | `session.save_training_run(name)` |
| Save champion, Load champion into all | none | | File dialogs (`champion.json`, `.json`) |
| Advanced settings header (closed) | none | | Filled by `_build_method_settings` |

The tooltip on the control row reads: Start begins a run with the settings below; Pause holds it between iterations; Stop ends it after the current one; Reset forgets it; Watch agent gives the champion to the starred tributes and plays a game live.

#### `Dashboard._build_method_settings`

`def _build_method_settings(self) -> None`

Four groups, one visible at a time, plus the reward and curriculum headers. The factories `im`, `ga`, `ne` bind a field of `self.imitation`, `self.ga`, `self.neat`; `rl` sets the same field on both `self.rl` and `self.ppo`; `rw` sets a field of `session.config.reward`.

`im_group` (shown when the method is `imitation`):

| Widget | Field | Default | Range |
| --- | --- | --- | --- |
| teacher brain combo | `teacher` | `voting` | only `voting` |
| demonstration games | `demonstration_games` | 12 | 1 to 200 |
| learn only from the top N placings (0 = all) | `winners_top` | 0 | 0 to 24 |
| epochs | `epochs` | 30 | 1 to 1000 |
| batch size | `batch_size` | 256 | 8 to 4096 |
| learning rate | `learning_rate` | 0.001 | any (`%.5f`) |
| validation games | `validation_games` | 1 | 0 to 20 |
| CPU workers | `workers` | 1 | 1 to 32 |

`ga_group` (`genetic`):

| Widget | Field | Default | Range |
| --- | --- | --- | --- |
| brain to evolve | `brain_name` | `neural` | `neural`, `voting` |
| opponents | `opponents` | `voting` | `voting` (each genome is the learner against the voting brain, scored by return) or `self` (the population plays itself, scored by placement) |
| population | `population_size` | 48 | 4 to 480 |
| generations | `generations` | 20 | 1 to 1000 |
| games per genome | `rounds_per_generation` | 2 | 1 to 10 |
| elite fraction | `elite_fraction` | 0.1 | 0 to 0.5 |
| mutation rate | `mutation_rate` | 0.1 | 0 to 1 |
| mutation scale | `mutation_scale` | 0.1 | 0.001 to 1 |
| crossover rate | `crossover_rate` | 0.5 | 0 to 1 |
| validation games | `validation_games` | 2 | 0 to 20 |
| CPU workers | `workers` | 1 | 1 to 32 |

`neat_group` (`neat`):

| Widget | Field | Default | Range |
| --- | --- | --- | --- |
| population | `population_size` | 48 | 4 to 480 |
| generations | `generations` | 30 | 1 to 1000 |
| target species | `target_species` | 8 | 1 to 40 |
| add node rate | `neat.add_node_rate` | 0.03 | 0 to 0.5 |
| add connection rate | `neat.add_connection_rate` | 0.08 | 0 to 0.5 |
| validation games | `validation_games` | 2 | 0 to 20 |
| CPU workers | `workers` | 1 | 1 to 32 |

`rl_group` (`reinforce` and `ppo`; the shared fields are written to both dataclasses):

| Widget | Field | Default | Range |
| --- | --- | --- | --- |
| epochs | `epochs` | 30 | 1 to 10000 |
| games per epoch | `episodes_per_epoch` | 4 | 1 to 64 |
| learner copies per game | `learners_per_game` | 6 | 1 to 24 |
| learning rate | `learning_rate` | 0.001 | any (`%.5f`) |
| entropy bonus | `entropy_bonus` | 0.01 | 0 to 0.2 |
| PPO clip ratio | `ppo.clip_ratio` only | 0.2 | 0.05 to 0.5 |
| PPO passes per batch | `ppo.update_epochs` only | 4 | 1 to 20 |
| validation games | `validation_games` | 2 | 0 to 20 |
| CPU workers | `workers` | 1 | 1 to 32 |

Inside `rl_group`, the closed **Reward function** header edits `session.config.reward` directly:

| Slider | Field | Default | Range |
| --- | --- | --- | --- |
| per tick alive | `survive_tick` | 0.01 | 0 to 0.1 |
| win | `win` | 5.0 | 0 to 20 |
| death | `death` | -3.0 | -20 to 0 |
| kill | `kill` | 1.0 | 0 to 10 |
| per health lost | `damage_taken` | -2.0 | -10 to 0 |
| per need restored | `need_gain` | 0.5 | 0 to 5 |
| approach water/food (dense, off by default) | `approach` | 0.0 | 0 to 0.5 |
| placement | `placement` | 2.0 | 0 to 10 |
| discount | `discount` | 0.98 | 0.8 to 1 |

Outside the groups, the closed **Curriculum settings** header edits `self.curriculum`:

| Widget | Field | Default |
| --- | --- | --- |
| opponents per stage (comma-separated integers) | `opponents` | `1,3,7,11,23`; a text with no integers becomes `(23,)` |
| promotion threshold (mean score) | `threshold` | 3.0, range -5 to 10 |
| max iterations per stage | `max_iterations_per_stage` | 40, range 1 to 1000 |

The curriculum's `window` (5 iterations averaged for the promotion test) has no widget.

#### `Dashboard._on_method`

`def _on_method(self, sender, value) -> None`

Stores `self.method`, writes `METHOD_HELP[value]` into `method_help`, and shows exactly one group: `im_group` for `imitation`, `ga_group` for `genetic`, `neat_group` for `neat`, `rl_group` for `reinforce` and `ppo`.

#### `Dashboard._current_settings`

`def _current_settings(self)`

A fresh copy of the current method's dataclass: `ImitationConfig(**vars(self.imitation))`, `TrainingConfig(**vars(self.ga))`, `NeatTrainerConfig(**vars(self.neat))`, `RLConfig(**vars(self.rl))`, or `PPOConfig(**vars(self.ppo))` for anything else.

#### `Dashboard._on_start_training`

`def _on_start_training(self) -> None`

Resets `_plotted_steps` to -1 and `_events_shown` to 0 so the plots and the monitor redraw from scratch, then calls `session.start_training(self._current_settings(), self.method, bool(dpg.get_value("warm_start")), CurriculumConfig(**vars(self.curriculum)))`. The curriculum copy carries `enabled` from the checkbox; the session only builds a `Curriculum` when it is enabled.

#### `Dashboard._on_pause_training`

`def _on_pause_training(self) -> None`

`session.pause_training(not session.training_paused)`: a toggle.

#### `Dashboard._on_reset_training`

`def _on_reset_training(self) -> None`

`session.reset_training()`, then resets the two counters and empties the series `perf_train`, `perf_val`, `perf_mean`, `stab_entropy`, `len_series`, `time_bars`, `gene_same`, `gene_changed`, `score_bars` and the texts `event_text`, `train_summary`, `learn_stats`.

#### `Dashboard._on_champion_selected`

`def _on_champion_selected(self) -> None`

`session.give_champion([selected_id])` and a table rebuild, when something is selected.

#### `Dashboard._on_watch_champion`

`def _on_watch_champion(self) -> None`

Calls `session.start_champion_game(all_slots=False)`; if that returns False (no champion) nothing else happens. Otherwise rebuilds the roster table, sets the speed to 8 ticks per second and opens the Network tab on the right. Only the trainer's learner slots receive the champion, so those tributes carry stars.

#### `Dashboard._refresh_training`

`def _refresh_training(self) -> None`

Called every frame.

1. Progress: `done, total = session.training_progress`. `train_progress` and `rollout_bar` both show `done / total`. The rollout bar's overlay is `rollout {done}/{total} games ({percent}%)` while `total` is non-zero, else `rollout`. `steps = len(session.training_history())`. The progress bar's overlay is `iteration {steps}` (plus ` (paused)` when paused) while a run is alive, else `{steps} iterations done` or empty. `train_start` is disabled while running; `train_pause` is labelled `Resume` when paused, else `Pause`.
2. Event monitor: when the trainer's event count differs from `_events_shown`, write `session.training_events()` (the last 14 lines) into `event_text`.
3. Statistics: `stats = session.learning_stats()` and `system = session.system.read()`. `system_stats` reads `CPU {cpu}%   memory {MB} MB ({percent}% of RAM)   GPU: {gpu}`. `learn_stats` is two lines: `iteration N   seed S   X s/iteration   max score M   learning time T s` and `stage K (O opponents)   mean score ..   entropy ..   mean length .. ticks`.
4. Plots, only when `len(session.training_rows())` changed since the last draw: x is each row's `iteration`; `perf_train` = `mean_score`, `perf_val` = `val_score`, `perf_mean` = `best_score`, `stab_entropy` = `entropy`, `len_series` = `mean_length`, `time_bars` = `seconds`; `score_bars` = `session.latest_scores()` indexed 0, 1, 2, ...; genes from `session.champion_genes()`, the first 400 values split into `gene_same` and `gene_changed` by the changed mask, with `GENE_NAMES` as tick labels when the genome has exactly eight genes (the voting brain). Then `_refresh_evolution()` and `fit_axis_data` on the twelve axes.
5. Summary: from the last row, `"{method}: {n} iteration(s), mean score .., validation .., win rate ... "` followed by up to four `extra_*` values (floats to three decimals), for example `train_loss`, `val_loss`, `train_accuracy`, `val_accuracy` for imitation, `policy_loss`, `value_loss` for reinforce and PPO, `worst_fitness` for genetic, `species`, `hidden_nodes`, `connections`, `threshold` for NEAT.

#### `Dashboard._build_research`

`def _build_research(self) -> None`

Parameter sweep: `sweep_param` (default `chaos`), `sweep_values` (default `0,0.25,0.5,0.75,1`), `sweep_games` (20), `sweep_workers` (1), `sweep_telemetry` (on), Start sweep (`_on_start_sweep`), Stop (`session.stop_sweep`), `sweep_progress`, `sweep_results`. Exports: `export_folder` (default `output/watched`), Export behaviour charts, Forget watched games. Then the fixed reviewer text under "Answers a reviewer will ask for":

> Method: imitation (behaviour cloning of the voting brain), genetic algorithm (neuroevolution), NEAT (neuroevolution of topologies), REINFORCE with a value baseline, or PPO (clipped policy gradient), chosen on the Train tab, with warm starts between them and an optional opponent curriculum; experiments/run_comparison.py trains them all under one budget and runs a 75-game tournament. Rewards: the Reward function section there (the dense approach reward is off by default). Observation: a 50-value vector (Brains tab lists it), not a grid. Dashboard: custom, Dear PyGui; charts by matplotlib.

#### `Dashboard._on_start_sweep`

`def _on_start_sweep(self) -> None`

Parses the comma-separated values (`true`/`false` to booleans, then int, then float, else text), builds a `SweepConfig` whose `name` is the parameter with dots replaced by underscores, and calls `session.start_sweep`.

#### `Dashboard._refresh_research`

`def _refresh_research(self) -> None`

Progress bar and overlay (`{done}/{total} values` while running), and one results line per finished value: value, victor rate, mean days, pvp share, natural share.

#### `Dashboard._build_transport`

`def _build_transport(self) -> None`

New game, `play_button` (width 70), Step, To end, Rewind; `speed_slider` (0.5 to 400); `playhead` (frame scrub); the gold `headline` text.

#### `Dashboard._ensure_game`, `_on_new_game`, `_on_play`, `_on_step`, `_on_to_end`

Start a game if none is loaded; new game; toggle `session.playing`; stop and `step_once()`; `run_to_end()`.

#### `Dashboard._refresh_transport`

`def _refresh_transport(self) -> None`

Updates `loot_count`, `loot_weapon_name` and `map_coverage`, the play button label, then the headline: "No game yet. Press New game or Play." without a recording; otherwise the playhead range and `Day D   tick T   alive A/N   frame F/L`, plus `VICTOR: name` or `no victor (draw)` on the last frame of a finished game, and the feed label in front while the feed is on.

#### `Dashboard._build_inspector`, `Dashboard._refresh_inspector`

`insp_title`, `insp_facts`, three progress bars `insp_thirst`, `insp_hunger`, `insp_health`, `insp_more`, and the `insp_log` event log. The refresh shows the roster facts (with ` (trained)` after the brain name when the spec has a genome), the granted items and podium while editing, or the live bars, outcome, weapon and reach, food, medkits, kills, favour and last action during a game.

#### `Dashboard._build_network`

`def _build_network(self) -> None`

An explanation, the `network_caption` text, the closed `evolution_header` holding `evo_plot` (series `evo_change` and `evo_mean`) and `evo_heat_plot` (RdBu colormap, series recreated as `evo_heat_series`), then the `network_holder` child window into which `visualizer.build` draws.

#### `Dashboard._refresh_evolution`

`def _refresh_evolution(self) -> None`

Reads `session.network_evolution()`; returns when it is None. Sets the two line series, deletes and recreates the heat series with a symmetric scale, and fits the axes.

#### `Dashboard._refresh_network`

`def _refresh_network(self) -> None`

`snapshot = session.network_snapshot(selected_id)`; `architecture = [VECTOR_SIZE, *hidden_layers, MENU_SIZE]`; `visualizer.render(snapshot, architecture)`. The caption is `Live: {name}. Chosen: {action}. Red = positive activation, blue = negative.` with a snapshot (a NEAT graph snapshot included), else `Architecture: 50 -> ... -> 16 (activation, initializer).`

#### `Dashboard._build_charts`, `Dashboard._refresh_charts`

`chart_actions` (bar series `chart_actions_bars`), `chart_instinct` (lines `chart_drink`, `chart_eat`, `chart_flee`), and the 30 by 30 `chart_heat` heat series (`chart_heat_series`, Viridis). The refresh merges `session.watched_summary()` every 30 frames.

#### `Dashboard._on_mouse_down`, `_on_mouse_release`, `_on_mouse_click`

Held left button paints (Paint terrain tool, no game) or drags a tribute (Move tribute tool, no game). Release ends the stroke (`finish_painting`, `reposition_off_void`) and the drag. A click selects with the Select tool or any right click outside the loot tool; with the Place loot tool a left click places and a right click removes.

#### `Dashboard._file_dialog`

`def _file_dialog(self, callback, extension: str, default_filename: str | None = None) -> None`

A modal 760 by 460 file dialog. The chosen path gets the extension appended if missing, then `callback(path)`; any exception becomes `session.status = "Error: ..."`.

#### `Dashboard._save_config` ... `Dashboard._load_champion`

`_save_config`, `_load_config` (loads, then writes every `cfg_*` widget, the hidden-layer fields, applies the neural settings, regenerates the map and rebuilds the table), `_save_scenario`, `_load_scenario` (rebuilds the table and sets `cfg_size`), `_save_replay`, `_load_replay` (rebuilds the table), `_export_gif` (uses `gif_step`), `_save_champion`, `_load_champion` (`session.load_champion_into(path)` for every tribute, then a table rebuild).

### `launch`

`def launch() -> None`

`Dashboard().run()`. Blocks until the window is closed.

### Tag index

| Area | Tags |
| --- | --- |
| Window | `root`, `status_text`, `left_panel`, `center_panel`, `right_panel`, `left_tabs`, `right_tabs`, `tab_*` |
| Setup | `cfg_shape`, `cfg_layout`, `cfg_size`, `cfg_seed`, `cfg_chaos`, `cfg_days`, `cfg_tpd`, `cfg_players`, `cfg_thirst`, `cfg_hunger`, `cfg_health`, `cfg_vision`, `cfg_landmark`, `cfg_thirst_days`, `cfg_hunger_days`, `cfg_cannon`, `cfg_endgame`, `cfg_sponsors`, `cfg_gift`, `cfg_gm`, `cfg_quiet`, `cfg_close`, `cfg_water_podiums` |
| Map | `map_preset`, `stamp_radius`, `map_coverage` |
| Loot | `loot_weapon_name`, `loot_count` |
| Tributes | `podium_preset`, `roster_table`, `editor_header`, `ed_name`, `ed_district`, `ed_sex`, `ed_score`, `ed_survival`, `ed_brain`, `ed_weapon`, `ed_food`, `ed_medicine`, `ed_favor`, `ed_thirst`, `ed_hunger`, `ed_health` |
| Brains | `cfg_brain`, `nn_layer_count`, `nn_widths_group`, `nn_width_<i>`, `nn_activation`, `nn_init`, `nn_init_note`, `nn_scale`, `nn_sparsity`, `nn_summary` |
| Play | `auto_next_box`, `gif_step` |
| Train | `train_method`, `method_help`, `warm_start`, `curriculum_on`, `feed_mode`, `train_start`, `train_pause`, `train_progress`, `train_summary`, `score_plot`, `score_x`, `score_y`, `score_bars`, `event_monitor`, `event_text`, `perf_plot`, `perf_x`, `perf_y`, `perf_train`, `perf_val`, `perf_mean`, `stab_plot`, `stab_x`, `stab_y`, `stab_entropy`, `len_plot`, `len_x`, `len_y`, `len_series`, `learn_stats`, `rollout_bar`, `system_stats`, `gene_plot`, `gene_x`, `gene_y`, `gene_same`, `gene_changed`, `time_plot`, `time_x`, `time_y`, `time_bars`, `run_name`, `im_group`, `ga_group`, `neat_group`, `rl_group` |
| Research | `sweep_param`, `sweep_values`, `sweep_games`, `sweep_workers`, `sweep_telemetry`, `sweep_progress`, `sweep_results`, `export_folder` |
| Transport | `play_button`, `speed_slider`, `playhead`, `headline` |
| Inspector | `insp_title`, `insp_facts`, `insp_thirst`, `insp_hunger`, `insp_health`, `insp_more`, `insp_log` |
| Network | `network_caption`, `evolution_header`, `evo_plot`, `evo_x`, `evo_y`, `evo_change`, `evo_mean`, `evo_heat_plot`, `evo_heat_x`, `evo_heat_y`, `evo_heat_series`, `network_holder`, `network_canvas` (in visualizer.py) |
| Charts | `chart_actions`, `chart_actions_x`, `chart_actions_y`, `chart_actions_bars`, `chart_instinct`, `chart_instinct_x`, `chart_instinct_y`, `chart_drink`, `chart_eat`, `chart_flee`, `chart_heat`, `chart_heat_x`, `chart_heat_y`, `chart_heat_series` |

The individual settings inside `im_group`, `ga_group`, `neat_group` and `rl_group`, the reward sliders and the curriculum fields have no tags.

## How to use it / experiment

- **Drive the Train tab from a script.** Build `d = Dashboard()`, call `dpg.create_context()`, `d.build()`, then `dpg.set_value("train_method", "ppo"); d._on_method(None, "ppo")` and `d._on_start_training()`. The screenshot tool ([screenshots.md](screenshots.md)) does this kind of thing with `_tutorial_action`.
- **Add a method.** Append its name to `METHODS`, a sentence to `METHOD_HELP`, a settings dataclass on the `Dashboard`, a group in `_build_method_settings`, a line in `_on_method`'s table and a branch in `_current_settings`. The session needs a matching entry in its `builders` dictionary.
- **Add a plot.** Create a `dpg.plot` with tagged axes and series in `_build_train`, set its data in step 4 of `_refresh_training` from a `training_rows()` column, add the axes to the `fit_axis_data` list, and add the series to `_on_reset_training`'s clear list.
- **Change what the summary shows.** The summary prints the first four `extra_*` keys of the last row. Reorder the `extra` dictionary in the trainer to change which four appear.
- **Watch the event monitor.** It only redraws when the trainer's event count changes, so `self.session.trainer.events.add("info", "hello")` from a thread shows up on the next frame.
- **Add a config slider.** Inside `_build_setup`, add a widget with `callback=self._setter("<field>")` and `tag="cfg_<field>"`, and add the tag to `_load_config`'s list so loading a file refreshes it.
- **Add a tutorial step.** Append `("10. My step", "What to do.", "mystep")` to `TUTORIAL_STEPS` and an `elif name == "mystep":` branch to `_tutorial_action` that ends with `dpg.set_value("left_tabs", "tab_...")`.

## Gotchas

- The dashboard starts on `imitation` and with "start from the current champion" ticked. A second Start of any method continues from the first run's champion unless you untick the box.
- The `perf_*` series tags are historical: `perf_train` draws the mean score, `perf_val` the validation score and `perf_mean` the best score. Read the legend, not the tag.
- Reset clears the Train tab's series and texts but not the Network tab's evolution plots. After a reset `network_evolution()` returns None, so `_refresh_evolution` leaves the old lines and heat map in place until the next run adds a row.
- `train_progress`'s overlay counts `session.training_history()`, which is the trainer's method-specific history (`GenerationStats`, `EpochStats`, `ImitationStats`, or the unified list for NEAT). The plots use `training_rows()`, the unified history. They have the same length for every method.
- The reward sliders write straight into `session.config.reward`. They take effect for the next Start, are saved by Save config, are not refreshed by Load config, and are not reset by Reset.
- The shared `rl_group` fields write to both `self.rl` and `self.ppo`. The two PPO-only widgets write only to `self.ppo`, so a `reinforce` run never sees them.
- The Tributes tab's brain combo lists `BRAIN_REGISTRY` (`voting`, `random`, `neural`). `neat` is not in it; a tribute becomes `neat` only by Champion to all, Champion to selected, Watch agent, the live feed or Load champion into all with a NEAT file. Picking a brain in the combo for that tribute turns it back into a registry brain.
- The event monitor shows the last 14 events (`training_events(count=14)`); the run folder's `events.txt` has all of them (up to 500).
- The tutorial's train step starts a run even if the Train tab's settings say otherwise, but it does nothing while a run is already alive (the session refuses a second one).
- `_refresh_training` runs every frame and reads `SystemMonitor.read()` every frame. Without psutil the CPU and memory readings are zeros and the GPU text is "not used (numpy on the CPU)".
- Painting, loot and dragging are ignored while a game is loaded (`session.game is not None`). Load replay, or the `replay` feed, clears the live game; Watch agent and the `live` feed start one.
- `cfg_size` and `cfg_players` apply on Enter, not on every keystroke. The "nodes in hidden layer N" fields have no callbacks; nothing changes until Apply network settings.
- `_tutorial_action("network")` calls `_on_brain_all`, which drops every trained genome in the roster. After that a warm start has nothing to pick up from the roster (a trainer's champion still counts).
- Training plots, including the evolution plots on the Network tab, are updated only when the row count changes, so a slow iteration shows a frozen plot with a moving rollout bar. They fill in even while the Network tab is not open.
- `dpg.set_axis_ticks("gene_x", ...)` is only set for the voting brain's 8 genes; after a voting run followed by a neural run the names stay on the axis until the dashboard restarts.
- The GIF export and To end run on the GUI thread and block the window until they finish.
