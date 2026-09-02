# `app.py`

**Source:** [hunger_games/ui/app.py](../../hunger_games/ui/app.py)
**Depends on:** `time`, `pathlib.Path`, `dearpygui.dearpygui as dpg`, `numpy`; project modules [../brain/init.md](../brain/init.md) (`BRAIN_REGISTRY`), [../brain/initializers.md](../brain/initializers.md) (`ACTIVATIONS`, `INITIALIZER_NOTES`, `INITIALIZERS`), [../brain/neural.md](../brain/neural.md) (`MENU_NAMES`, `MENU_SIZE`, `NeuralBrain`), [../brain/voting.md](../brain/voting.md) (`GENE_NAMES`), [../config.md](../config.md) (`ArenaShape`, `LayoutName`, `NeuralConfig`), [../districts.md](../districts.md) (`DISTRICT_INDUSTRIES`, `SEXES`), [../perception.md](../perception.md) (`VECTOR_NAMES`, `VECTOR_SIZE`), [../research/experiments.md](../research/experiments.md) (`SweepConfig`), [../research/telemetry.md](../research/telemetry.md) (`NEED_BIN_LABELS`), [../resources.md](../resources.md) (`ResourceKind`, `weapon_name`), [../terrain.md](../terrain.md) (`TerrainType`), [../training/init.md](../training/init.md) (`RLConfig`, `TrainingConfig`), [canvas.md](canvas.md) (`ArenaCanvas`), [painter.md](painter.md) (`MapPainter.PRESETS`), [session.md](session.md) (`Session`), [visualizer.md](visualizer.md) (`NetworkVisualizer`)
**Used by:** [init.md](init.md) (`launch` builds a `Dashboard`), [screenshots.md](screenshots.md) (builds a `Dashboard` by hand and calls `_tutorial_action`)

## Purpose

`app.py` is the window. One primary window fills the viewport and holds three child panels that resize with it: control tabs on the left (Tutorial, Setup, Map, Loot, Tributes, Brains, Play, Train, Research), the arena with a transport bar in the centre, and Inspector, Network and Charts tabs on the right. Every widget callback changes the [`Session`](session.md) or a tool setting on the `Dashboard`; every frame `on_frame` advances playback and refreshes everything that changes. Nothing here simulates, paints or trains.

## Concepts you need

- **Context, viewport, frame loop.** `dpg.create_context()` initialises Dear PyGui, `dpg.create_viewport(...)` is the OS window, widgets are built before `dpg.setup_dearpygui()`, and `dpg.show_viewport()` displays it. The loop calls `dpg.render_dearpygui_frame()` until `dpg.is_dearpygui_running()` is false. It is written by hand so `on_frame()` runs before every render.
- **Primary window and child windows.** `dpg.set_primary_window("root", True)` makes one window fill the viewport. `dpg.child_window` is a resizable, scrollable box inside it; the three panels are child windows whose sizes `_layout` recomputes from `dpg.get_viewport_client_width()` and `..._height()` on every resize.
- **Tags.** Any widget can have `tag="..."`. `dpg.get_value`, `dpg.set_value` and `dpg.configure_item` address it later. Every widget this file updates has a tag; they are listed per tab below. Tab bars and tabs are tagged too: `dpg.set_value("left_tabs", "tab_map")` switches the left panel to the Map tab. The tutorial and the screenshot tool use that.
- **Callbacks.** A callback is called as `callback(sender, app_data, user_data)`, but Dear PyGui passes only as many as the function accepts, so callbacks here take `()`, `(s, a)` or `(sender, value, player_id)`. `app_data` is the new value, or a dictionary for a file dialog.
- **Callback factories.** `_setter`, `setter`, `ga`, `rl` and `rw` return callbacks bound to one attribute name; `lambda s, a, u=speed: ...` inside a loop freezes `speed` per button. `_tip` attaches a tooltip to the widget created just before it.
- **Collapsing headers, groups, tables, plots.** `dpg.collapsing_header` folds a section. `dpg.group(tag=...)` with `show=False` hides a block; `dpg.delete_item(tag, children_only=True)` empties a group so it can be refilled. A table has columns in slot 0 and rows in slot 1. A plot holds axes; line, bar and heat series are children of the Y axis; `dpg.set_value(series, [xs, ys])` replaces the data and `dpg.fit_axis_data(axis)` rescales.
- **Heat series.** `dpg.add_heat_series(values, rows, cols, ...)` draws a matrix as coloured cells; the plot's colormap (`dpg.bind_colormap`) maps `scale_min..scale_max` to colours. A heat series cannot change its row count after creation, so the evolution heat map is deleted and recreated each time a step is added.
- **Handler registry and threads.** `dpg.handler_registry()` catches mouse down (fires every frame while held), release and click events that are not tied to a widget. Training and sweeps run in session threads; this file only polls each frame and never blocks.

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

Creates `session = Session()`, `canvas = ArenaCanvas(session)`, `visualizer = NetworkVisualizer()`. Tool state: `tool = "Select"`, `brush_terrain = GRASS`, `brush_radius = 2`, `loot_kind = WEAPON`, `loot_quantity = 1`, `loot_quality = 0.8`, `drag_id = None`, `painting = False`, `auto_next = False`. Bookkeeping: `_last_time`, `_plotted_steps = -1`, `_frame = 0`. Training settings being edited: `ga = TrainingConfig()`, `rl = RLConfig()`, `method = "genetic"`.

#### `Dashboard.run`

`def run(self) -> None`

Creates the context, loads the font and theme, creates the viewport (title "Infinite Hunger Games - Game Makers' Dashboard", 1500 by 920, minimum 1100 by 700), calls `build()`, registers the three mouse handlers, sets `_layout` as the viewport resize callback, sets up and shows the viewport, runs `_layout()` once, then loops `on_frame()` and `render_dearpygui_frame()` until the window closes. Destroys the context at the end. [screenshots.md](screenshots.md) repeats these steps by hand, without the mouse handlers, so it can render frames one at a time.

#### `Dashboard._load_font`, `Dashboard._apply_theme`

`_load_font` binds the first existing `FONT_CANDIDATES` file at size 15. `_apply_theme` binds a dark theme to everything: near-black backgrounds, crimson buttons and active tabs `(150, 28, 48)`, purple headers, gold slider grabs, check marks and progress bars `(242, 214, 72)`, light text, rounded frames, and padding of 8 by 5.

#### `Dashboard._tip`

`@staticmethod def _tip(text: str) -> None`

Attaches a tooltip (text wrapped at 320 pixels) to the most recently created widget. Used after most controls.

#### `Dashboard.build`

`def build(self) -> None`

Lays out the primary window `root` (no title bar, not resizable or movable, no scrollbar): a title row ("INFINITE HUNGER GAMES", "game makers' dashboard", and `status_text`), then a horizontal group of three child windows.

| Panel | Tag | Contents |
| --- | --- | --- |
| Left | `left_panel` | "Mouse tool" radio button over `TOOLS`, a separator, then the tab bar `left_tabs` |
| Centre | `center_panel` | `canvas.build("center_panel")` then `_build_transport()` |
| Right | `right_panel` | The tab bar `right_tabs` |

The tabs are tagged so code can switch to them with `dpg.set_value(<tab bar>, <tab tag>)`:

| Tab bar | Tab | Tag | Built by |
| --- | --- | --- | --- |
| `left_tabs` | Tutorial | `tab_tutorial` | `_build_tutorial` |
| | Setup | `tab_setup` | `_build_setup` |
| | Map | `tab_map` | `_build_map` |
| | Loot | `tab_loot` | `_build_loot` |
| | Tributes | `tab_tributes` | `_build_tributes` |
| | Brains | `tab_brains` | `_build_brains` |
| | Play | `tab_play` | `_build_play` |
| | Train | `tab_train` | `_build_train` |
| | Research | `tab_research` | `_build_research` |
| `right_tabs` | Inspector | `tab_inspector` | `_build_inspector` |
| | Network | `tab_network` | `_build_network` |
| | Charts | `tab_charts` | `_build_charts` |

Ends with `dpg.set_primary_window("root", True)`.

#### `Dashboard._layout`

`def _layout(self) -> None`

Reads the viewport client size. `panel_height = max(400, height - 60)`; `left = width * 0.27`, `right = width * 0.27`, `center = max(300, width - left - right - 50)`. Applies them to the three panels, then `canvas.resize(min(center - 24, panel_height - 190))` and `visualizer.resize(right - 30, panel_height - 120)`.

#### `Dashboard.on_frame`

`def on_frame(self) -> None`

Every frame: delta time capped at 0.25 s; `session.update(seconds)` (which first advances the training feed, then playback); auto-next (if `auto_next`, the game is over, the playhead is at the live edge, playback stopped and `auto_next_box` is ticked, start a new game and play); brush preview (`canvas.brush_preview = (x, y, brush_radius)` when the Paint terrain tool hovers a cell, else `None`); `canvas.render()`; then `_refresh_transport`, `_refresh_inspector`, `_refresh_network`, `_refresh_training`, `_refresh_research`; `_refresh_charts` every 30th frame; finally `status_text`.

#### `Dashboard.TUTORIAL_STEPS`

A class attribute: a list of `(title, text, action)` tuples, one per tutorial step. `action` is a short name handed to `_tutorial_action`, or `None` for a step with no button.

| Title | Action | What the text says |
| --- | --- | --- |
| Welcome | `None` | The three panels, tooltips, how the "Show me" buttons work, and that the written version is `docs/tutorial/README.md` |
| 1. Build an arena | `"arena"` | Setup picks shape, layout and size; Map loads presets and paints. Show me loads `lake_island` |
| 2. Paint terrain | `"paint"` | Pick Paint terrain, choose terrain and radius, drag; tributes are moved off void |
| 3. Edit the tributes | `"tributes"` | The roster table and editor; podium presets or the Move tribute tool |
| 4. Place loot | `"loot"` | Kind, quantity, quality; left-click places, right-click removes; marker shapes |
| 5. Play a game | `"play"` | New game records every tick; the transport bar; circles are female, squares male |
| 6. Inspect and watch a network think | `"network"` | The Inspector, and the Network tab for neural tributes. Show me gives everyone a neural brain and starts a game |
| 7. Train and watch training | `"train"` | Genetic or reinforce; the training feed's `replay` and `live` modes. Show me starts a short evolution of voting brains |
| 8. Research | `"research"` | Sweeps and behaviour chart exports; run folders under `results/` |
| 9. Save and share | `"files"` | Where each file is saved; the docs folder |

#### `Dashboard._build_tutorial`

`def _build_tutorial(self) -> None`

One collapsing header per step, labelled with the title; the first two (Welcome and step 1) start open. Each holds the text (wrapped at 360) and, when the step has an action, a "Show me" button whose callback is `lambda s, a, u=action: self._tutorial_action(u)`. The default argument freezes the action name per button.

#### `Dashboard._tutorial_action`

`def _tutorial_action(self, name: str) -> None`

Performs one step with the same session and dashboard methods the real controls use, then switches to the tab where the step lives.

| `name` | What it does | Tabs shown |
| --- | --- | --- |
| `"arena"` | `session.apply_preset("lake_island")`, `session.reposition_off_void()` | `tab_map` |
| `"paint"` | `tool = "Paint terrain"` | `tab_map` |
| `"tributes"` | `_select(first tribute's id)` if the roster is not empty | `tab_tributes` |
| `"loot"` | `tool = "Place loot"` | `tab_loot` |
| `"play"` | `tool = "Select"`, `session.new_game()`, `_set_speed(8.0)`, `session.playing = True` | `tab_play` |
| `"network"` | `config.brain_name = "neural"`, `cfg_brain` set to `neural`, `_on_brain_all()` (every tribute becomes neural and drops any genome), `session.new_game()`, `_set_speed(4.0)`, play, select the first tribute | `tab_network` on the right, `tab_brains` on the left |
| `"train"` | `self.ga = TrainingConfig(brain_name="voting", population_size=24, generations=5, rounds_per_generation=1, validation_games=1)`, `session.feed_mode = "live"`, `feed_mode` radio set to `live`, `_on_start_training()` | `tab_train` |
| `"research"` | Nothing else | `tab_research` |
| `"files"` | Nothing else | `tab_play` |

Any other name does nothing. Note that `"train"` starts whichever method the Train tab's radio button currently shows: `_on_start_training` reads `self.method`, so with `reinforce` selected the new `self.ga` is ignored and a reinforce run starts from `self.rl`.

#### `Dashboard._setter`

`def _setter(self, name: str, convert=lambda v: v, react: str | None = None)`

Returns a callback `(sender, value)` that sets `session.config.<name> = convert(value)` and, if `react` is given, calls `session.apply_config_change(react)` and rebuilds the roster table.

#### `Dashboard._build_setup`

`def _build_setup(self) -> None`

Four collapsing headers plus buttons. Every control writes the config through `_setter`.

| Header | Control | Tag | Widget, range, default | React |
| --- | --- | --- | --- | --- |
| Arena (open) | shape | `cfg_shape` | combo `open_field`, `round` | `shape` |
| | loot layout | `cfg_layout` | combo `cornucopia`, `ring` | `layout` |
| | size (cells) | `cfg_size` | int 30..300, Enter to apply, default 120 | `_on_size` |
| | seed (-1 = random) | `cfg_seed` | int; below 0 becomes `None` | |
| | chaos | `cfg_chaos` | float 0..1, default 0.5 | |
| | max days | `cfg_days` | int 1..60, default 24 | |
| | ticks per day | `cfg_tpd` | int 4..96, default 24 | |
| Tributes and starting bars (open) | tributes | `cfg_players` | int 2..96, Enter to apply, default 24 | `players` |
| | min thirst, min hunger, min health | `cfg_thirst`, `cfg_hunger`, `cfg_health` | float 0.05..1, default 1.0 | |
| | Everyone starts full, Random above 0.5 | | buttons, `_set_start_bars(1.0 / 0.5)` | |
| | vision radius | `cfg_vision` | int 2..30, default 8 | |
| | landmark radius | `cfg_landmark` | int 5..80, default 30 | |
| | days to die of thirst | `cfg_thirst_days` | float 1..10, default 3 | |
| | days to starve | `cfg_hunger_days` | float 2..30, default 7 | |
| What tributes know (closed) | cannon and nightly sky | `cfg_cannon` | checkbox, default on | |
| | endgame instinct | `cfg_endgame` | checkbox, default off | |
| Sponsors and game makers (closed) | sponsor gifts | `cfg_sponsors` | checkbox, default on | |
| | gift chance / day | `cfg_gift` | float 0..1, default 0.5 | |
| | game maker circle | `cfg_gm` | checkbox, default on | |
| | quiet days before it | `cfg_quiet` | float 0.25..5, default 1 | |
| | days to close | `cfg_close` | float 1..20, default 6 | |
| | podiums may stand in water | `cfg_water_podiums` | checkbox, default on | |
| Buttons | Regenerate arena | | `generate_arena()` then `reposition_off_void()` | |
| | New roster | | `_on_generate_roster` | |
| | Save config, Load config | | file dialogs, `.json` | |

#### `Dashboard._set_start_bars`, `_on_size`, `_on_generate_roster`

`_set_start_bars(value)` sets all three starting minimums in the config and the three sliders. `_on_size(sender, value)` sets `config.width = config.height = value` (the arena is always square from here) and calls `apply_config_change("size")`. `_on_generate_roster()` rolls a roster and rebuilds the table.

#### `Dashboard._build_map`

`def _build_map(self) -> None`

| Header | Control | Tag | Effect |
| --- | --- | --- | --- |
| Brush | terrain radio `void water sand grass rock` | | `brush_terrain`, default grass |
| | brush radius | | int 0..20, `brush_radius`, default 2 |
| Presets and stamps | preset | `map_preset` | combo over `MapPainter.PRESETS`, default `perlin` |
| | Load preset | | `apply_preset` then `reposition_off_void` |
| | Carve round | | `painter.carve_round()`, `finish()`, `reposition_off_void()` |
| | Fill grass, Fill water | | `painter.fill(...)` then `finish()` |
| | stamp radius | `stamp_radius` | int 2..100, default 20 |
| | Circle at centre, Square at centre | | `_stamp_circle`, `_stamp_square` |
| | show tribute labels | | `canvas.show_labels`, default on |
| | coverage line | `map_coverage` | filled each frame |
| Scenario files | Save scenario, Load scenario | | file dialogs, `.json` |

#### `Dashboard._stamp_circle`, `Dashboard._stamp_square`

`_stamp_circle()` calls `painter.stamp_circle(centre, stamp_radius, brush_terrain)`; `_stamp_square()` calls `painter.stamp_rectangle` from centre minus `r` to centre plus `r`. Both then `finish()` and `reposition_off_void()`.

#### `Dashboard._build_loot`

`def _build_loot(self) -> None`

| Control | Tag | Effect |
| --- | --- | --- |
| kind radio `food weapon medicine` | | `loot_kind`, default weapon |
| quantity | | int 1..20, `loot_quantity` |
| quality | | float 0..1, `loot_quality`, default 0.8 |
| weapon name preview | `loot_weapon_name` | `weapon_name(loot_quality)`, each frame |
| also scatter the layout's loot | | `scenario.use_layout_loot`, default on |
| Clear hand-placed loot | | `session.clear_loot` |
| stack count | `loot_count` | each frame |

#### `Dashboard._build_tributes`

`def _build_tributes(self) -> None`

Podium presets: combo `podium_preset` over `Session.PODIUM_PRESETS` (default `edge ring`, width 160) and an "Arrange podiums" button. Then the table `roster_table` (columns name, district, sex, score, brain; height 200; scrolls) filled by `_rebuild_roster_table`. Then "Add tribute" and "Remove selected". Then the collapsing header `editor_header` "Selected tribute" whose local `setter(name, convert)` writes one field of the selected spec and rebuilds the table:

| Control | Tag | Field |
| --- | --- | --- |
| name | `ed_name` | `name` |
| district (combo "1 Luxury" ... "12 Mining") | `ed_district` | `district` (first word as int) |
| sex | `ed_sex` | `sex` |
| training score 1..12 | `ed_score` | `training_score` |
| survival score 0.05..0.95 | `ed_survival` | `survival_score` |
| brain (combo over `BRAIN_REGISTRY`) | `ed_brain` | `brain_name` |
| granted weapon 0..1 | `ed_weapon` | `weapon_quality` |
| granted food 0..20 | `ed_food` | `food` |
| granted medkits 0..5 | `ed_medicine` | `medicine` |
| sponsor favour bonus 0..1 | `ed_favor` | `favor_bonus` |
| start thirst / hunger / health (0 = config) | `ed_thirst`, `ed_hunger`, `ed_health` | `start_*`, 0 becomes `None` |
| Forget trained genome | | `genome = None` |

#### `Dashboard._rebuild_roster_table`

`def _rebuild_roster_table(self) -> None`

Deletes the rows (slot 1) and adds one `table_row` per tribute: a selectable spanning the columns (callback `_on_select_row`, `user_data=player_id`, highlighted when selected), district, sex, score, and the brain name with ` *` when a genome is stored. Does nothing if `roster_table` does not exist yet.

#### `Dashboard._on_select_row`, `Dashboard._select`

`_on_select_row(sender, value, player_id)` is the row callback and just calls `_select(player_id)`. `_select(player_id: int | None)` sets `session.selected_id`, rebuilds the table, relabels `editor_header` ("Selected tribute: <name>" or ": none"), and copies the spec into every `ed_*` widget.

#### `Dashboard._on_add_tribute`, `Dashboard._on_remove_tribute`

Add: `session.add_tribute()` then select it. Remove: `session.remove_tribute(selected_id)` then `_select(None)`.

#### `Dashboard._build_brains`

`def _build_brains(self) -> None`

| Control | Tag | Effect |
| --- | --- | --- |
| default brain (combo over `BRAIN_REGISTRY`: voting, random, neural) | `cfg_brain` | `config.brain_name` |
| Give this brain to every tribute | | `_on_brain_all` |
| shape line | | text "50 inputs (the perception) -> hidden layers -> 16 outputs (the action menu)" |
| number of hidden layers | `nn_layer_count` | int 1..6, clamped; default `len(config.neural.hidden_layers)`, which is 1; callback `_on_layer_count` |
| nodes in hidden layer N (one field per layer) | `nn_width_0`, `nn_width_1`, ... inside the group `nn_widths_group` | int 1..512, clamped; defaults from `config.neural.hidden_layers`, so one field of 16 |
| activation | `nn_activation` | combo over `ACTIVATIONS`, default `tanh` |
| initializer | `nn_init` | combo over `INITIALIZERS`, default `xavier_uniform`; updates the note |
| initializer note | `nn_init_note` | `INITIALIZER_NOTES[name]` |
| init scale | `nn_scale` | float, default 0.05 |
| sparsity | `nn_sparsity` | float 0.01..1, default 0.1 |
| Apply network settings | | `_on_apply_neural` |
| summary | `nn_summary` | the network's `describe()` |
| Inputs (50) and outputs (16) header | | lists `VECTOR_NAMES` and `MENU_NAMES` with their indices |

The width fields are created by `_rebuild_width_fields(list(n.hidden_layers))` right after the empty `nn_widths_group` is made. The width fields have no callbacks: nothing reaches the config until "Apply network settings" is pressed. `_on_apply_neural()` is called once at the end of the build so the summary is filled in.

The 50 inputs, in order: thirst, hunger, health, survival score, training score, weapon quality, reach, food carried, medkits carried, in water, hunt difficulty, downhill dx/dy, water dx/dy/distance, grass dx/dy/distance, centre dx/dy/distance, loot here kind/qty/quality, nearby loot dx/dy/distance/kind, threat dx/dy/distance/level/health, players in sight, in danger zone, hazard distance, hazard closing, safe dx/dy, day fraction, alive fraction, field known, field strength, strongest remaining, my rank, on water/sand/grass/rock. The 16 outputs: rest, drink, eat, hunt, pick_up, heal, attack, flee, and eight moves (up-left, up, up-right, left, right, down-left, down, down-right).

#### `Dashboard._on_brain_all`

`def _on_brain_all(self) -> None`

Sets every spec's `brain_name` to `config.brain_name` and `genome` to `None`, then rebuilds the table.

#### `Dashboard._rebuild_width_fields`

`def _rebuild_width_fields(self, widths: list[int]) -> None`

Empties `nn_widths_group` (`dpg.delete_item(..., children_only=True)`) and adds one `dpg.add_input_int` per entry of `widths`, labelled "nodes in hidden layer 1", "nodes in hidden layer 2" and so on, tagged `nn_width_0`, `nn_width_1`, ..., each clamped to 1..512 and parented to the group. Called at build time, when the layer count changes and after Load config.

#### `Dashboard._on_layer_count`

`def _on_layer_count(self, sender, count) -> None`

The callback of `nn_layer_count`. Reads the widths typed so far with `_read_widths()`, then builds the new list as `(current + [16] * count)[:count]`: existing widths are kept in order, extra layers start at 16 nodes, and surplus layers are dropped from the end. Rebuilds the fields with `_rebuild_width_fields`.

#### `Dashboard._read_widths`

`def _read_widths(self) -> list[int]`

Collects `int(dpg.get_value(f"nn_width_{index}"))` for `index = 0, 1, 2, ...` until a tag does not exist. Returns the list, so `[32, 16]` for two fields holding 32 and 16.

#### `Dashboard._on_apply_neural`

`def _on_apply_neural(self) -> None`

Reads the widths into a tuple and replaces `config.neural` with `NeuralConfig(hidden_layers=layers or (16,), activation=nn_activation, initializer=nn_init, init_scale=float(nn_scale), sparsity=float(nn_sparsity))`. Then builds a sample `NeuralBrain(config=..., rng=default_rng(0))` and writes "Network: " plus its `describe()` into `nn_summary`, for example `Network: 50 -> 16 -> 16, tanh, xavier_uniform, 1088 params`. Also called at build time and after loading a config.

#### `Dashboard._build_play`

`def _build_play(self) -> None`

A hint text, four speed buttons from `SPEEDS` (two rows), the checkbox `auto_next_box` "start a new game when this one ends (back to back)", Save replay / Load replay (`.replay`), the slider `gif_step` "GIF ticks per frame" (1..12, default 2) and "Export GIF of this game" (`.gif`).

#### `Dashboard._set_speed`

`def _set_speed(self, ticks_per_second: float) -> None`

Sets `session.ticks_per_second` and the `speed_slider`.

#### `Dashboard._build_train`

`def _build_train(self) -> None`

A radio button `genetic` / `reinforce` (`_on_method`), then two groups of which one is shown.

| Group | Control | Field | Range, default |
| --- | --- | --- | --- |
| `ga_group` | brain to evolve | `ga.brain_name` | `neural`, `voting`; default neural |
| | population | `ga.population_size` | 4..480, default 48 |
| | generations | `ga.generations` | 1..1000, default 20 |
| | games per genome | `ga.rounds_per_generation` | 1..10, default 2 |
| | elite fraction | `ga.elite_fraction` | 0..0.5, default 0.1 |
| | mutation rate | `ga.mutation_rate` | 0..1, default 0.1 |
| | mutation scale | `ga.mutation_scale` | 0.001..1, default 0.1 |
| | crossover rate | `ga.crossover_rate` | 0..1, default 0.5 |
| | validation games | `ga.validation_games` | 0..20, default 2 |
| | CPU workers | `ga.workers` | 1..32, default 1 |
| `rl_group` (hidden at start) | epochs | `rl.epochs` | 1..10000, default 30 |
| | games per epoch | `rl.episodes_per_epoch` | 1..64, default 4 |
| | learners per game | `rl.learners_per_game` | 1..24, default 6 |
| | learning rate | `rl.learning_rate` | float, default 0.001 |
| | value learning rate | `rl.value_learning_rate` | float, default 0.003 |
| | entropy bonus | `rl.entropy_bonus` | 0..0.2, default 0.01 |
| | validation games | `rl.validation_games` | 0..20, default 2 |
| | CPU workers | `rl.workers` | 1..32, default 1 |
| Reward function header (inside `rl_group`) | per tick alive | `config.reward.survive_tick` | 0..0.1, default 0.01 |
| | win | `reward.win` | 0..20, default 5 |
| | death | `reward.death` | -20..0, default -3 |
| | kill | `reward.kill` | 0..10, default 1 |
| | per health lost | `reward.damage_taken` | -10..0, default -2 |
| | per need restored | `reward.need_gain` | 0..5, default 0.5 |
| | placement | `reward.placement` | 0..10, default 2 |
| | discount | `reward.discount` | 0.8..1, default 0.98 |

Below the groups: "Start training" (`train_start`) and "Stop after this step"; progress bar `train_progress`; summary `train_summary`; the training feed; four plots; champion buttons "Champion to all", "Champion to selected", "Watch champion"; the run name field `run_name` (default `run`) with "Save run folder"; and "Save champion" / "Load champion into all" (`.json`).

The training feed is a horizontal group: the text "Training feed" and a radio button tagged `feed_mode` over `Session.FEED_MODES` (`off`, `replay`, `live`), default `session.feed_mode` (which starts as `off`). Its callback sets `session.feed_mode`. The tooltip explains: `replay` replays one real evaluation game from every step (the population playing itself); `live` gives the newest champion to the learner slots and plays a fresh game live so the Network tab shows real activations; the next step is shown when the current game ends. The feed itself is driven by `Session._advance_feed` (see [session.md](session.md)).

| Plot | Tag | Axes | Series |
| --- | --- | --- | --- |
| Performance | `perf_plot` | `perf_x`, `perf_y` | lines `perf_train`, `perf_val`, `perf_mean` |
| Stability | `stab_plot` | `stab_x`, `stab_y` | lines `stab_ploss`, `stab_vloss`, `stab_entropy` |
| Time per step (s) | `time_plot` | `time_x`, `time_y` | bars `time_bars` |
| Champion genes | `gene_plot` | `gene_x`, `gene_y` | bars `gene_same`, `gene_changed` |

#### `Dashboard._on_method`, `_on_start_training`, `_on_champion_selected`, `_on_watch_champion`

| Method | What it does |
| --- | --- |
| `_on_method(sender, value)` | Stores `method` and shows `ga_group` or `rl_group` |
| `_on_start_training()` | Resets `_plotted_steps`, then `session.start_training(TrainingConfig(**vars(self.ga)), "genetic")` or `session.start_training(RLConfig(**vars(self.rl)), "reinforce")` |
| `_on_champion_selected()` | `session.give_champion([selected_id])` and rebuild the table |
| `_on_watch_champion()` | `give_champion()` to everyone (status "No champion yet: train first" if it returns 0), rebuild the table, `new_game()`, speed 8, play |

#### `Dashboard._refresh_training`

`def _refresh_training(self) -> None`

Each frame: the progress bar shows games done in the current step with overlay "step N: done/total games" while running or "N steps done" after; `train_start` is disabled while running. The plots are updated only when `training_rows()` grew.

| Plot | Genetic | Reinforce |
| --- | --- | --- |
| Performance | best fitness, validation fitness, population mean by generation | training return, validation return, win rate by epoch |
| Stability | action entropy from each generation's telemetry (losses empty) | policy loss, value loss, policy entropy |
| Time per step | seconds per generation | seconds per epoch |
| Champion genes | latest champion's values, gold where changed since the previous step; first 400 genes; gene names on the axis for the voting brain | same, from the latest epoch's policy |

After the timing bars it calls `_refresh_evolution()` so the Network tab's evolution plots grow with the same step. The summary line reads, for genetic, "N generation(s), Ts total. Best fitness x (gen g), validation y." and for reinforce "N epoch(s), Ts total. Train return x, validation y, survival t ticks, win rate w, entropy e."

#### `Dashboard._build_research`

`def _build_research(self) -> None`

| Section | Control | Tag | Default |
| --- | --- | --- | --- |
| Parameter sweep | parameter | `sweep_param` | combo over `SWEEPABLE`, `chaos` |
| | values | `sweep_values` | text `0,0.25,0.5,0.75,1` |
| | games per value | `sweep_games` | int, min 1, 20 |
| | CPU workers | `sweep_workers` | int 1..32, 1 |
| | collect behaviour telemetry | `sweep_telemetry` | on |
| | Start sweep, Stop | | `_on_start_sweep`, `session.stop_sweep` |
| | progress, results | `sweep_progress`, `sweep_results` | |
| Charts of the games you have watched | folder | `export_folder` | `output/watched` |
| | Export behaviour charts | | `session.export_behaviour_plots(folder)` |
| | Forget watched games | | clears `watched_summaries` |
| Answers a reviewer will ask for | text | | method, rewards, observation, tooling |

#### `Dashboard._on_start_sweep`

`def _on_start_sweep(self) -> None`

Parses the values (`true`/`false` to bool, then int, then float, else text) and starts `SweepConfig(name=parameter with dots replaced, parameter, values, games_per_value, workers, telemetry)`.

#### `Dashboard._refresh_research`

`def _refresh_research(self) -> None`

Progress bar with "done/total values" while running; one results line per finished value: victors %, mean days, pvp %, natural %.

#### `Dashboard._build_transport`

`def _build_transport(self) -> None`

Under the arena: buttons New game, Play (`play_button`, width 70), Step, To end, Rewind; slider `speed_slider` "ticks / second" 0.5..400; slider `playhead` "frame"; the gold `headline` text.

#### `Dashboard._ensure_game`, `_on_new_game`, `_on_play`, `_on_step`, `_on_to_end`

`_ensure_game()` calls `session.new_game()` if no recording is loaded. New game: `session.new_game()`. Play: `_ensure_game()` then toggle `playing`, so Play starts a game if there is none. Step: `_ensure_game()`, stop, `step_once()`. To end: `_ensure_game()`, `run_to_end()`.

#### `Dashboard._refresh_transport`

`def _refresh_transport(self) -> None`

Fills `loot_count`, `loot_weapon_name` and `map_coverage`; labels the play button "Pause" or "Play"; without a recording the headline says "No game yet. Press New game or Play."; otherwise sets the scrub slider's range and value and writes "Day d   tick t   alive a/n   frame i/N", adding "VICTOR: name" or "no victor (draw)" on the last frame of a finished game. When `session.feed_mode` is not `off` and `session.feed_label` is not empty, the label is put in front: "training feed: replaying a real generation 3 game   |   Day d ...".

#### `Dashboard._build_inspector`, `Dashboard._refresh_inspector`

`_build_inspector()` creates `insp_title`, `insp_facts`, three progress bars `insp_thirst`, `insp_hunger`, `insp_health`, `insp_more`, then "Event log" and `insp_log`. `_refresh_inspector()` runs every frame and always writes the event log. With a selection: name, district and industry, sex, scores, brain (" (trained)" with a genome). Before a game: bars show the starting values (the spec's or the config's minimums) and `insp_more` lists granted items and the podium. During a game: bars from the frame's snapshot, then alive or "eliminated day d (weapon by killer), placed p", weapon name and reach (1 below 0.6, 2 below 0.9, else 3), food, medkits, kills, favour, last action.

#### `Dashboard._build_network`

`def _build_network(self) -> None`

A hint, the caption `network_caption`, the evolution section, and the child window `network_holder` holding `visualizer.build(...)`.

The evolution section is the collapsing header `evolution_header` "How the champion network changed over training", closed by default, with two plots:

| Plot | Tag | Label | Axes | Series |
| --- | --- | --- | --- | --- |
| Change per step | `evo_plot` (height 150) | "Genome change per step (L2) and mean \|weight\|" | `evo_x` "step", `evo_y` "value" | lines `evo_change` "change from previous step", `evo_mean` "mean \|weight\|" |
| Genome heat map | `evo_heat_plot` (height 170) | "Champion genome by step (rows = steps, columns = first 200 genes)" | `evo_heat_x` "gene", `evo_heat_y` "step" | `evo_heat_series`, created by `_refresh_evolution`; colormap `mvPlotColormap_RdBu` |

#### `Dashboard._refresh_evolution`

`def _refresh_evolution(self) -> None`

Called from `_refresh_training` whenever a new step exists. Takes `session.network_evolution()` and returns if it is `None`. Sets `evo_change` to `(steps, change)` and `evo_mean` to `(steps, mean_abs)` and fits both axes. Then deletes `evo_heat_series` if it exists and adds a new `dpg.add_heat_series` under `evo_heat_y` from `data["genes"]` (steps by up to 200 genes, flattened row by row), with `scale_min = -limit` and `scale_max = limit` where `limit` is the largest absolute gene value (1.0 if every gene is zero), no cell labels (`format=""`), and bounds from `(0, 0)` to `(gene columns, steps)`. Fits the heat axes. Red and blue therefore mean positive and negative weights of equal size on either side of zero.

#### `Dashboard._refresh_network`

`def _refresh_network(self) -> None`

`visualizer.render(session.network_snapshot(selected_id), [50, *hidden_layers, 16])`. Caption: "Live: name. Chosen: action. Red = positive activation, blue = negative." or "Architecture: 50 -> 16 -> 16 (tanh, xavier_uniform)."

#### `Dashboard._build_charts`

`def _build_charts(self) -> None`

Three plots: "Action distribution (%)" (`chart_actions`, axes `chart_actions_x/_y`, bars `chart_actions_bars`); "Instinct curves (%)" (`chart_instinct`, axes `chart_instinct_x/_y`, lines `chart_drink`, `chart_eat`, `chart_flee`); "Where tributes spend time" (`chart_heat`, equal aspects, axes `chart_heat_x/_y`, a 30 by 30 heat series `chart_heat_series` with the Viridis colormap).

#### `Dashboard._refresh_charts`

`def _refresh_charts(self) -> None`

From `session.watched_summary()`: action shares in percent with action names as ticks; the share of decisions that were drink, eat and flee in each of the five need bins (`NEED_BIN_LABELS` as ticks); the position heatmap flipped so row 0 is at the top and scaled to its maximum.

#### `Dashboard._on_mouse_down`, `_on_mouse_release`, `_on_mouse_click`

All three take `(sender, button)` and use `canvas.mouse_cell()`.

| Handler | Behaviour |
| --- | --- |
| `_on_mouse_down` | Left button only, over the arena, with no game loaded: Paint terrain paints a dab and sets `painting`; Move tribute picks `drag_id = tribute_at(cell)` on the first frame and moves that podium every frame while held |
| `_on_mouse_release` | Ends a stroke (`finish_painting()`, `reposition_off_void()`) and drops the dragged tribute |
| `_on_mouse_click` | Over the arena: the Select tool, or a right click in any tool except Place loot, selects `tribute_at(cell)`. Place loot with no game loaded: left places, right removes |

#### `Dashboard._file_dialog`

`def _file_dialog(self, callback, extension: str, default_filename: str | None = None) -> None`

Opens a modal `dpg.file_dialog` (760 by 460) filtering on `extension` and `.*`. On choice it appends the extension if missing and calls `callback(path)`; any exception goes to the status bar as "Error: ...".

#### File callbacks: `_save_config`, `_load_config`, `_save_scenario`, `_load_scenario`, `_save_replay`, `_load_replay`, `_export_gif`, `_save_champion`, `_load_champion`

Each takes `(self, path: str)` and is handed to `_file_dialog`.

| Method | What it does |
| --- | --- |
| `_save_config` | `session.save_config(path)` |
| `_load_config` | `session.load_config(path)`, then every `cfg_*` widget is refreshed from the config; `nn_layer_count` is set to the number of hidden layers and `_rebuild_width_fields` rebuilds the width fields; `nn_activation`, `nn_init`, `nn_scale` and `nn_sparsity` are refreshed; `_on_apply_neural()` runs; `apply_config_change("size")` regenerates the map; the table is rebuilt |
| `_save_scenario` | `session.save_scenario(path)` |
| `_load_scenario` | `session.load_scenario(path)`, rebuild the table, set `cfg_size` |
| `_save_replay` | `session.save_replay(path)` |
| `_load_replay` | `session.load_replay(path)` and rebuild the table |
| `_export_gif` | `session.export_gif(path, step=gif_step)` |
| `_save_champion` | `session.save_champion(path)` |
| `_load_champion` | `session.load_champion_into(path)` for everyone, then rebuild the table |

### `launch`

`def launch() -> None`

`Dashboard().run()`. The public entry point is the same-named function in [init.md](init.md).

### Tag index

| Group | Tags |
| --- | --- |
| Window and panels | `root`, `status_text`, `left_panel`, `center_panel`, `right_panel` |
| Tab bars and tabs | `left_tabs`: `tab_tutorial`, `tab_setup`, `tab_map`, `tab_loot`, `tab_tributes`, `tab_brains`, `tab_play`, `tab_train`, `tab_research`; `right_tabs`: `tab_inspector`, `tab_network`, `tab_charts` |
| Setup | `cfg_shape`, `cfg_layout`, `cfg_size`, `cfg_seed`, `cfg_chaos`, `cfg_days`, `cfg_tpd`, `cfg_players`, `cfg_thirst`, `cfg_hunger`, `cfg_health`, `cfg_vision`, `cfg_landmark`, `cfg_thirst_days`, `cfg_hunger_days`, `cfg_cannon`, `cfg_endgame`, `cfg_sponsors`, `cfg_gift`, `cfg_gm`, `cfg_quiet`, `cfg_close`, `cfg_water_podiums` |
| Map and Loot | `map_preset`, `stamp_radius`, `map_coverage`, `loot_weapon_name`, `loot_count` |
| Tributes | `podium_preset`, `roster_table`, `editor_header`, `ed_name`, `ed_district`, `ed_sex`, `ed_score`, `ed_survival`, `ed_brain`, `ed_weapon`, `ed_food`, `ed_medicine`, `ed_favor`, `ed_thirst`, `ed_hunger`, `ed_health` |
| Brains | `cfg_brain`, `nn_layer_count`, `nn_widths_group`, `nn_width_0` ... `nn_width_5`, `nn_activation`, `nn_init`, `nn_init_note`, `nn_scale`, `nn_sparsity`, `nn_summary` |
| Play | `auto_next_box`, `gif_step` |
| Train | `ga_group`, `rl_group`, `train_start`, `train_progress`, `train_summary`, `feed_mode`, `perf_plot`, `perf_x`, `perf_y`, `perf_train`, `perf_val`, `perf_mean`, `stab_plot`, `stab_x`, `stab_y`, `stab_ploss`, `stab_vloss`, `stab_entropy`, `time_plot`, `time_x`, `time_y`, `time_bars`, `gene_plot`, `gene_x`, `gene_y`, `gene_same`, `gene_changed`, `run_name` |
| Research | `sweep_param`, `sweep_values`, `sweep_games`, `sweep_workers`, `sweep_telemetry`, `sweep_progress`, `sweep_results`, `export_folder` |
| Transport | `play_button`, `speed_slider`, `playhead`, `headline` |
| Inspector | `insp_title`, `insp_facts`, `insp_thirst`, `insp_hunger`, `insp_health`, `insp_more`, `insp_log` |
| Network | `network_caption`, `evolution_header`, `evo_plot`, `evo_x`, `evo_y`, `evo_change`, `evo_mean`, `evo_heat_plot`, `evo_heat_x`, `evo_heat_y`, `evo_heat_series`, `network_holder` |
| Charts | `chart_actions`, `chart_actions_x`, `chart_actions_y`, `chart_actions_bars`, `chart_instinct`, `chart_instinct_x`, `chart_instinct_y`, `chart_drink`, `chart_eat`, `chart_flee`, `chart_heat`, `chart_heat_x`, `chart_heat_y`, `chart_heat_series` |
| Other files | `arena_canvas`, `arena_texture_<n>` ([canvas.md](canvas.md)); `network_canvas` ([visualizer.md](visualizer.md)) |

## How to use it / experiment

**Add a config slider.** Inside `_build_setup`, add `dpg.add_slider_float(label=..., default_value=c.<field>, min_value=..., max_value=..., callback=self._setter("<field>"), tag="cfg_<field>")`, and add the tag to the list in `_load_config` so loading a file refreshes it. Pass `react="size"` if the map must be regenerated.

**Add a tutorial step.** Append `("10. My step", "What to do.", "mystep")` to `TUTORIAL_STEPS` and add an `elif name == "mystep":` branch to `_tutorial_action` that calls session methods and ends with `dpg.set_value("left_tabs", "tab_...")`. To take a picture of it, add a row to `SHOTS` in [screenshots.md](screenshots.md).

**Switch tabs from code.** `dpg.set_value("left_tabs", "tab_train")` and `dpg.set_value("right_tabs", "tab_network")`. Reading `dpg.get_value("left_tabs")` gives the tag of the open tab.

**Set an architecture from code.** `dpg.set_value("nn_layer_count", 2)` does not fire the callback, so call `self._rebuild_width_fields([32, 16])` and then `self._on_apply_neural()`, which is what `_load_config` does.

**Add a mouse tool.** Append a name to `TOOLS`, then handle it in `_on_mouse_down` or `_on_mouse_click` with `self.canvas.mouse_cell()` and a session method.

**Add a chart.** Build a `dpg.plot` in `_build_charts` with tagged axes and series, and fill it in `_refresh_charts` from `summary` (the keys are listed in `BehaviorTelemetry.summary`). To change the panel split, edit `LEFT_FRACTION` and `RIGHT_FRACTION` or the margins in `_layout`; the arena stays square.

## Gotchas

- Painting, loot and dragging are ignored while a game is loaded (`session.game is not None`). Press Load replay or restart the dashboard to clear a game; a loaded replay sets `game` to `None`, so editing works again, but the map shown is the replay's. The training feed in `replay` mode loads recordings the same way.
- `cfg_size` and `cfg_players` apply on Enter, not on every keystroke.
- The "nodes in hidden layer N" fields have no callbacks. Typing a width changes nothing until "Apply network settings" is pressed; the caption on the Network tab and any new game keep the old architecture until then.
- Changing "number of hidden layers" rebuilds the width fields at once, but still through `_read_widths`, so widths already typed are kept. Shrinking drops layers from the end.
- `_tutorial_action("train")` starts a fixed short evolution (`TrainingConfig(brain_name="voting", population_size=24, generations=5, rounds_per_generation=1, validation_games=1)`) without touching `self.ga`, switches the Train tab to the genetic group and sets the feed to live; the Train tab widgets keep showing your own settings.
- `_tutorial_action("train")` starts whichever method the Train tab radio shows. With `reinforce` selected it starts a reinforce run and ignores the voting settings it just made.
- `_tutorial_action("network")` calls `_on_brain_all`, which drops every trained genome in the roster.
- `_setter` writes the config field even while a game is playing; the running game keeps its own config copy, so changes apply to the next New game.
- Load replay does not refresh the Setup widgets even though the session adopted the recording's config; Load config does. The training feed's `replay` mode adopts a config the same way.
- The Reward function sliders write `session.config.reward` directly, not a copy, and the sliders are not refreshed by Load config.
- Training plots, including the evolution plots on the Network tab, are updated only when the row count changes, so a slow generation shows a frozen plot with a moving progress bar. Because they update from `_refresh_training`, they fill in even while the Network tab is not open.
- `evo_heat_series` is deleted and recreated on every new step, and the colour scale is recomputed from the largest absolute gene so far, so colours can shift between steps. Only the first 200 genes are drawn.
- `_refresh_charts` runs every 30 frames and is skipped entirely until a game has produced telemetry.
- The GIF export and To end run on the GUI thread and block the window until they finish.
- `dpg.set_axis_ticks("gene_x", ...)` is only set for the voting brain's 8 genes; after a voting run followed by a neural run the names stay on the axis until the dashboard restarts.
