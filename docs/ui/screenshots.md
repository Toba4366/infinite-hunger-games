# `screenshots.py`

**Source:** [hunger_games/ui/screenshots.py](../../hunger_games/ui/screenshots.py)
**Depends on:** `platform`, `subprocess`, `sys`, `time`, `pathlib.Path`, `dearpygui.dearpygui as dpg`; `Quartz` (imported inside `find_window_id`, macOS only; it comes from the `pyobjc-framework-Quartz` package, listed in `requirements.txt` for macOS); project modules [../terrain.md](../terrain.md) (`TerrainType`), [app.md](app.md) (`Dashboard`)
**Used by:** [../tutorial/README.md](../tutorial/README.md) (embeds the pictures it writes to `docs/tutorial/images/`); run by hand with `python -m hunger_games.ui.screenshots`

## Purpose

`screenshots.py` takes the tutorial's pictures from the real dashboard. It opens the window, performs each tutorial step with the same code the Tutorial tab's "Show me" buttons run (`Dashboard._tutorial_action`), switches to the tabs that step lives in, renders a few frames so the picture settles, and saves the window as a PNG. Because the pictures come from the live interface, the written tutorial never drifts from what the dashboard shows.

## Concepts you need

- **Driving frames by hand.** `Dashboard.run` loops `on_frame()` and `dpg.render_dearpygui_frame()` until the window closes. This file repeats the setup steps from `run` but keeps the loop for itself, so it can render exactly `n` frames, then save, then move on. No mouse handlers are registered; nothing here reacts to the real mouse.
- **Window id.** Every on-screen window on macOS has a number the window server knows it by. `Quartz.CGWindowListCopyWindowInfo` lists the windows with their titles (`kCGWindowName`) and numbers (`kCGWindowNumber`). Finding our window by title gives the id that `screencapture` needs.
- **`screencapture -l`.** macOS's command-line screenshot tool. `-l <id>` captures one window by id, `-x` makes no camera sound, `-o` leaves out the window's shadow. Nothing outside that window ends up in the file.
- **Frame buffer export.** `dpg.output_frame_buffer(path)` writes what Dear PyGui just rendered. It does not work on macOS, which is why the operating system's tool is used there.
- **Screen Recording permission.** Since macOS 10.15 an app may only read other windows' contents (and, for other apps, their titles) if it has Screen Recording permission. The app that runs the script (Terminal, iTerm, VS Code, and so on) must be allowed under System Settings, Privacy & Security, Screen Recording, or the capture fails or comes out blank.
- **Tabs by tag.** `dpg.set_value("left_tabs", "tab_map")` switches a tab bar to the tab with that tag. The tags are listed in [app.md](app.md).

## Walkthrough

### `SHOTS`

```python
SHOTS = [
    ("01_overview.png", None, ("tab_tutorial", "tab_inspector"), 3),
    ("02_arena.png", "arena", ("tab_map", "tab_inspector"), 3),
    ("03_paint.png", "paint", ("tab_map", "tab_inspector"), 3),
    ("04_tributes.png", "tributes", ("tab_tributes", "tab_inspector"), 3),
    ("05_loot.png", "loot", ("tab_loot", "tab_inspector"), 3),
    ("06_play.png", "play", ("tab_play", "tab_inspector"), 40),
    ("07_network.png", "network", ("tab_brains", "tab_network"), 30),
    ("08_train.png", "train", ("tab_train", "tab_network"), 60),
    ("09_research.png", "research", ("tab_research", "tab_charts"), 3),
]
```

One row per picture: the file name, the tutorial action to perform first (`None` for the overview), the left and right tabs to show, and how many frames to render before saving. The actions are the names `Dashboard._tutorial_action` accepts; their effects are tabled in [app.md](app.md). The larger frame counts give a game time to play (`play`, `network`) and the training plots time to fill (`train`). Since the shots run in order on one dashboard, each picture also shows everything the earlier steps did: the lake island from `arena`, the rock stroke from `paint`, and so on.

| File | Action | Left tab | Right tab | Frames |
| --- | --- | --- | --- | --- |
| `01_overview.png` | none | Tutorial | Inspector | 3 |
| `02_arena.png` | `arena` (loads `lake_island`) | Map | Inspector | 3 |
| `03_paint.png` | `paint` (Paint terrain tool) plus the demo stroke | Map | Inspector | 3 |
| `04_tributes.png` | `tributes` (selects the first tribute) | Tributes | Inspector | 3 |
| `05_loot.png` | `loot` (Place loot tool) | Loot | Inspector | 3 |
| `06_play.png` | `play` (new game at 8 ticks per second) | Play | Inspector | 40 |
| `07_network.png` | `network` (everyone neural, game at 4 ticks per second, first tribute selected) | Brains | Network | 30 |
| `08_train.png` | `train` (5 generations of voting brains, feed `live`) | Train | Network | 60 |
| `09_research.png` | `research` | Research | Charts | 3 |

### `WINDOW_TITLE`

```python
WINDOW_TITLE = "tutorial screenshots"
```

The viewport title used for the screenshot session. It is not the dashboard's normal title, so a dashboard you already have open is never captured by mistake. `find_window_id` looks the window up by this exact string.

### `find_window_id`

`def find_window_id(title: str) -> int | None`

Returns `None` at once on anything but macOS (`platform.system() != "Darwin"`). On macOS it imports `Quartz` (only there, hence the import inside the function), asks for the on-screen windows with `CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)`, and returns `int(window["kCGWindowNumber"])` of the first whose `kCGWindowName` equals `title`. `None` if no window matches.

### `save_window_image`

`def save_window_image(path: Path) -> None`

Saves a picture of the dashboard window only.

- On macOS: `find_window_id(WINDOW_TITLE)`; without an id it raises `RuntimeError("Could not find the dashboard window to capture")`, because there is nothing safe to capture. With an id it runs `screencapture -x -o -l <id> <path>` with `check=True`, so a failed capture raises `CalledProcessError`.
- Elsewhere: `dpg.output_frame_buffer(str(path))`.

Why not the frame buffer everywhere: Dear PyGui's frame-buffer export does not work on macOS. Why not a whole-screen capture: it would put everything else on the screen, other windows included, into a file that is committed to the repository. Capturing by window id keeps the picture to our own window and nothing else.

### `capture_tutorial_images`

`def capture_tutorial_images(folder: str | Path, width: int = 1440, height: int = 880) -> list[Path]`

Opens the dashboard, walks the tutorial and saves one PNG per row of `SHOTS` into `folder`. Returns the paths written.

1. Makes `folder` (and parents) if needed.
2. Builds the dashboard as `Dashboard.run` does, but by hand: `Dashboard()`, `dpg.create_context()`, `_load_font()`, `_apply_theme()`, `dpg.create_viewport(title=WINDOW_TITLE, width=width, height=height)`, `build()`, the resize callback, `dpg.setup_dearpygui()`, `dpg.show_viewport()`, `_layout()`. No mouse handlers.
3. Defines `frames(count)`, which calls `dashboard.on_frame()` then `dpg.render_dearpygui_frame()` `count` times.
4. For each shot, in order:
   - Performs the action with `dashboard._tutorial_action(action)`, if there is one.
   - For `paint`, also paints the demo stroke (below) so the picture shows the brush at work.
   - For `train`, renders single frames with a 10 ms sleep between them until training has stopped or `training_history()` holds at least 3 steps, so the plots have points.
   - Sets `left_tabs` and `right_tabs` to the shot's tabs.
   - Renders the shot's frame count, saves with `save_window_image`, renders 2 more frames so the save can complete, and remembers the path.
5. Stops training with `session.stop_training()` and renders single frames (10 ms apart) until `training_running` is false.
6. `dpg.destroy_context()` and returns the list.

**The paint demo stroke.** Twelve dabs of `TerrainType.ROCK` with radius 3, through `session.paint(painter.width // 4 + step * 2, painter.height // 3 + step, ROCK, 3)` for `step` 0 to 11: a short diagonal line starting a quarter of the way across and a third of the way down the map. Then `session.finish_painting()` recomputes the heights, and `canvas.brush_preview` is set to `(width // 4 + 24, height // 3 + 12, 3)`, the cell just past the end of the stroke, so the white brush ring is drawn there.

### The `__main__` block

```text
python -m hunger_games.ui.screenshots [folder]
```

Calls `capture_tutorial_images(sys.argv[1])`, or `"docs/tutorial/images"` when no folder is given, and prints one written path per line.

## How to use it / experiment

**Regenerate the tutorial's pictures.** From the project root, with the Screen Recording permission granted to your terminal on macOS:

```text
python -m hunger_games.ui.screenshots docs/tutorial/images
```

A window titled "tutorial screenshots" opens, runs through the nine steps in a few seconds (the training step takes the longest), and closes. The nine PNGs replace the ones [../tutorial/README.md](../tutorial/README.md) embeds. Do not touch the mouse or switch windows while it runs.

**Take one picture from a script.**

```python
from pathlib import Path
from hunger_games.ui.screenshots import capture_tutorial_images
paths = capture_tutorial_images(Path("output/shots"), width=1280, height=800)
print(paths[0])
```

**Add a picture.** Append a row to `SHOTS`, for example `("10_charts.png", None, ("tab_play", "tab_charts"), 3)`. To photograph a new tutorial step, add its action to `Dashboard._tutorial_action` first (see [app.md](app.md)). Then reference the new file from the tutorial page.

**Check the window is found.** In a Python shell on macOS while the dashboard is open under that title: `from hunger_games.ui.screenshots import find_window_id; find_window_id("tutorial screenshots")`. `None` means the title does not match or the window is not on screen.

## Gotchas

- macOS needs Screen Recording permission for the app that runs the command. Without it `screencapture` cannot read the window and the PNG is blank or missing, and window titles may be hidden from `find_window_id`, which then raises "Could not find the dashboard window to capture".
- `Quartz` comes from `pyobjc-framework-Quartz`, which `requirements.txt` installs on macOS only; on other systems the frame buffer path is used and Quartz is never imported.
- `dpg.output_frame_buffer` is only used off macOS. Whether it works there depends on the Dear PyGui build and the graphics driver; if the PNGs are empty, that is where to look.
- `Dashboard.on_frame` draws the ring at the real mouse position when the mouse is over the arena, and otherwise at `Dashboard.brush_demo`, which the paint step sets; so the ring appears in the picture unless the mouse happens to be over the arena. The rock stroke itself is in the map and always appears.
- The `train` shot starts a real training run in a background thread. The `network` action before it gave everyone a neural brain, but the run evolves voting brains (`brain_name="voting"`) with the feed set to `live`. The feed only takes over the arena once the game from the `network` step is over and fully watched; that game is usually still running after 30 plus 60 frames, so `08_train.png` normally shows it, not a champion game, with the headline unchanged. The wait loop stops when 3 steps exist, so the plots show 3 generations. If the Train tab's method were `reinforce` the same action would start a reinforce run instead; the tool never changes it, so a fresh `Dashboard` uses `genetic`.
- The shots share one dashboard, so their order matters: swapping rows changes what the later pictures contain.
- The teardown waits for the trainer to finish the current generation before destroying the context; with a large population that can take a few seconds.
- The tool opens a real window, so it needs a display. It is not a test and is not run by `pytest`.
