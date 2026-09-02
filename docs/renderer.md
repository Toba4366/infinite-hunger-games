# `renderer.py`

**Source:** [hunger_games/renderer.py](../hunger_games/renderer.py)
**Depends on:** `pathlib.Path` (standard library), `numpy`, `matplotlib` (`pyplot`, `matplotlib.animation.FuncAnimation`, `matplotlib.patches.Circle`); project modules `districts.py` (`SEX_MARKERS`, `district_color_rgb`), [game.md](game.md) (`Game`), [recorder.md](recorder.md) (`Frame`, `Recording`, `RosterEntry`, and `Recorder` imported inside `Renderer.__init__`), [resources.md](resources.md) (`ResourceKind`), [terrain.md](terrain.md) (`TERRAIN_COLORS`, `TerrainType`).
**Used by:** [main.md](main.md) (the `watch` command builds a `Renderer`), `ui/session.py` (`export_recording_gif` for the dashboard's GIF button), `ui/canvas.py` (`terrain_image` for the dashboard's map).

## Purpose

This file draws a game with matplotlib. It is split into three layers so the same picture can be produced from a live game or from a saved recording.

- `terrain_image` turns the terrain and height grids into an RGB image with light relief shading.
- `ArenaFigure` owns one matplotlib figure and knows how to redraw any `Frame` on it: supplies, tributes, parachutes and the game makers' circle.
- `Renderer` wraps a live `Game` in a `Recorder` and animates it. `export_recording_gif` walks the frames of a finished `Recording` instead.

Tributes are coloured by district using the table in `districts.py`, and shaped by sex: circles for female, squares for male. That lets you tell the two tributes from one district apart. Parachutes appear as white triangles above whoever received a gift that tick.

The dashboard (`hunger_games/ui`) draws its own live view with Dear PyGui. It only borrows `terrain_image` and `export_recording_gif` from here.

## Concepts you need

**Image coordinates.** `imshow(origin="upper")` puts row 0 at the top. The axes are flipped to match (`set_ylim(height - 0.5, -0.5)`), so a tribute at `(x, y)` is plotted at `(x, y)` with no conversion.

**Scatter plots as layers.** Each kind of marker is one `scatter` object created empty in `__init__`. Redrawing means calling `set_offsets` with new positions, which is much faster than clearing and replotting.

**`zorder`.** Higher numbers draw on top. Supplies are 3, the circle 4, tributes 5, parachutes 6.

**Broadcasting.** `rgb * shade[..., None]` multiplies a `(H, W, 3)` image by a `(H, W, 1)` array, so one shade value applies to all three colour channels of a cell.

**`FuncAnimation`.** matplotlib calls your function once per frame with the next value from `frames`. `frames` can be a list (fixed length) or a generator (runs until it stops). The animation object must be kept alive or it is garbage-collected mid-run.

**Writers.** Pillow writes GIFs. ffmpeg writes MP4 and must be installed separately.

## Walkthrough

### `terrain_image(terrain: np.ndarray, heights: np.ndarray) -> np.ndarray`

Builds an `(H, W, 3)` float image. Each terrain type is painted with its `TERRAIN_COLORS` entry. Then `shade = 0.7 + 0.5 * heights` brightens high ground and darkens low ground, except `VOID` cells, which stay unshaded. The result is clipped to `0.0..1.0`.

```python
from hunger_games.config import SimulationConfig
from hunger_games.game import Game
from hunger_games.renderer import terrain_image

game = Game(SimulationConfig(seed=1))
img = terrain_image(game.arena.terrain, game.arena.heights)
print(img.shape, img.min(), img.max())   # (120, 120, 3) 0.0 ... <= 1.0
```

### `class ArenaFigure`

#### `__init__(self, terrain: np.ndarray, heights: np.ndarray, roster: list[RosterEntry], center: tuple[float, float], figsize: tuple[float, float] = (9, 9)) -> None`

Creates the figure and axes, draws the terrain once as the background, and creates every overlay empty.

| Attribute | Marker | Colour | Size | zorder |
| --- | --- | --- | --- | --- |
| `weapon_dots` | `^` | red | 14 | 3 |
| `food_dots` | `.` | white | 10 | 3 |
| `medicine_dots` | `+` | magenta | 16 | 3 |
| `female_dots` | `SEX_MARKERS["F"]` (`o`) | district colour, white edge | 70 | 5 |
| `male_dots` | `SEX_MARKERS["M"]` (`s`) | district colour, white edge | 70 | 5 |
| `gift_dots` | `v` | white, black edge | 60 | 6 |
| `safe_circle` | `Circle` | red dashed, width 2 | radius 1.0 | 4 |

`self.roster` is a dict from `player_id` to `RosterEntry`. Axis ticks are removed and the limits are set to show exactly the grid.

#### `_set_points(scatter, xs, ys, colors=None) -> None` (static)

Updates a scatter's positions with `set_offsets`, handling the empty case by passing a `(0, 2)` array. If `colors` is given, updates face colours too, again with an empty `(0, 3)` array when there are none.

#### `draw_frame(self, frame: Frame, title: str) -> None`

Redraws every overlay for one frame:

1. For each of `WEAPON`, `FOOD`, `MEDICINE`, find the cells of that kind in `frame.resource_kind` and move the markers there.
2. For each sex, take the living snapshots whose roster entry has that sex, and set their positions and district colours.
3. Collect the `player_id`s in `frame.gifts`, find those players' positions, and draw a parachute 1.5 cells above each.
4. Show or hide the circle from `frame.circle_visible` and set its radius from `frame.safe_radius`.
5. Set the title.

### `frame_title(frame: Frame, total_players: int, result=None) -> str`

`"Day {day}  tick {tick}  alive {n}/{total}"`. If a `result` is given and at most one tribute is alive, appends `"  -  VICTOR: name"` or `"  -  no victor"`.

### `class Renderer`

#### `__init__(self, game: Game, ticks_per_frame: int = 1, figsize: tuple[float, float] = (9, 9)) -> None`

Stores the game and `max(1, ticks_per_frame)`. Creates a `Recorder(game)`, which captures frame 0 at once. Builds an `ArenaFigure` from the arena's grids, the recording's roster and `(arena.center_x, arena.center_y)`. Exposes `self.fig` and `self.ax` as aliases, then calls `_redraw()`.

#### `_redraw(self) -> None`

Draws the last recorded frame with a title from `frame_title`, passing `game.result()` only when the game is over.

#### `_advance(self, _frame) -> list`

The animation callback. Calls `recorder.step()` up to `ticks_per_frame` times, stopping early if the game ends, then `_redraw()`. Returns the two tribute scatters.

#### `_frames(self)`

A generator yielding frame numbers while the game runs, plus one more so the victor title is drawn.

#### `show(self, interval_ms: int = 50) -> None`

Builds a `FuncAnimation` over `_frames` and calls `plt.show()`, which blocks until the window closes. The animation is stored on `self.animation` so it survives.

#### `save(self, path: str | Path, fps: int = 20, max_frames: int = 600) -> None`

Builds a `FuncAnimation` over a fixed list of `max_frames` frame numbers and writes it. The writer is `"ffmpeg"` for paths ending in `.mp4`, otherwise `"pillow"`.

#### `snapshot(self, path: str | Path) -> None`

`fig.savefig(path, dpi=120, bbox_inches="tight")`. A PNG of the current frame.

```python
from hunger_games.config import SimulationConfig
from hunger_games.game import Game
from hunger_games.renderer import Renderer

r = Renderer(Game(SimulationConfig(seed=9, width=60, height=60)), ticks_per_frame=2)
r.snapshot("start.png")
r.save("game.gif", fps=15, max_frames=200)
```

### `export_recording_gif(recording: Recording, path: str | Path, fps: int = 15, step: int = 2, max_frames: int = 600, figsize: tuple[float, float] = (8, 8)) -> Path`

Draws a finished recording to a file. The centre is `(width // 2, height // 2)` from the terrain shape, matching the arena. Frame indices are `range(0, length, step)` with the final frame always appended, then cut to `max_frames`. Each animation frame calls `figure.draw_frame` with the standard title, passing `recording.result` only on the last frame. Writes with Pillow or ffmpeg by extension, closes the figure, and returns the `Path`.

```python
from hunger_games.recorder import Recording
from hunger_games.renderer import export_recording_gif

rec = Recording.load("game7.replay")
export_recording_gif(rec, "game7.gif", fps=10, step=4)
```

## How to use it / experiment

**Watch from the command line.**

```bash
python -m hunger_games watch --seed 3 --speed 2
python -m hunger_games watch --seed 3 --save game3.gif
```

**Slow it down.** `Renderer.show(interval_ms=200)` gives five frames a second. `ticks_per_frame=1` shows every tick.

**Draw a single recorded tick.** Build an `ArenaFigure` yourself and call `draw_frame` once, then `figure.fig.savefig(...)`. This is how you get a still of tick 200 from a replay without exporting the whole GIF.

**Change the look.** Marker shapes and sizes are all in `ArenaFigure.__init__`. District colours live in `districts.py`. Terrain colours live in `terrain.py`. The relief strength is the `0.5` in `terrain_image`.

**Keep GIFs small.** `export_recording_gif(step=4, max_frames=300)` quarters the frame count. Lower `figsize` shrinks each frame.

## Gotchas

- `Renderer.save` always writes exactly `max_frames` frames. After the game ends `_advance` keeps redrawing the final state, so a short game produces a GIF that freezes on the victor for the rest of its length. `export_recording_gif` does not have this problem because it only has as many frames as the recording.
- `Renderer.show` and `Renderer.save` share one recorder. Calling `save` after `show` continues from wherever the window was closed, and a finished game yields a static file.
- `frame_title` only announces the outcome when at most one tribute is alive. A game that hits the day limit with several survivors shows no "no victor" text.
- The figure background is drawn once. A scenario whose terrain changes mid-game is not supported.
- Parachutes are drawn for one frame only, on the frame whose `gifts` list is non-empty. At `ticks_per_frame > 1` or `step > 1` you can miss them entirely.
- MP4 needs ffmpeg on your `PATH`. Without it, use `.gif`.
- `plt.show()` needs a display. On a headless machine use `save` or `snapshot`, or set `MPLBACKEND=Agg`.
- `export_recording_gif` closes its own figure. `Renderer` does not; call `plt.close(r.fig)` in loops that build many renderers.
- Dead tributes vanish from the map immediately. There is no corpse marker.
