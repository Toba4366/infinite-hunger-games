# `canvas.py`

**Source:** [hunger_games/ui/canvas.py](../../hunger_games/ui/canvas.py)
**Depends on:** `math`, `dearpygui.dearpygui as dpg`, `numpy`; project modules [../districts.md](../districts.md) (`district_color_255`), [../renderer.md](../renderer.md) (`terrain_image`), [../resources.md](../resources.md) (`ResourceKind`), [session.md](session.md) (`Session`)
**Used by:** [app.md](app.md) (`Dashboard.canvas`; built into the centre panel, resized by `_layout`, rendered every frame by `on_frame`, asked for the cell under the mouse by the mouse handlers)

## Purpose

`canvas.py` draws the arena inside the dashboard and converts between mouse pixels and grid cells. The terrain is uploaded once as a texture and again only when the painter's version changes. Everything that moves is redrawn every frame on separate draw layers: loot markers, tributes (circles for female, squares for male, coloured by district), a gold star on every tribute driven by the learner network being trained, the yellow selection ring, the white brush preview ring, parachutes, red X marks for fresh eliminations and the game makers' circle.

The class reads the `Session` and never writes to it. Mouse handling lives in [app.md](app.md); this file only answers "which cell is under the mouse".

## Concepts you need

- **Drawlist.** `dpg.drawlist(width, height)` is a blank pixel area you draw onto with `draw_circle`, `draw_rectangle`, `draw_line`, `draw_triangle`, `draw_polygon`, `draw_text` and `draw_image`. Coordinates are pixels from the drawlist's top-left corner.
- **Draw layers.** `dpg.add_draw_layer()` inside a drawlist makes a group of drawings. `dpg.delete_item(layer, children_only=True)` clears just that group. Layers are drawn in creation order, so the terrain layer sits underneath and the overlay layer on top.
- **Dynamic textures.** `dpg.add_dynamic_texture(width, height, default_value=...)` inside a `dpg.texture_registry()` uploads an image. The value is a flat list of floats, RGBA per pixel, row by row, each channel 0.0 to 1.0. `dpg.set_value(tag, pixels)` updates it in place at the same size; a new size needs a new texture.
- **Tags.** The drawlist is `"arena_canvas"`; textures are `"arena_texture_1"`, `"arena_texture_2"` and so on. Tags must be unique, which is why recreated textures get a fresh number.
- **Hover and mouse position.** `dpg.is_item_hovered(tag)` says whether the mouse is over an item. `dpg.get_drawing_mouse_pos()` gives the mouse position relative to the drawlist under it.
- **Cell size.** The drawing is a square of `size` pixels and the map may be up to 300 cells, so each cell is `size / max(width, height)` pixels, usually not a whole number. Every marker is scaled by that number so it stays readable on small and large maps.
- **Version counters.** The painter bumps `version` whenever the map changes. The canvas remembers the version it last uploaded and re-uploads only when the number moved.
- **Learner stars.** Training trains one network. It plays a few roster slots (the learner slots) while every other tribute uses the voting brain. `Session.learner_ids_on_screen()` names those slots; the canvas draws a five-pointed gold star on each so you can follow the network being trained through a live game or a replayed training game. A polygon is drawn from a list of points with `dpg.draw_polygon`; the star is ten points that alternate between an outer and an inner radius.

## Walkthrough

### `LOOT_COLORS`

```python
LOOT_COLORS = {
    ResourceKind.FOOD: (255, 255, 255, 220),
    ResourceKind.WEAPON: (255, 60, 60, 255),
    ResourceKind.MEDICINE: (255, 80, 255, 255),
}
```

RGBA colours for loot markers: food white (slightly transparent), weapons red, medicine magenta.

### `class ArenaCanvas`

"The arena drawing plus coordinate conversion and hit testing."

#### `ArenaCanvas.__init__`

`def __init__(self, session: Session, size_px: int = 760) -> None`

Remembers the session and the pixel size. `tag = "arena_canvas"`. Texture bookkeeping: `texture_tag = None`, `_texture_version = -1`, `_texture_shape = None`, `_texture_serial = 0`. `show_labels = True` draws `D4F` style labels. `brush_preview = None` is an `(x, y, radius)` ring to draw. The four layer tags start as `None` and are filled by `build`.

#### `ArenaCanvas.resize`

`def resize(self, size_px: int) -> None`

Ignores the same size or anything under 100 pixels. Otherwise stores the size, reconfigures the drawlist if it exists, and sets `_texture_version = -1` so the terrain image is redrawn at the new scale on the next render.

#### `ArenaCanvas.build`

`def build(self, parent) -> None`

Creates the drawlist inside `parent` with four layers in order: `image_layer`, `loot_layer`, `tribute_layer`, `overlay_layer`.

#### `ArenaCanvas.cell`

`@property def cell(self) -> float`

Pixels per grid cell: `size / max(painter.width, painter.height)`.

#### `ArenaCanvas.to_px`, `ArenaCanvas.to_cell`

`def to_px(self, x: float, y: float) -> tuple[float, float]`, `def to_cell(self, px: float, py: float) -> tuple[int, int]`

A cell centre to pixels (`(x + 0.5) * cell`), and pixels to a cell by floor division.

#### `ArenaCanvas.mouse_cell`

`def mouse_cell(self) -> tuple[int, int] | None`

`None` unless the drawlist is hovered; otherwise the cell under `dpg.get_drawing_mouse_pos()`, or `None` when that cell is off the grid.

#### `ArenaCanvas._refresh_texture`

`def _refresh_texture(self) -> None`

Returns at once if the painter's version and shape match what was uploaded. Otherwise builds the RGB image with `terrain_image(painter.terrain, painter.heights)`, adds an alpha channel and flattens to floats. A new shape creates a new dynamic texture with a fresh serial tag and deletes the old one; the same shape updates the pixels in place. Then it clears the image layer and draws the texture over `width * cell` by `height * cell` pixels.

#### `ArenaCanvas.render`

`def render(self) -> None`

`_refresh_texture()`, `_draw_loot()`, `_draw_tributes()`, `_draw_overlays()`.

#### `ArenaCanvas._draw_loot`

`def _draw_loot(self) -> None`

Clears the loot layer. Marker half-size is `max(1.5, cell * 0.35)`. With a frame on screen, every non-empty cell of `frame.resource_kind` is drawn by kind; while editing, only the scenario's hand-placed loot.

#### `ArenaCanvas._loot_marker`

`def _loot_marker(self, kind: ResourceKind, x: int, y: int, half: float, color) -> None`

Weapons are a filled triangle, food a filled dot of radius `half * 0.6`, medicine a plus sign of two lines.

#### `ArenaCanvas._draw_tributes`

`def _draw_tributes(self) -> None`

Clears the tribute layer. Marker radius is `max(4.0, cell * 0.9)`. Positions come from `session.positions()` (the frame while watching, else podiums). Facts come from the editable roster, or from the recording's roster when the editable roster is empty and a recording is loaded. `learners = session.learner_ids_on_screen()` is the set of tributes driven by a learner brain.

For each positioned tribute with known facts: the district colour fills a circle (female) or a square (male) with a white outline of thickness 1.5. If the tribute's id is in `learners`, `_draw_star(px, py, radius * 1.1)` puts a gold star on top of the marker. With `show_labels` on, `D{district}{sex}` is drawn to the right at size `max(10, cell * 1.6)`.

#### `ArenaCanvas._draw_star`

`def _draw_star(self, px: float, py: float, radius: float) -> None`

A five-pointed gold star centred on a pixel position. Ten points are computed around the centre: even points at the full `radius`, odd points at `radius * 0.45`, at angles starting straight up (`-pi / 2`) and stepping by `pi / 5`. The points are drawn as one filled polygon on the tribute layer: fill `(255, 215, 0, 255)` (gold), outline `(40, 30, 0, 255)`, thickness 1.0. Because it is drawn after the marker, it sits on top of the district-coloured shape, and because it is on the tribute layer, the selection ring and parachutes still draw over it.

#### `ArenaCanvas._draw_overlays`

`def _draw_overlays(self) -> None`

Clears the overlay layer, then draws, in order:

| Overlay | When | What |
| --- | --- | --- |
| Selection ring | `session.selected_id` has a position | Yellow ring `(255, 230, 0)` of radius `max(7, cell * 1.6)`, thickness 2.5 |
| Brush preview | `brush_preview` is set | White ring of radius `(brush radius + 0.5) * cell`, thickness 1.5 |
| Parachutes | Frame on screen; each `frame.gifts` entry whose receiver has a position | White triangle with black outline above the tribute, size `max(4, cell)` |
| Eliminations | Frame on screen; each `frame.eliminations` entry | Red X of two lines, half-size `max(5, cell * 1.5)`, thickness 3, at the cell where they fell |
| Game makers' circle | `frame.circle_visible` | Red ring centred on `(width // 2, height // 2)` with radius `frame.safe_radius * cell`, thickness 2 |

Everything from parachutes on needs a frame; while editing the method returns after the brush preview.

## How to use it / experiment

**See the stars.** On the Train tab press Watch agent after a run, or set the training feed to `live`. The champion is written into the learner slots, and those tributes carry a star while the game plays. Click a starred tribute and open the Network tab to watch its network think. With the `replay` feed the stars follow the recording's roster, so they mark the slots the trainer's learner played in that game.

**Turn labels off.** The Map tab's "show tribute labels" checkbox sets `canvas.show_labels`. From code: `dashboard.canvas.show_labels = False`.

**Draw the brush ring somewhere.** `canvas.brush_preview = (60, 60, 5)` draws a ring of radius 5.5 cells at cell (60, 60) on the next render. `Dashboard.on_frame` sets and clears it every frame, so set it from there if you add a new tool.

**Change the star.** Edit the two colours in `_draw_star`, or the inner radius factor `0.45` (smaller makes sharper points). To star different tributes, change `Session.learner_ids_on_screen` or `Session.LEARNER_KINDS` rather than the canvas.

**Change a colour.** Edit `LOOT_COLORS` for loot, or the literal tuples in `_draw_tributes` and `_draw_overlays`. The district palette itself is in [../districts.md](../districts.md).

**Add an overlay.** Draw it in `_draw_overlays` on `self.overlay_layer`, after the selection ring so it stays on top of the terrain. For something that needs the live game rather than the frame, read `self.session.game`, but remember it is `None` for a loaded replay.

## Gotchas

- The canvas never writes to the session. Selection, painting and loot placement happen in [app.md](app.md).
- A star means "this tribute has a learner brain" (`neural` or `neat`) according to `Session.learner_ids_on_screen`. While editing or during a live game it also requires a stored genome, so a tribute set to `neural` on the Tributes tab but never trained gets no star; while watching a loaded replay or a replay-feed game it only requires the recording's roster to say `neural` or `neat`, so every neural tribute in a replay is starred, trained or not.
- With Champion to all every tribute is starred, which defeats the point of the star. Watch agent and the live feed star only the trainer's learner slots.
- `resize` does not re-upload a texture on its own; it only resets `_texture_version`. The image is redrawn at the next `render`, which happens every frame anyway.
- `mouse_cell` returns `None` outside the drawlist even when the mouse is inside the centre panel. Clicks on the transport bar never reach the arena.
- Positions of the dead are not drawn: `session.positions()` only returns living tributes when a frame is on screen. The red X marks where they fell, for one frame only.
- The game makers' circle is a solid ring, not the dashed line the matplotlib renderer draws. Dear PyGui's drawlist has no dashed circle.
- The centre of the circle is the painter's centre, `(width // 2, height // 2)`, which is also where `GameMaker` measures from. On a non-square map the circle is still round.
- A frame's `resource_kind` grid is one byte per cell, copied per tick. Loot drawn while watching is exactly what the game had at that tick, including the layout's scattered supplies; loot drawn while editing is only what you placed.
- `to_cell` uses floor division, so a mouse position just inside the right edge of the drawlist can map to a cell equal to the map width, which `mouse_cell` then rejects with `in_bounds`.
- Everything but the terrain is deleted and recreated every frame. With 96 tributes, labels and stars on, that is a few hundred items per frame, which is fine; heavy overlays should be throttled like the charts in `app.py`.
