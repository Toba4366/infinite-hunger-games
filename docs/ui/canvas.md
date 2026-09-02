# `canvas.py`

**Source:** [hunger_games/ui/canvas.py](../../hunger_games/ui/canvas.py)
**Depends on:** `dearpygui.dearpygui as dpg`, `numpy`; project modules [../districts.md](../districts.md) (`district_color_255`), [../renderer.md](../renderer.md) (`terrain_image`), [../resources.md](../resources.md) (`ResourceKind`), [session.md](session.md) (`Session`)
**Used by:** [app.md](app.md) (`Dashboard.canvas`; built into the centre panel, resized by `_layout`, rendered every frame by `on_frame`, asked for the cell under the mouse by the mouse handlers)

## Purpose

`canvas.py` draws the arena inside the dashboard and converts between mouse pixels and grid cells. The terrain is uploaded once as a texture and again only when the painter's version changes. Everything that moves is redrawn every frame on separate draw layers: loot markers, tributes (circles for female, squares for male, coloured by district), the yellow selection ring, the white brush preview ring, parachutes, red X marks for fresh eliminations and the game makers' circle.

The class reads the `Session` and never writes to it. Mouse handling lives in [app.md](app.md); this file only answers "which cell is under the mouse".

## Concepts you need

- **Drawlist.** `dpg.drawlist(width, height)` is a blank pixel area you draw onto with `draw_circle`, `draw_rectangle`, `draw_line`, `draw_triangle`, `draw_text` and `draw_image`. Coordinates are pixels from the drawlist's top-left corner.
- **Draw layers.** `dpg.add_draw_layer()` inside a drawlist makes a group of drawings. `dpg.delete_item(layer, children_only=True)` clears just that group. Layers are drawn in creation order, so the terrain layer sits underneath and the overlay layer on top.
- **Dynamic textures.** `dpg.add_dynamic_texture(width, height, default_value=...)` inside a `dpg.texture_registry()` uploads an image. The value is a flat list of floats, RGBA per pixel, row by row, each channel 0.0 to 1.0. `dpg.set_value(tag, pixels)` updates it in place at the same size; a new size needs a new texture.
- **Tags.** The drawlist is `"arena_canvas"`; textures are `"arena_texture_1"`, `"arena_texture_2"` and so on. Tags must be unique, which is why recreated textures get a fresh number.
- **Hover and mouse position.** `dpg.is_item_hovered(tag)` says whether the mouse is over an item. `dpg.get_drawing_mouse_pos()` gives the mouse position relative to the drawlist under it.
- **Cell size.** The drawing is a square of `size` pixels and the map may be up to 300 cells, so each cell is `size / max(width, height)` pixels, usually not a whole number. Every marker is scaled by that number so it stays readable on small and large maps.
- **Version counters.** The painter bumps `version` whenever the map changes. The canvas remembers the version it last uploaded and re-uploads only when the number moved.

## Walkthrough

### `LOOT_COLORS`

```python
LOOT_COLORS = {
    ResourceKind.FOOD: (255, 255, 255, 220),
    ResourceKind.WEAPON: (255, 60, 60, 255),
    ResourceKind.MEDICINE: (255, 80, 255, 255),
}
```

RGBA colours for loot markers: food white (slightly transparent), weapons red, medicine magenta. The dictionary order is also the drawing order when a frame is on screen.

### `class ArenaCanvas`

"The arena drawing plus coordinate conversion and hit testing."

#### `ArenaCanvas.__init__`

```python
def __init__(self, session: Session, size_px: int = 760) -> None
```

Remembers the session and the pixel size, sets `tag = "arena_canvas"`, and initialises the texture bookkeeping: `texture_tag = None`, `_texture_version = -1`, `_texture_shape = None`, `_texture_serial = 0`. Two public settings: `show_labels = True` (the `D4F` labels) and `brush_preview = None`, a `(x, y, radius)` tuple in cells that the dashboard sets while the Paint terrain tool is hovering the arena. The four layer handles are `None` until `build`.

#### `ArenaCanvas.resize`

```python
def resize(self, size_px: int) -> None
```

Changes the pixel size on a window resize. Does nothing if the size is unchanged or below 100. Otherwise stores the new size, reconfigures the drawlist's `width` and `height` if it exists, and sets `_texture_version = -1` so the next `render` redraws the terrain image at the new scale. `Dashboard._layout` calls it with `min(center_width - 24, panel_height - 190)`.

#### `ArenaCanvas.build`

```python
def build(self, parent) -> None
```

Creates the square drawlist inside `parent` with four layers in this order: `image_layer` (terrain), `loot_layer`, `tribute_layer`, `overlay_layer`.

#### `ArenaCanvas.cell`

```python
@property
def cell(self) -> float
```

Pixels per grid cell: `size / max(painter.width, painter.height)`. Read from the painter every time, so it is right after a scenario or replay changes the map size.

#### `ArenaCanvas.to_px`

```python
def to_px(self, x: float, y: float) -> tuple[float, float]
```

Grid cell to the pixel centre of that cell: `((x + 0.5) * cell, (y + 0.5) * cell)`.

#### `ArenaCanvas.to_cell`

```python
def to_cell(self, px: float, py: float) -> tuple[int, int]
```

Drawlist pixels to a grid cell by floor division: `(int(px // cell), int(py // cell))`.

#### `ArenaCanvas.mouse_cell`

```python
def mouse_cell(self) -> tuple[int, int] | None
```

The cell under the mouse, or `None` when the mouse is not hovering the drawlist or the cell is off the grid (`painter.in_bounds`). The dashboard calls this for painting, dragging, clicking and the brush preview.

#### `ArenaCanvas._refresh_texture`

```python
def _refresh_texture(self) -> None
```

Uploads the terrain image if the map changed. Returns early when `painter.version` equals `_texture_version` and the shape is unchanged. Otherwise builds the RGB image with `terrain_image(painter.terrain, painter.heights)`, appends an alpha channel of ones, flattens it to a list of `float32`, and either creates a new dynamic texture (new shape: fresh tag `arena_texture_<serial>`, old texture deleted) or updates the existing one with `dpg.set_value`. Finally it remembers the version, clears the image layer and draws the texture from `(0, 0)` to `(width * cell, height * cell)`.

#### `ArenaCanvas.render`

```python
def render(self) -> None
```

The per-frame entry point: `_refresh_texture`, `_draw_loot`, `_draw_tributes`, `_draw_overlays`, in that order.

#### `ArenaCanvas._draw_loot`

```python
def _draw_loot(self) -> None
```

Clears the loot layer. Marker half-size is `max(1.5, cell * 0.35)`. If `session.current_frame` is not `None` (a game or replay is on screen), it draws every non-zero cell of the frame's `resource_kind` grid, one kind at a time. Otherwise it draws the hand-placed stacks in `session.scenario.loot`. The layout's own scattered loot is not shown while editing; it only exists once a game has been built.

#### `ArenaCanvas._loot_marker`

```python
def _loot_marker(self, kind: ResourceKind, x: int, y: int, half: float, color) -> None
```

One marker at the cell's pixel centre: weapons are filled triangles, food is a dot of radius `half * 0.6`, medicine is a plus sign of two lines with thickness 2.

#### `ArenaCanvas._draw_tributes`

```python
def _draw_tributes(self) -> None
```

Clears the tribute layer. Marker radius is `max(4.0, cell * 0.9)`. Positions come from `session.positions()` (live positions of the living when a frame is on screen, else podiums). Facts (district, sex) come from `session.tributes`; if that roster is empty and a recording is loaded, from `recording.roster`. Unknown ids are skipped. Female tributes are circles, male tributes squares, both filled with `district_color_255(district)` and outlined white at thickness 1.5. With `show_labels` on, the text `D{district}{sex}` is drawn to the right at size `max(10, cell * 1.6)`.

#### `ArenaCanvas._draw_overlays`

```python
def _draw_overlays(self) -> None
```

Clears the overlay layer, then draws in this order:

| Overlay | When | Look |
| --- | --- | --- |
| Selection ring | `session.selected_id` has a position | Yellow ring, radius `max(7, cell * 1.6)`, thickness 2.5 |
| Brush preview | `brush_preview` is set | White ring `(255, 255, 255, 200)`, radius `(brush_radius + 0.5) * cell`, thickness 1.5 |
| Parachutes | Frame on screen; each `frame.gifts` entry whose receiver has a position | White triangle with black outline above the tribute, size `max(4, cell)` |
| Eliminations | Frame on screen; each `frame.eliminations` entry | Red X of two lines, half-size `max(5, cell * 1.5)`, thickness 3, at the cell where they fell |
| Game makers' circle | `frame.circle_visible` | Red ring centred on `(width // 2, height // 2)` with radius `frame.safe_radius * cell`, thickness 2 |

Everything from parachutes on needs a frame; while editing the method returns after the brush preview.

## How to use it / experiment

**Turn labels off.** The Map tab's "show tribute labels" checkbox sets `canvas.show_labels`. From code: `dashboard.canvas.show_labels = False`.

**Draw the brush ring somewhere.** `canvas.brush_preview = (60, 60, 5)` draws a ring of radius 5.5 cells at cell (60, 60) on the next render. `Dashboard.on_frame` sets and clears it every frame, so set it from there if you add a new tool.

**Change a colour.** Edit `LOOT_COLORS` for loot, or the literal tuples in `_draw_tributes` and `_draw_overlays`. The district palette itself is in [../districts.md](../districts.md).

**Add an overlay.** Draw it in `_draw_overlays` on `self.overlay_layer`, after the selection ring so it stays on top of the terrain. For something that needs the live game rather than the frame, read `self.session.game`, but remember it is `None` for a loaded replay.

**Use a different image builder.** `_refresh_texture` accepts any function that returns an `(height, width, 3)` float RGB array; `terrain_image` is shared with the matplotlib renderer so both views match.

## Gotchas

- The canvas never writes to the session. Selection, painting and loot placement happen in [app.md](app.md).
- `resize` does not re-upload a texture on its own; it only resets `_texture_version`. The image is redrawn at the next `render`, which happens every frame anyway.
- `mouse_cell` returns `None` outside the drawlist even when the mouse is inside the centre panel. Clicks on the transport bar never reach the arena.
- Positions of the dead are not drawn: `session.positions()` only returns living tributes when a frame is on screen. The red X marks where they fell, for one frame only.
- The game makers' circle is a solid ring, not the dashed line the matplotlib renderer draws. Dear PyGui's drawlist has no dashed circle.
- The centre of the circle is the painter's centre, `(width // 2, height // 2)`, which is also where `GameMaker` measures from. On a non-square map the circle is still round.
- A frame's `resource_kind` grid is one byte per cell, copied per tick. Loot drawn while watching is exactly what the game had at that tick, including the layout's scattered supplies; loot drawn while editing is only what you placed.
- `to_cell` uses floor division, so a mouse position just inside the right edge of the drawlist can map to a cell equal to the map width, which `mouse_cell` then rejects with `in_bounds`.
- Everything but the terrain is deleted and recreated every frame. With 96 tributes and labels on, that is a few hundred items per frame, which is fine; heavy overlays should be throttled like the charts in `app.py`.
