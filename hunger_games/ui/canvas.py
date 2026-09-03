"""ui/canvas.py - draws the arena in the dashboard and maps mouse clicks to cells.

The terrain is uploaded once as a texture (and again whenever it is
painted). Everything that moves is redrawn each frame on draw layers:
loot, tributes (circles for female, squares for male, coloured by
district), the selection ring, parachutes and the game makers' circle.
"""

# Trigonometry for the star.
import math

# Dear PyGui.
import dearpygui.dearpygui as dpg

# numpy for the texture.
import numpy as np

# District colours.
from hunger_games.districts import district_color_255

# The terrain image builder shared with the matplotlib renderer.
from hunger_games.renderer import terrain_image

# Supply kinds.
from hunger_games.resources import ResourceKind

# Terrain kinds.
# The dashboard state.
from hunger_games.ui.session import Session

# Colours for loot markers, RGBA 0..255.
LOOT_COLORS = {
    ResourceKind.FOOD: (255, 255, 255, 220),
    ResourceKind.WEAPON: (255, 60, 60, 255),
    ResourceKind.MEDICINE: (255, 80, 255, 255),
}


class ArenaCanvas:
    """The arena drawing plus coordinate conversion and hit testing."""

    def __init__(self, session: Session, size_px: int = 760) -> None:
        """Remember the session and the pixel size of the drawing."""
        # The state to draw.
        self.session = session
        # The drawing is a square of this many pixels.
        self.size = size_px
        # The drawlist tag.
        self.tag = "arena_canvas"
        # The texture tag currently in use (recreated when the map size changes).
        self.texture_tag: str | None = None
        # Which painter version the texture shows.
        self._texture_version = -1
        # Which map shape the texture has.
        self._texture_shape: tuple[int, int] | None = None
        # A counter so recreated textures get fresh tags.
        self._texture_serial = 0
        # Whether to draw "D4F" labels next to tributes.
        self.show_labels = True
        # A brush preview to draw: (x, y, radius in cells), or None.
        self.brush_preview: tuple[int, int, int] | None = None
        # Layer tags, filled in by build().
        self.image_layer = self.loot_layer = self.tribute_layer = self.overlay_layer = None

    # ------------------------------------------------------------ layout

    def resize(self, size_px: int) -> None:
        """Change the pixel size (on window resize) and force a redraw of the terrain."""
        # Nothing to do.
        if size_px == self.size or size_px < 100:
            return
        # New size.
        self.size = size_px
        # The drawlist.
        if dpg.does_item_exist(self.tag):
            dpg.configure_item(self.tag, width=size_px, height=size_px)
        # Force the image to be redrawn at the new scale.
        self._texture_version = -1

    def build(self, parent) -> None:
        """Create the drawlist and its layers inside a parent window."""
        # The drawlist.
        with dpg.drawlist(width=self.size, height=self.size, tag=self.tag, parent=parent):
            # Terrain image.
            self.image_layer = dpg.add_draw_layer()
            # Loot markers.
            self.loot_layer = dpg.add_draw_layer()
            # Tributes.
            self.tribute_layer = dpg.add_draw_layer()
            # Rings, circles, parachutes.
            self.overlay_layer = dpg.add_draw_layer()

    @property
    def cell(self) -> float:
        """Pixels per grid cell."""
        # Fit the larger dimension.
        return self.size / max(self.session.painter.width, self.session.painter.height)

    def to_px(self, x: float, y: float) -> tuple[float, float]:
        """Grid cell centre to drawlist pixels."""
        # Centre of the cell.
        return (x + 0.5) * self.cell, (y + 0.5) * self.cell

    def to_cell(self, px: float, py: float) -> tuple[int, int]:
        """Drawlist pixels to grid cell."""
        # Floor division by the cell size.
        return int(px // self.cell), int(py // self.cell)

    def mouse_cell(self) -> tuple[int, int] | None:
        """The grid cell under the mouse, or None if the mouse is not over the arena."""
        # Only when hovering the drawlist.
        if not dpg.is_item_hovered(self.tag):
            return None
        # Mouse position relative to the drawlist.
        px, py = dpg.get_drawing_mouse_pos()
        # Convert.
        x, y = self.to_cell(px, py)
        # Only cells on the grid.
        return (x, y) if self.session.painter.in_bounds(x, y) else None

    # ----------------------------------------------------------- texture

    def _refresh_texture(self) -> None:
        """Upload the terrain image if the map changed since last time."""
        # The painter.
        painter = self.session.painter
        # Nothing changed.
        if painter.version == self._texture_version and self._texture_shape == painter.terrain.shape:
            return
        # The RGB image, plus an alpha channel.
        rgb = terrain_image(painter.terrain, painter.heights)
        # RGBA flat list of floats, as Dear PyGui wants.
        rgba = np.concatenate([rgb, np.ones((*rgb.shape[:2], 1))], axis=2).astype(np.float32).ravel().tolist()
        # Height and width.
        height, width = painter.terrain.shape
        # A new size needs a new texture object.
        if self._texture_shape != painter.terrain.shape:
            # Fresh tag.
            self._texture_serial += 1
            # New tag.
            new_tag = f"arena_texture_{self._texture_serial}"
            # Create it.
            with dpg.texture_registry():
                dpg.add_dynamic_texture(width, height, default_value=rgba, tag=new_tag)
            # Drop the old one.
            if self.texture_tag is not None and dpg.does_item_exist(self.texture_tag):
                dpg.delete_item(self.texture_tag)
            # Use the new one.
            self.texture_tag = new_tag
            # Remember the shape.
            self._texture_shape = painter.terrain.shape
        # Same size: just update the pixels.
        else:
            dpg.set_value(self.texture_tag, rgba)
        # Remember the version.
        self._texture_version = painter.version
        # Redraw the image layer.
        dpg.delete_item(self.image_layer, children_only=True)
        # The image covers width x height cells.
        dpg.draw_image(self.texture_tag, (0, 0), (width * self.cell, height * self.cell), parent=self.image_layer)

    # ------------------------------------------------------------ drawing

    def render(self) -> None:
        """Redraw everything for the current session state."""
        # Terrain.
        self._refresh_texture()
        # Loot.
        self._draw_loot()
        # Tributes.
        self._draw_tributes()
        # Overlays.
        self._draw_overlays()

    def _draw_loot(self) -> None:
        """Draw supplies: from the current frame if watching, else the hand-placed loot."""
        # Clear.
        dpg.delete_item(self.loot_layer, children_only=True)
        # Marker half-size in pixels.
        half = max(1.5, self.cell * 0.35)
        # The frame on screen.
        frame = self.session.current_frame
        # Watching a game: draw the frame's grid.
        if frame is not None:
            # Each kind.
            for kind, color in LOOT_COLORS.items():
                # Where.
                ys, xs = np.nonzero(frame.resource_kind == int(kind))
                # Draw each.
                for x, y in zip(xs.tolist(), ys.tolist(), strict=False):
                    self._loot_marker(kind, x, y, half, color)
        # Editing: draw the hand-placed loot.
        else:
            for loot in self.session.scenario.loot:
                self._loot_marker(ResourceKind(loot.kind), loot.x, loot.y, half, LOOT_COLORS[ResourceKind(loot.kind)])

    def _loot_marker(self, kind: ResourceKind, x: int, y: int, half: float, color) -> None:
        """One loot marker: weapons are triangles, food dots, medicine plus signs."""
        # Pixel centre.
        px, py = self.to_px(x, y)
        # Weapons: triangle.
        if kind is ResourceKind.WEAPON:
            dpg.draw_triangle(
                (px, py - half),
                (px - half, py + half),
                (px + half, py + half),
                color=color,
                fill=color,
                parent=self.loot_layer,
            )
        # Food: dot.
        elif kind is ResourceKind.FOOD:
            dpg.draw_circle((px, py), half * 0.6, color=color, fill=color, parent=self.loot_layer)
        # Medicine: plus.
        else:
            dpg.draw_line((px - half, py), (px + half, py), color=color, thickness=2, parent=self.loot_layer)
            dpg.draw_line((px, py - half), (px, py + half), color=color, thickness=2, parent=self.loot_layer)

    def _draw_tributes(self) -> None:
        """Draw every tribute at their podium or live position."""
        # Clear.
        dpg.delete_item(self.tribute_layer, children_only=True)
        # Marker radius in pixels.
        radius = max(4.0, self.cell * 0.9)
        # Who is where.
        positions = self.session.positions()
        # Facts about each tribute.
        roster = {t.player_id: t for t in self.session.tributes}
        # If watching a loaded replay, the roster comes from the recording.
        if self.session.recording is not None and not roster:
            roster = {e.player_id: e for e in self.session.recording.roster}
        # Which tributes are driven by a learner brain (they get a star).
        learners = self.session.learner_ids_on_screen()
        # Draw each.
        for player_id, (x, y) in positions.items():
            # Facts.
            entry = roster.get(player_id)
            # Skip unknowns.
            if entry is None:
                continue
            # District colour.
            r, g, b = district_color_255(entry.district)
            # Pixel centre.
            px, py = self.to_px(x, y)
            # Female: circle.
            if entry.sex == "F":
                dpg.draw_circle(
                    (px, py),
                    radius,
                    color=(255, 255, 255, 255),
                    fill=(r, g, b, 255),
                    thickness=1.5,
                    parent=self.tribute_layer,
                )
            # Male: square.
            else:
                dpg.draw_rectangle(
                    (px - radius, py - radius),
                    (px + radius, py + radius),
                    color=(255, 255, 255, 255),
                    fill=(r, g, b, 255),
                    thickness=1.5,
                    parent=self.tribute_layer,
                )
            # A gold star on learners, so you can follow the network being trained.
            if player_id in learners:
                self._draw_star(px, py, radius * 1.1)
            # Label.
            if self.show_labels:
                dpg.draw_text(
                    (px + radius + 1, py - radius),
                    f"D{entry.district}{entry.sex}",
                    color=(255, 255, 255, 230),
                    size=max(10, self.cell * 1.6),
                    parent=self.tribute_layer,
                )

    def _draw_star(self, px: float, py: float, radius: float) -> None:
        """A five-pointed gold star centred on a pixel position."""
        # Ten points alternating outer and inner radius.
        points = []
        for i in range(10):
            r = radius if i % 2 == 0 else radius * 0.45
            angle = -math.pi / 2 + i * math.pi / 5
            points.append((px + r * math.cos(angle), py + r * math.sin(angle)))
        # Draw as a filled polygon.
        dpg.draw_polygon(
            points, color=(40, 30, 0, 255), fill=(255, 215, 0, 255), thickness=1.0, parent=self.tribute_layer
        )

    def _draw_overlays(self) -> None:
        """Selection ring, parachutes and the game makers' circle."""
        # Clear.
        dpg.delete_item(self.overlay_layer, children_only=True)
        # Where everyone is.
        positions = self.session.positions()
        # Selection ring.
        if self.session.selected_id in positions:
            # Pixel centre.
            px, py = self.to_px(*positions[self.session.selected_id])
            # A yellow ring.
            dpg.draw_circle(
                (px, py), max(7.0, self.cell * 1.6), color=(255, 230, 0, 255), thickness=2.5, parent=self.overlay_layer
            )
        # The brush preview while painting.
        if self.brush_preview is not None:
            # Pixel centre.
            px, py = self.to_px(self.brush_preview[0], self.brush_preview[1])
            # A white ring the size of the brush.
            dpg.draw_circle(
                (px, py),
                (self.brush_preview[2] + 0.5) * self.cell,
                color=(255, 255, 255, 200),
                thickness=1.5,
                parent=self.overlay_layer,
            )
        # The frame on screen.
        frame = self.session.current_frame
        # Nothing more while editing.
        if frame is None:
            return
        # Parachutes above receivers.
        for gift in frame.gifts:
            # Their position.
            if gift.player_id in positions:
                # Pixel centre.
                px, py = self.to_px(*positions[gift.player_id])
                # A small white parachute triangle.
                s = max(4.0, self.cell)
                dpg.draw_triangle(
                    (px, py - s * 1.2),
                    (px - s, py - s * 3),
                    (px + s, py - s * 3),
                    color=(0, 0, 0, 255),
                    fill=(255, 255, 255, 255),
                    parent=self.overlay_layer,
                )
        # Fresh eliminations flash a red X.
        for elimination in frame.eliminations:
            # Pixel centre of where they fell.
            px, py = self.to_px(elimination.x, elimination.y)
            # Size.
            s = max(5.0, self.cell * 1.5)
            # Two red lines.
            dpg.draw_line(
                (px - s, py - s), (px + s, py + s), color=(255, 40, 40, 255), thickness=3, parent=self.overlay_layer
            )
            dpg.draw_line(
                (px - s, py + s), (px + s, py - s), color=(255, 40, 40, 255), thickness=3, parent=self.overlay_layer
            )
        # The game makers' circle.
        if frame.circle_visible:
            # Centre of the arena.
            cx, cy = self.to_px(self.session.painter.width // 2, self.session.painter.height // 2)
            # Dashed look is not available, so a thin red ring.
            dpg.draw_circle(
                (cx, cy),
                frame.safe_radius * self.cell,
                color=(255, 40, 40, 255),
                thickness=2,
                parent=self.overlay_layer,
            )
