"""renderer.py - draw a game (live, or from a recording) with matplotlib.

`ArenaFigure` knows how to draw one frame: terrain, supplies, tributes in
their district colours (circles for female, squares for male), parachutes
and the game makers' circle. `Renderer` feeds it a live `Game`;
`export_recording_gif` feeds it the frames of a `Recording`. The dashboard
does its own drawing, but uses `export_recording_gif` for GIF files.
"""

# Filesystem paths for saving.
from pathlib import Path

# matplotlib for drawing and animating.
import matplotlib.pyplot as plt

# numpy for the image array.
import numpy as np

# Frame-by-frame animation.
from matplotlib.animation import FuncAnimation

# A circle patch for the safe zone.
from matplotlib.patches import Circle

# District colours and sex markers.
from hunger_games.districts import SEX_MARKERS, district_color_rgb

# The game being drawn.
from hunger_games.game import Game

# Recordings.
from hunger_games.recorder import Frame, Recording, RosterEntry

# Supply kinds.
from hunger_games.resources import ResourceKind

# Terrain colours.
from hunger_games.terrain import TERRAIN_COLORS, TerrainType


def terrain_image(terrain: np.ndarray, heights: np.ndarray) -> np.ndarray:
    """Build an RGB image of the terrain with a little relief shading."""
    # An empty RGB image.
    rgb = np.zeros((*terrain.shape, 3), dtype=float)
    # Paint each terrain type its colour.
    for kind, color in TERRAIN_COLORS.items():
        # Cells of this type.
        mask = terrain == int(kind)
        # Fill them in.
        rgb[mask] = color
    # Higher ground is drawn a little brighter, lower ground a little darker.
    shade = 0.7 + 0.5 * heights
    # Don't shade the void.
    shade[terrain == int(TerrainType.VOID)] = 1.0
    # Apply the shading to all three channels.
    rgb = rgb * shade[..., None]
    # Keep the values legal.
    return np.clip(rgb, 0.0, 1.0)


class ArenaFigure:
    """A matplotlib figure that can redraw any frame of a game."""

    def __init__(
        self,
        terrain: np.ndarray,
        heights: np.ndarray,
        roster: list[RosterEntry],
        center: tuple[float, float],
        figsize: tuple[float, float] = (9, 9),
    ) -> None:
        """Set up the figure, the background and the empty overlays."""
        # The fixed facts about every tribute, by id.
        self.roster = {entry.player_id: entry for entry in roster}
        # Grid size.
        self.height, self.width = terrain.shape
        # The figure and the single axes inside it.
        self.fig, self.ax = plt.subplots(figsize=figsize)
        # Draw the terrain once as a background image.
        self.ax.imshow(terrain_image(terrain, heights), origin="upper", interpolation="nearest")
        # Weapons drawn as red triangles.
        self.weapon_dots = self.ax.scatter([], [], marker="^", c="red", s=14, zorder=3)
        # Food drawn as small white dots.
        self.food_dots = self.ax.scatter([], [], marker=".", c="white", s=10, zorder=3)
        # Medicine drawn as small magenta plus signs.
        self.medicine_dots = self.ax.scatter([], [], marker="+", c="magenta", s=16, zorder=3)
        # Female tributes: circles.
        self.female_dots = self.ax.scatter(
            [], [], marker=SEX_MARKERS["F"], s=70, edgecolors="white", linewidths=0.9, zorder=5
        )
        # Male tributes: squares.
        self.male_dots = self.ax.scatter(
            [], [], marker=SEX_MARKERS["M"], s=70, edgecolors="white", linewidths=0.9, zorder=5
        )
        # Parachutes drawn as white downward triangles.
        self.gift_dots = self.ax.scatter([], [], marker="v", c="white", s=60, edgecolors="black", zorder=6)
        # The game makers' safe circle, hidden until it matters.
        self.safe_circle = Circle(center, 1.0, fill=False, color="red", linestyle="--", linewidth=2.0, zorder=4)
        # Add it to the axes.
        self.ax.add_patch(self.safe_circle)
        # Hide it for now.
        self.safe_circle.set_visible(False)
        # No axis ticks: this is a map, not a chart.
        self.ax.set_xticks([])
        # Same for the y axis.
        self.ax.set_yticks([])
        # Show exactly the grid, edge to edge.
        self.ax.set_xlim(-0.5, self.width - 0.5)
        # Row 0 at the top, matching the image.
        self.ax.set_ylim(self.height - 0.5, -0.5)

    @staticmethod
    def _set_points(scatter, xs, ys, colors=None) -> None:
        """Update a scatter plot's points (and colours), handling the empty case."""
        # matplotlib needs an (N, 2) array even when N is 0.
        points = np.column_stack([xs, ys]) if len(xs) else np.empty((0, 2))
        # Apply the positions.
        scatter.set_offsets(points)
        # Apply the colours if given.
        if colors is not None:
            scatter.set_facecolors(np.array(colors) if len(colors) else np.empty((0, 3)))

    def draw_frame(self, frame: Frame, title: str) -> None:
        """Redraw every overlay for one frame."""
        # Supplies of each kind.
        for kind, scatter in (
            (ResourceKind.WEAPON, self.weapon_dots),
            (ResourceKind.FOOD, self.food_dots),
            (ResourceKind.MEDICINE, self.medicine_dots),
        ):
            # Where those supplies are.
            ys, xs = np.nonzero(frame.resource_kind == int(kind))
            # Move the markers there.
            self._set_points(scatter, xs, ys)
        # Living tributes, split by sex.
        for sex, scatter in (("F", self.female_dots), ("M", self.male_dots)):
            # Those alive and of this sex.
            group = [p for p in frame.players if p.alive and self.roster[p.player_id].sex == sex]
            # Positions and district colours.
            self._set_points(
                scatter,
                [p.x for p in group],
                [p.y for p in group],
                [district_color_rgb(self.roster[p.player_id].district) for p in group],
            )
        # Parachutes land next to their receiver.
        receivers = {gift.player_id for gift in frame.gifts}
        # Their positions.
        landed = [p for p in frame.players if p.player_id in receivers]
        # Draw them.
        self._set_points(self.gift_dots, [p.x for p in landed], [p.y - 1.5 for p in landed])
        # The safe circle appears once the game makers have intervened.
        self.safe_circle.set_visible(frame.circle_visible)
        # Match its size to the current safe radius.
        self.safe_circle.set_radius(frame.safe_radius)
        # The headline.
        self.ax.set_title(title)


def frame_title(frame: Frame, total_players: int, result=None) -> str:
    """The standard title line: day, tick, alive count, and the victor when known."""
    # How many are alive in this frame.
    alive = sum(p.alive for p in frame.players)
    # Base title.
    title = f"Day {frame.day}  tick {frame.tick}  alive {alive}/{total_players}"
    # Announce the outcome on the final frame.
    if result is not None and alive <= 1:
        title += f"  -  VICTOR: {result.winner_name}" if result.winner_name else "  -  no victor"
    # Done.
    return title


class Renderer:
    """Draws and animates one live `Game` with matplotlib."""

    def __init__(self, game: Game, ticks_per_frame: int = 1, figsize: tuple[float, float] = (9, 9)) -> None:
        """Set up the figure and draw the first frame."""
        # The game to draw.
        self.game = game
        # How many simulation ticks pass per animation frame.
        self.ticks_per_frame = max(1, ticks_per_frame)
        # A recorder gives us frames in the same shape the exporter uses.
        from hunger_games.recorder import Recorder

        # Start recording (frame 0 is captured immediately).
        self.recorder = Recorder(game)
        # The figure.
        self.figure = ArenaFigure(
            game.arena.terrain,
            game.arena.heights,
            self.recorder.recording.roster,
            (game.arena.center_x, game.arena.center_y),
            figsize,
        )
        # Convenience aliases used by older code and tests.
        self.fig, self.ax = self.figure.fig, self.figure.ax
        # Draw the first frame's overlays.
        self._redraw()

    def _redraw(self) -> None:
        """Refresh every overlay to match the current game state."""
        # The latest frame.
        frame = self.recorder.recording.frames[-1]
        # Draw it.
        self.figure.draw_frame(
            frame, frame_title(frame, len(self.game.players), self.game.result() if self.game.is_over else None)
        )

    def _advance(self, _frame) -> list:
        """Animation callback: step the game and redraw."""
        # Step several ticks per frame if asked.
        for _ in range(self.ticks_per_frame):
            # Stop stepping once the games end.
            if not self.game.is_over:
                self.recorder.step()
        # Refresh the overlays.
        self._redraw()
        # Return the artists that changed (blitting is off, so this is informational).
        return [self.figure.female_dots, self.figure.male_dots]

    def _frames(self):
        """Yield one frame number per animation step until the game ends."""
        # Frame counter.
        frame = 0
        # Keep going while the game is running.
        while not self.game.is_over:
            # Hand out the next frame number.
            yield frame
            # Count it.
            frame += 1
        # One last frame so the victor title is drawn.
        yield frame

    def show(self, interval_ms: int = 50) -> None:
        """Open a window and animate the game live."""
        # Build the animation (kept on self so it is not garbage-collected).
        self.animation = FuncAnimation(
            self.fig, self._advance, frames=self._frames, interval=interval_ms, repeat=False, cache_frame_data=False
        )
        # Show the window (blocks until it is closed).
        plt.show()

    def save(self, path: str | Path, fps: int = 20, max_frames: int = 600) -> None:
        """Write the animation to a GIF (or MP4 if ffmpeg is installed)."""
        # Limit the frame count so the file stays small.
        frames = list(range(max_frames))
        # Build the animation over a fixed frame list.
        animation = FuncAnimation(self.fig, self._advance, frames=frames, interval=1000 // fps, repeat=False)
        # Pick a writer from the file extension.
        writer = "ffmpeg" if str(path).endswith(".mp4") else "pillow"
        # Write the file.
        animation.save(str(path), writer=writer, fps=fps)

    def snapshot(self, path: str | Path) -> None:
        """Save the current frame as a PNG."""
        # Write the figure.
        self.fig.savefig(str(path), dpi=120, bbox_inches="tight")


def export_recording_gif(
    recording: Recording,
    path: str | Path,
    fps: int = 15,
    step: int = 2,
    max_frames: int = 600,
    figsize: tuple[float, float] = (8, 8),
) -> Path:
    """Write a recording to a GIF (or MP4), drawing every `step`-th frame."""
    # The arena centre for the safe circle.
    center = (recording.terrain.shape[1] // 2, recording.terrain.shape[0] // 2)
    # A figure to draw on.
    figure = ArenaFigure(recording.terrain, recording.heights, recording.roster, center, figsize)
    # Which frames to draw, always including the last one.
    indices = list(range(0, recording.length, max(1, step)))
    # Make sure the final frame is included.
    if indices[-1] != recording.length - 1:
        indices.append(recording.length - 1)
    # Cap the frame count.
    indices = indices[:max_frames]
    # Total tributes.
    total = len(recording.roster)

    # The per-frame callback.
    def draw(index: int) -> list:
        # The frame to draw.
        frame = recording.frames[indices[index]]
        # Draw it with the standard title.
        figure.draw_frame(
            frame, frame_title(frame, total, recording.result if indices[index] == recording.length - 1 else None)
        )
        # Nothing to blit.
        return []

    # Build the animation.
    animation = FuncAnimation(figure.fig, draw, frames=len(indices), interval=1000 // fps, repeat=False)
    # Pick a writer from the file extension.
    writer = "ffmpeg" if str(path).endswith(".mp4") else "pillow"
    # Write the file.
    animation.save(str(path), writer=writer, fps=fps)
    # Close the figure so memory is freed.
    plt.close(figure.fig)
    # Hand back the path.
    return Path(path)
