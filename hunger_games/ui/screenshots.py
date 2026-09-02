"""ui/screenshots.py - take the tutorial's pictures from the real dashboard.

`capture_tutorial_images` opens the dashboard, performs each tutorial step
with the same code the 'Show me' buttons run, renders a few frames, and
saves the window to a PNG. The repo tutorial (docs/tutorial/README.md)
embeds those pictures, so they always show the real interface.

    python -m hunger_games.ui.screenshots docs/tutorial/images
"""

# Running the operating system's screenshot tool.
import platform
import subprocess

# Command-line argument.
import sys

# Waiting for background training.
import time

# Paths.
from pathlib import Path

# Dear PyGui.
import dearpygui.dearpygui as dpg

# Terrain kinds for the demo stroke.
from hunger_games.terrain import TerrainType

# The dashboard.
from hunger_games.ui.app import Dashboard

# The pictures to take: (file name, tutorial action or None, tabs to show, frames to render first).
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


# The window title the screenshot session uses (the capture looks the window up by this name).
WINDOW_TITLE = "tutorial screenshots"


def find_window_id(title: str) -> int | None:
    """Ask macOS for the id of the on-screen window with this title (None elsewhere or if not found)."""
    # Only macOS has this API.
    if platform.system() != "Darwin":
        return None
    # The window server's list of on-screen windows.
    import Quartz  # noqa: PLC0415 - only importable on macOS

    windows = Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID)
    # Find ours.
    for window in windows:
        if window.get("kCGWindowName") == title:
            return int(window["kCGWindowNumber"])
    # Not found.
    return None


def save_window_image(path: Path) -> None:
    """Save a picture of the dashboard window only.

    Dear PyGui's own frame-buffer export does not work on macOS, so there the
    operating system's `screencapture` grabs just our window by its id (never
    the rest of the screen); on other systems the frame buffer is used.
    """
    # macOS: capture only our window.
    if platform.system() == "Darwin":
        window_id = find_window_id(WINDOW_TITLE)
        # Without an id there is nothing safe to capture.
        if window_id is None:
            raise RuntimeError("Could not find the dashboard window to capture")
        subprocess.run(["screencapture", "-x", "-o", "-l", str(window_id), str(path)], check=True)
        return
    # Elsewhere: the frame buffer.
    dpg.output_frame_buffer(str(path))


def capture_tutorial_images(folder: str | Path, width: int = 1440, height: int = 880) -> list[Path]:
    """Open the dashboard, walk the tutorial, and save one PNG per step into `folder`."""
    # Where the pictures go.
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    # Build the dashboard exactly as `run()` does, but drive the frames ourselves.
    dashboard = Dashboard()
    dpg.create_context()
    dashboard._load_font()
    dashboard._apply_theme()
    dpg.create_viewport(title=WINDOW_TITLE, width=width, height=height)
    dashboard.build()
    dpg.set_viewport_resize_callback(lambda: dashboard._layout())
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dashboard._layout()

    # Render a number of frames.
    def frames(count: int) -> None:
        for _ in range(count):
            dashboard.on_frame()
            dpg.render_dearpygui_frame()

    # The saved paths.
    written = []
    # Each shot.
    for name, action, (left_tab, right_tab), count in SHOTS:
        # Perform the step.
        if action is not None:
            dashboard._tutorial_action(action)
        # The paint step also paints a demo stroke so the picture shows the brush at work.
        if action == "paint":
            painter = dashboard.session.painter
            for step in range(12):
                dashboard.session.paint(painter.width // 4 + step * 2, painter.height // 3 + step, TerrainType.ROCK, 3)
            dashboard.session.finish_painting()
            dashboard.brush_demo = (painter.width // 4 + 24, painter.height // 3 + 12, 3)
        # Training: finish the game on screen so the feed can show a champion game, and wait for a few steps.
        if action == "train":
            dashboard.brush_demo = None
            dashboard.session.run_to_end()
            while dashboard.session.training_running and len(dashboard.session.training_history()) < 3:
                frames(1)
                time.sleep(0.01)
            # Let the feed pick up the newest champion.
            frames(5)
        # Show the right tabs.
        dpg.set_value("left_tabs", left_tab)
        dpg.set_value("right_tabs", right_tab)
        # Let the picture settle.
        frames(count)
        # Save.
        path = folder / name
        save_window_image(path)
        # Give the save a frame to complete.
        frames(2)
        written.append(path)
    # Let training finish before tearing down.
    dashboard.session.stop_training()
    while dashboard.session.training_running:
        frames(1)
        time.sleep(0.01)
    # Clean up.
    dpg.destroy_context()
    # Done.
    return written


# `python -m hunger_games.ui.screenshots <folder>`.
if __name__ == "__main__":
    paths = capture_tutorial_images(sys.argv[1] if len(sys.argv) > 1 else "docs/tutorial/images")
    print("\n".join(str(p) for p in paths))
