"""ui - the game makers' dashboard, built with Dear PyGui.

    python -m hunger_games ui

The package is split so the interesting logic has no GUI in it:

    painter.py   edits a terrain grid (brushes, stamps, presets)
    session.py   holds everything the dashboard is doing (no GUI code)
    canvas.py    draws the arena and handles mouse clicks on it
    app.py       lays out the windows, tabs and buttons

`launch()` opens the dashboard. Dear PyGui is only imported when it is
called, so `session.py` and `painter.py` can be used and tested without a
display.
"""


def launch() -> None:
    """Open the dashboard window (blocks until it is closed)."""
    # Import here so importing the package never needs a display.
    from hunger_games.ui.app import Dashboard

    # Build and run.
    Dashboard().run()


# What `from hunger_games.ui import *` exposes.
__all__ = ["launch"]
