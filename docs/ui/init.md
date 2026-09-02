# `__init__.py`

**Source:** [hunger_games/ui/__init__.py](../../hunger_games/ui/__init__.py)
**Depends on:** nothing at import time; `launch()` imports [app.md](app.md) (`Dashboard`) when called
**Used by:** [../main.md](../main.md) (`python -m hunger_games ui` calls `launch`), [main.md](main.md) (`python -m hunger_games.ui`)

## Purpose

This file makes `hunger_games/ui/` a Python package and gives it one public function, `launch()`, which opens the dashboard and blocks until the window is closed. Its docstring is also the map of the package:

| File | Job | Imports Dear PyGui? |
| --- | --- | --- |
| [painter.md](painter.md) | Edits a terrain grid: brushes, stamps, presets | No |
| [session.md](session.md) | Holds everything the dashboard is doing | No |
| [canvas.md](canvas.md) | Draws the arena and converts mouse positions to cells | Yes |
| [visualizer.md](visualizer.md) | Draws a neural network as a node graph | Yes |
| [app.md](app.md) | Lays out the panels, tabs and buttons | Yes |

The docstring names the first four; `visualizer.py` follows the same rule as `canvas.py`. The design rule stated here is worth remembering: the interesting logic has no GUI in it. Only `canvas.py`, `visualizer.py` and `app.py` import Dear PyGui, and this file makes sure that importing the package does not drag them in.

## Concepts you need

- **Packages.** A folder with an `__init__.py` is importable as `hunger_games.ui`. Whatever this file defines is what `import hunger_games.ui` gives you. Submodules like `hunger_games.ui.session` are imported separately and on demand.
- **Lazy imports.** `from hunger_games.ui.app import Dashboard` sits inside `launch()`, not at the top of the file. Importing `app` imports `dearpygui`, and importing `dearpygui` needs a graphics environment. Putting the import inside the function means `import hunger_games.ui.session` or `import hunger_games.ui.painter` never touches Dear PyGui, so those two modules work in tests, in scripts and on machines without a display.
- **`__all__`.** A list of names that `from hunger_games.ui import *` exports. Here it is just `["launch"]`. It is also a signal to readers: "this is the public surface".
- **Blocking calls.** `Dashboard().run()` contains the frame loop and does not return until the window closes, so `launch()` blocks too. Anything after `launch()` in a script runs after the user closes the window.
- **Docstrings as documentation.** The triple-quoted string at the top of the file is the module docstring. `help(hunger_games.ui)` prints it, which is why it carries the package map and the launch command.

## Walkthrough

### Module docstring

Explains the package layout and the launch command `python -m hunger_games ui`. There are no module-level constants.

### `launch`

```python
def launch() -> None
```

"Open the dashboard window (blocks until it is closed)."

1. Imports `Dashboard` from `hunger_games.ui.app`. This is the moment Dear PyGui is loaded.
2. Builds a `Dashboard()` and calls `.run()`.

Returns nothing. Any exception raised during construction or the frame loop propagates to the caller.

```python
from hunger_games.ui import launch
launch()          # returns when the window is closed
print("closed")
```

Design reasoning: a plain function is easier to call from the command line, from a notebook or from another script than a class name is, and it hides the `Dashboard` type so the internals can change without touching callers. `app.py` also defines its own `launch()` with the same body; this one is the public entry point and the one [../main.md](../main.md) imports.

### `__all__`

```python
__all__ = ["launch"]
```

### What happens when you launch, step by step

This is the chain of calls that starts from this file. Each step is documented on the page named.

| Step | Where | What happens |
| --- | --- | --- |
| 1 | `launch()` here | Imports `hunger_games.ui.app`, which imports Dear PyGui, `canvas`, `visualizer`, `session`, `painter` |
| 2 | `Dashboard.__init__` in [app.md](app.md) | Creates a `Session()`, an `ArenaCanvas`, a `NetworkVisualizer`, the default tool settings and the `TrainingConfig` and `RLConfig` being edited |
| 3 | `Session.__init__` in [session.md](session.md) | Builds a `MapPainter`, generates a Perlin arena and a roster |
| 4 | `Dashboard.run` in [app.md](app.md) | Creates the Dear PyGui context, loads a font and the theme, creates the viewport, builds the primary window with its three panels, registers the mouse handlers and the resize callback, lays the panels out |
| 5 | `Dashboard.on_frame`, every frame | Advances playback, redraws the arena, refreshes the transport bar, inspector, network, training and research panels, and the charts every 30th frame |
| 6 | Window closed | `run` destroys the context and returns; `launch` returns |

Nothing in step 3 needs a display. That is why the session can be tested on its own.

### Which module to import for what

| You want to | Import | Needs Dear PyGui? |
| --- | --- | --- |
| Open the window | `from hunger_games.ui import launch` | Yes, when called |
| Play, scrub, train or sweep from a script | `from hunger_games.ui.session import Session` | No |
| Paint or stamp a terrain grid | `from hunger_games.ui.painter import MapPainter` | No |
| Draw the arena in your own Dear PyGui window | `from hunger_games.ui.canvas import ArenaCanvas` | Yes |
| Draw a network snapshot | `from hunger_games.ui.visualizer import NetworkVisualizer` | Yes |
| Build the whole dashboard object | `from hunger_games.ui.app import Dashboard` | Yes |

The files on disk, in the order they depend on each other: `painter.py`, `session.py` (uses the painter), `canvas.py` and `visualizer.py` (read the session), `app.py` (builds all of them), then this file and `__main__.py` on top.

## How to use it / experiment

**Three ways to open the dashboard.** They all end up here.

```text
python -m hunger_games ui        # the package command line
python -m hunger_games.ui        # this package's own __main__
python -c "from hunger_games.ui import launch; launch()"
```

**Use the logic without the window.**

```python
from hunger_games.ui.session import Session     # no Dear PyGui import happens
from hunger_games.ui.painter import MapPainter

s = Session()
s.new_game(seed=1)
s.run_to_end()
print(s.status, s.recording.result.winner_name)
```

**Expose more.** If you add a public helper to the package (say a `headless_session()` factory that returns a ready `Session`), define it here or import it here, and add its name to `__all__`:

```python
def headless_session(seed: int | None = None):
    """A Session with no window, for scripts and tests."""
    from hunger_games.config import SimulationConfig
    from hunger_games.ui.session import Session
    return Session(SimulationConfig(seed=seed))

__all__ = ["launch", "headless_session"]
```

The import stays inside the function for the same reason `launch` does it: keep `import hunger_games.ui` cheap.

**Try the lazy import for yourself.** Run `python -c "import hunger_games.ui, sys; print('dearpygui' in sys.modules)"`. It prints `False`. Then call `launch` and it becomes `True`.

**Run the dashboard from a script and do something afterwards.**

```python
from hunger_games.ui import launch
launch()
print("The window was closed. Files you saved are on disk.")
```

Because `launch` blocks, anything that should happen while the window is open has to be done inside the dashboard itself (a button), not in the script.

## Gotchas

- Do not move the `Dashboard` import to the top of the file. Tests that import `hunger_games.ui.session` would start needing Dear PyGui and a display.
- `launch()` builds a brand new `Dashboard` and `Session` each time. There is no way to pass a prepared session in; if you need that, construct `Dashboard()` yourself and replace `dashboard.session` and `dashboard.canvas` before `run()`, keeping in mind that the canvas holds its own reference to the session and that the `Session` constructor has already generated an arena and roster.
- Calling `launch()` twice in one process is untested. Dear PyGui's context is created and destroyed inside `run()`, so it may work, but the window position and widget tags are all recreated from scratch.
- `python -m hunger_games.ui` and `python -m hunger_games ui` look alike. The first uses the dot (this package's [main.md](main.md)); the second uses a space (the package command line's `ui` sub-command).
- `from hunger_games.ui import *` gives you only `launch`. To get `Session` or `MapPainter`, import them from their modules.
- If Dear PyGui is missing, the `ImportError` appears when `launch()` is called, not when the package is imported. The message names `dearpygui`; install it with `pip install dearpygui`.
- Training and sweeps with several CPU workers start worker processes that re-import the program. The `__name__` guard in [main.md](main.md) and the fact that this file only opens a window inside `launch()` keep those workers from opening extra windows.
- Closing the window while a trainer or sweep is running abandons it. Those threads are daemons and die with the process; nothing is saved unless Save run folder was pressed first (the sweep writes its folder only when it finishes or is stopped).
- Importing `hunger_games.ui.canvas` or `hunger_games.ui.visualizer` directly does import Dear PyGui. Only `session` and `painter` are display-free.
- The module docstring lists four files; `visualizer.py` was added later and is not named there. `help(hunger_games.ui)` prints the docstring as it is.
- `python -m hunger_games ui` imports matplotlib (through `analysis` and `renderer`) before it reaches `launch()`, so it starts a little slower than `python -m hunger_games.ui`.
