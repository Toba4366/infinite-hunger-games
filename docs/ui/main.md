# `__main__.py`

**Source:** [hunger_games/ui/__main__.py](../../hunger_games/ui/__main__.py)
**Depends on:** [init.md](init.md) (`launch`)
**Used by:** nobody imports it; Python runs it for `python -m hunger_games.ui`

## Purpose

This is the smallest file in the package. It exists so that `python -m hunger_games.ui` opens the dashboard. It imports `launch` from the package and calls it under the usual `if __name__ == "__main__":` guard. The whole file is eight lines.

The usual way to open the dashboard is still `python -m hunger_games ui` (with a space), which goes through the package command line in [../main.md](../main.md). This file is a convenience so the UI package is runnable on its own, the same way `hunger_games/__main__.py` makes the top-level package runnable.

## Concepts you need

- **`python -m package`.** When you run a package with `-m`, Python looks for a file called `__main__.py` inside it and runs that file as the main program. `python -m hunger_games` runs `hunger_games/__main__.py`; `python -m hunger_games.ui` runs `hunger_games/ui/__main__.py`.
- **`__name__`.** Every module has a `__name__`. When a file is run as the program it is `"__main__"`; when it is imported it is the module's dotted name. The guard `if __name__ == "__main__":` means "only do this when I am the program, not when someone imports me".
- **Relative versus absolute imports.** The file uses the absolute form `from hunger_games.ui import launch`. Under `-m` the package is properly set up, so this import works. Running the file by path (`python hunger_games/ui/__main__.py`) would not set the package up and the import would fail; always use `-m`.
- **Package `__init__` runs first.** Importing `hunger_games.ui` runs `hunger_games/ui/__init__.py`, which defines `launch`. Because that file delays importing Dear PyGui until `launch()` is called, the import line here is cheap; the heavy work happens on the last line.
- **The current directory matters.** `-m` searches `sys.path`, whose first entry is the directory you run from. So the command works from the project root, or from anywhere once the package is installed with `pip install -e .`.

## Walkthrough

### Module docstring

```python
"""Run the dashboard with `python -m hunger_games.ui`."""
```

One line. It states the only reason the file exists.

### The import

```python
from hunger_games.ui import launch
```

Brings in the public entry point from [init.md](init.md). Nothing else is imported, so the file adds no dependencies of its own.

### The guard and the call

```python
if __name__ == "__main__":
    launch()
```

When run with `-m`, `__name__` is `"__main__"`, so `launch()` runs, which builds a `Dashboard` and enters its frame loop. The call blocks until the window closes, after which the process exits normally.

Design reasoning: keeping the file to an import and a guarded call means there is nothing here to test and nothing to break. All behaviour lives in `launch` and, below it, in [app.md](app.md). If command-line options are ever wanted for the dashboard, the right place to add them is the `ui` sub-command in [../main.md](../main.md), where `argparse` already exists, rather than here.

### How this differs from the package command line

Both open the same window. The difference is what happens before `launch()`.

| | `python -m hunger_games ui` | `python -m hunger_games.ui` |
| --- | --- | --- |
| File that runs | `hunger_games/__main__.py` | `hunger_games/ui/__main__.py` (this file) |
| Parses arguments | Yes, with `argparse`; `ui` is one sub-command next to `watch`, `simulate`, `analyze` | No |
| Imports at start | `analysis`, `config`, `game`, `renderer`, `runner` (so matplotlib loads too) | Only `hunger_games.ui` |
| Accepts options | Only the sub-command name; the `ui` command takes no flags | None |
| Calls | `from hunger_games.ui import launch; launch()` | `launch()` |

This file starts a little faster because it skips matplotlib, which the other commands need.

## How to use it / experiment

**Run it.**

```text
python -m hunger_games.ui
```

**Compare with the package command line.** `python -m hunger_games ui` parses arguments with `argparse`, sees the `ui` command and then imports and calls the same `launch`. The result is identical. This file just skips the argument parsing.

**Add a flag.** If you want, say, `--seed` for the dashboard, the minimal change here is:

```python
import argparse
from hunger_games.ui import launch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="hunger_games.ui")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    launch()
```

`launch()` takes no arguments today, so to use `args.seed` you would also give `launch` and `Dashboard.__init__` a parameter that is passed into `Session(SimulationConfig(seed=...))`. See [app.md](app.md) and [session.md](session.md).

**Make a double-clickable launcher.** On macOS a two-line shell script works:

```text
#!/bin/sh
cd "$(dirname "$0")" && python -m hunger_games.ui
```

Save it next to the `hunger_games` folder and mark it executable with `chmod +x`.

**Confirm the guard works.** Import the file instead of running it and nothing should open:

```text
python -c "import hunger_games.ui.__main__; print('imported, no window')"
```

The import succeeds, `__name__` is `"hunger_games.ui.__main__"`, the guard is false, and no window appears.

### Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `No module named hunger_games` | Running from the wrong folder | `cd` to the project root, or `pip install -e .` |
| `No module named dearpygui` | Dear PyGui not installed | `pip install dearpygui` |
| `ImportError: attempted relative import` or similar when running by path | Ran `python hunger_games/ui/__main__.py` | Use `python -m hunger_games.ui` |
| Window opens, then extra windows appear during training | The `__name__` guard was removed | Restore the guard so worker processes do not call `launch()` |
| The command returns at once with no window | Another error printed above it | Read the traceback; it is usually the import of Dear PyGui |

## Gotchas

- Run it with `-m`. `python hunger_games/ui/__main__.py` fails with an import error because `hunger_games` is not on the path as a package.
- Run it from the project root (the folder that contains `hunger_games/`) or install the package, or Python will not find `hunger_games`.
- There are no options. Everything is set inside the window. Settings can be saved and loaded as JSON from the Setup tab instead.
- Dear PyGui is imported only when `launch()` runs. If it is not installed, the error appears at that line, not at the import at the top.
- On macOS, training with more than one CPU worker uses `multiprocessing`, which re-imports the main module in each worker. Because this file only calls `launch()` inside the `__name__` guard, the workers do not open extra windows. Keep that guard if you edit the file.
- The process exits when the window closes. Unsaved work in the dashboard (a painted map, a renamed tribute, a recording) is gone at that point, so save scenarios and replays from inside the window first.
