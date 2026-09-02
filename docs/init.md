# `__init__.py`

**Source:** [hunger_games/__init__.py](../hunger_games/__init__.py)
**Depends on:** project modules [config.py](config.md), [game.py](game.md), [runner.py](runner.md). No third-party imports of its own.
**Used by:** any code that does `import hunger_games` or `from hunger_games import ...`. Inside the package nothing imports it directly; the modules import each other by full path (`hunger_games.game` and so on). The README examples and the tests use it.

## Purpose

`__init__.py` is what makes the `hunger_games` folder a Python package. When you `import hunger_games`, Python runs this file first. It does two things. It holds the package docstring, which is a reading-order map of the modules. And it re-exports the five most useful names, so beginners can write `from hunger_games import Game, SimulationConfig` instead of remembering which submodule each class lives in.

There is no logic here. Every class it exposes is defined elsewhere. This file is a front door.

## Concepts you need

- **Packages.** A folder with an `__init__.py` is a package. Its modules are the `.py` files inside. `hunger_games.game` means the file `hunger_games/game.py`. Sub-folders with their own `__init__.py` are sub-packages: `hunger_games.brain`, `hunger_games.training`, `hunger_games.research` and `hunger_games.ui`.
- **Re-exporting.** Importing a name into `__init__.py` makes it an attribute of the package. `from hunger_games.game import Game` here means `hunger_games.Game` exists afterwards.
- **`__all__`.** A list of strings naming what `from hunger_games import *` brings in. It also documents what the package considers public.
- **Docstrings.** The triple-quoted string at the top is stored as `hunger_games.__doc__` and shown by `help(hunger_games)`.
- **Import side effects.** Importing a module runs it. Importing `Game` runs `game.py`, which imports the arena, players, brains, sponsors and so on. So `import hunger_games` loads most of the package even if you only wanted the config.

## Walkthrough

### The package docstring

The first line names the project and links the video chapter it is built from. Then comes a "Package map (read them in this order)" table with twenty-one entries:

| Line in the docstring | Module | Doc page |
| --- | --- | --- |
| `config.py      every setting, in one place` | `config.py` | [config.md](config.md) |
| `noise.py       Perlin noise, the height map generator` | `noise.py` | [noise.md](noise.md) |
| `terrain.py     heights -> water / sand / grass / rock` | `terrain.py` | [terrain.md](terrain.md) |
| `districts.py   district names, industries, colours, sexes` | `districts.py` | [districts.md](districts.md) |
| `resources.py   supplies and the two layouts (Cornucopia, ring)` | `resources.py` | [resources.md](resources.md) |
| `arena.py       the world: terrain (generated or painted) + supplies + navigation maps` | `arena.py` | [arena.md](arena.md) |
| `actions.py     the vocabulary of things a body can do` | `actions.py` | [actions.md](actions.md) |
| `perception.py  what a player senses each tick` | `perception.py` | [perception.md](perception.md) |
| `brain/         the decision makers (voting, random, neural) and initializers` | `brain/` | [brain/init.md](brain/init.md) |
| `player.py      the body` | `player.py` | [player.md](player.md) |
| `sponsors.py    parachutes for favoured tributes` | `sponsors.py` | [sponsors.md](sponsors.md) |
| `gamemaker.py   the slow safe circle when the games go quiet (on by default, toggleable)` | `gamemaker.py` | [gamemaker.md](gamemaker.md) |
| `scenario.py    painted map + loot + roster, saved as JSON` | `scenario.py` | [scenario.md](scenario.md) |
| `records.py     the spreadsheet rows` | `records.py` | [records.md](records.md) |
| `game.py        the referee for one game` | `game.py` | [game.md](game.md) |
| `recorder.py    tick-by-tick recordings for replay and GIF export` | `recorder.py` | [recorder.md](recorder.md) |
| `runner.py      play many games, write CSVs` | `runner.py` | [runner.md](runner.md) |
| `renderer.py    watch a game on screen, export GIFs` | `renderer.py` | [renderer.md](renderer.md) |
| `analysis.py    the chapter 3 charts` | `analysis.py` | [analysis.md](analysis.md) |
| `training/      the genetic algorithm and REINFORCE trainers, run folders` | `training/` | [training/init.md](training/init.md) |
| `research/      behaviour telemetry, one PNG per chart, parameter sweeps` | `research/` | [research/init.md](research/init.md) |
| `ui/            the game makers' dashboard (Dear PyGui)` | `ui/` | [ui/init.md](ui/init.md) |

Each module in that list only depends on the ones above it, so reading top to bottom means you never meet a name before it has been explained. `districts.py` sits right after `terrain.py` because it has no imports at all; `sponsors.py` comes after `player.py` because it reads a player's score, district and kills; `scenario.py` comes just before `game.py`, which is the first module that consumes one.

Two things are not in the map:

| Module | What it is | Where it sits in the order |
| --- | --- | --- |
| `research/` | `BehaviorTelemetry`, the one-PNG-per-chart plot functions, and parameter sweeps. `training/` and `ui/` both import it | between `game.py` and `training/` |
| `__main__.py` | The command line | last, see [main.md](main.md) |

The map matches the code: the game maker line says the circle is on by default and toggleable, and `training/` and `research/` are both listed.

```python
import hunger_games
print(hunger_games.__doc__)
```

### The imports

```python
from hunger_games.config import ArenaShape, LayoutName, SimulationConfig
from hunger_games.game import Game
from hunger_games.runner import Runner
```

- `SimulationConfig`: every setting for a game (see [config.md](config.md)).
- `ArenaShape` and `LayoutName`: the enums you pass into the config for `shape` and `layout`.
- `Game`: one complete game, optionally built from a `Scenario` (see [game.md](game.md)).
- `Runner`: many games and four CSV files (see [runner.md](runner.md)).

Importing `Game` pulls in nearly the whole simulation: `arena`, `player`, `brain` (including `mlp` and `initializers`), `gamemaker`, `records`, `scenario`, `sponsors` and `districts`. Through `Runner` it also loads pandas and `research.telemetry`. matplotlib is not loaded until you import `renderer`, `analysis` or `research.plots`. Dear PyGui is only imported when `hunger_games.ui.launch()` is called, never at import time.

### `__all__`

```python
__all__ = ["ArenaShape", "LayoutName", "SimulationConfig", "Game", "Runner"]
```

The five public names. `Renderer`, `make_report`, `Scenario`, `Recorder`, `GeneticTrainer`, `ReinforceTrainer`, `BehaviorTelemetry` and the brain classes are not included. Import them from their own modules.

## How to use it / experiment

### The short imports

```python
from hunger_games import ArenaShape, Game, LayoutName, Runner, SimulationConfig

config = SimulationConfig(seed=1, shape=ArenaShape.ROUND, layout=LayoutName.CORNUCOPIA)
result = Game(config).run()
print(result.winner_name, len(result.gifts))
```

### Everything else by full path

```python
from hunger_games.renderer import Renderer
from hunger_games.analysis import make_report
from hunger_games.brain.voting import VotingBrain
from hunger_games.records import EliminationMethod
from hunger_games.scenario import Scenario, TributeSpec
from hunger_games.recorder import Recorder
from hunger_games.training import GeneticTrainer, ReinforceTrainer, RLConfig, TrainingConfig, save_run
from hunger_games.research import BehaviorTelemetry
from hunger_games.research.experiments import Sweep, SweepConfig
from hunger_games.sponsors import SponsorPool
```

### Check what the package exposes

```python
import hunger_games
print(hunger_games.__all__)
print([name for name in dir(hunger_games) if not name.startswith("_")])
```

The second line also lists the submodules that have been imported as a side effect (`config`, `game`, `runner`, `arena`, `research` and so on), which is a quick way to see how much `import hunger_games` really loads.

### Add your own export

If you write a new brain and want `from hunger_games import MyBrain` to work, add an import line and the name to `__all__` in this file. Keep the import below the ones that are already there so the dependency order is preserved.

## Gotchas

- **Circular imports.** Do not import `hunger_games.renderer` or `hunger_games.analysis` from here without care. They import `hunger_games.game`, which is fine, but adding a module that imports `hunger_games` itself at the top would loop.
- **`import *` is limited on purpose.** It only gives you the five names in `__all__`.
- **The docstring map is a reading order, not a full index.** It omits `__main__.py` (the command line); for one page per file, tests included, use [README.md](README.md).
- **Running the package uses `__main__.py`, not this file.** `python -m hunger_games` executes `__init__.py` first (as an import) and then `__main__.py` (see [main.md](main.md)).
- **No display needed.** Importing the package never touches matplotlib windows or Dear PyGui, so it is safe in scripts, tests and CI.
- **Version pins live in `requirements.txt`.** This file has no `__version__`.
