# `__main__.py`

**Source:** [hunger_games/__main__.py](../hunger_games/__main__.py)
**Depends on:** `argparse` (standard library); project modules [analysis.py](analysis.md), [config.py](config.md), [game.py](game.md), [renderer.py](renderer.md), [runner.py](runner.md), and, only when the `ui` command runs, `hunger_games.ui`
**Used by:** nothing imports it. Python runs it when you type `python -m hunger_games`.

## Purpose

`__main__.py` is the command line for the package. When you run `python -m hunger_games`, Python looks for a file called `__main__.py` inside the package and executes it. This file parses the words after `hunger_games`, turns them into a `SimulationConfig`, and calls the right part of the library.

There are four subcommands. `watch` animates one game in a window or writes it to a GIF or MP4. `simulate` plays many games and writes the four CSV files. `analyze` reads those CSVs and draws the chapter 3 charts. `ui` opens the game makers' dashboard. `watch` and `simulate` share the same set of configuration flags, which map one to one onto fields of `SimulationConfig`.

The file is deliberately thin. All it does is translate flags into objects and hand off. If you want something the flags cannot express, such as a custom brain factory or a painted map, skip the command line and use the classes directly as shown in [game.md](game.md).

## Concepts you need

- **`python -m package`.** The `-m` switch runs a module as a script. For a package it runs `package/__main__.py`. Inside that file `__name__` is `"__main__"`, which is why the last two lines call `main()` only under that check.
- **argparse.** The standard library's command line parser. You build an `ArgumentParser`, call `add_argument` for each flag, then `parse_args()` returns a `Namespace` whose attributes are named after the flags with dashes turned into underscores.
- **Subparsers.** `parser.add_subparsers(dest="command", required=True)` creates a family of subcommands, each with its own flags. `commands.add_parser("watch")` returns a new parser for that verb. After parsing, `args.command` holds the verb that was used.
- **`choices`.** Restricts a flag to a fixed list of strings. Here the lists are built from the enum values in [config.md](config.md), so adding an enum member automatically adds a valid choice.
- **`action=argparse.BooleanOptionalAction`.** One `add_argument("--x", ...)` call creates a pair of flags, `--x` and `--no-x`, that write `True` or `False` into `args.x`. Unlike `store_true`, the pair has a real default, and here the default is read from `SimulationConfig()`, so the command line and the Python API always agree about what "on" means.
- **`action="store_true"`.** A flag with no value and no `--no-` twin. Present means `True`, absent means `False`. Only `--show` on `analyze` uses it.
- **Enums from strings.** `ArenaShape("round")` looks up the enum member whose value is `"round"`. This is how the string from the command line becomes the typed setting the config expects. The reverse, `defaults.shape.value`, turns the enum back into the string argparse needs for a default.
- **Lazy imports.** `from hunger_games.ui import launch` sits inside the `ui` branch, not at the top of the file. Dear PyGui is only loaded when someone actually asks for the dashboard.

## Walkthrough

### `add_config_arguments`

```python
def add_config_arguments(parser: argparse.ArgumentParser) -> None
```

Adds the ten settings that `watch` and `simulate` both accept. Each flag maps to one field of `SimulationConfig`.

The first line builds `defaults = SimulationConfig()` and every flag reads its default from that object (`defaults.shape.value`, `defaults.chaos`, `defaults.gamemaker_enabled` and so on). That way the command line and the Python API can never disagree: change a default in [config.py](config.md) and the flag follows.

| Flag | Type | Default (taken from `SimulationConfig`) | Changes | Effect |
| --- | --- | --- | --- | --- |
| `--shape` | choice of `open_field`, `round` | `open_field` (`defaults.shape.value`) | `shape` | Square 74th-games field, or the round 75th-games arena with corners cut away |
| `--layout` | choice of `cornucopia`, `ring` | `ring` (`defaults.layout.value`) | `layout` | One central pile with podiums around it, or the video's ring redesign with podiums at the edge and weapons in the middle |
| `--chaos` | float | `0.5` (`defaults.chaos`) | `chaos` | The randomness dial: hunt and fight luck, terrain roughness, and how often a brain picks a lower-voted action |
| `--seed` | int | `None` (`defaults.seed`) | `seed` | Fixes the random generator. Same seed and settings replay the same game. `None` draws a fresh seed each run |
| `--size` | int | `120` (`defaults.width`) | `width` and `height` | The arena is always square from the command line |
| `--players` | int | `24` (`defaults.num_players`) | `num_players` | Number of tributes. Districts cycle 1 to 12 two at a time, female then male |
| `--brain` | str | `voting` (`defaults.brain_name`) | `brain_name` | `voting`, `random` or `neural`. Unknown names raise a `KeyError` from the brain registry when the first game is built |
| `--gamemaker` / `--no-gamemaker` | on/off pair | on (`defaults.gamemaker_enabled`) | `gamemaker_enabled` | The slowly shrinking safe circle that starts after a quiet day. On by default because a strict day cutoff alone rarely produces a victor |
| `--sponsors` / `--no-sponsors` | on/off pair | on (`defaults.sponsors_enabled`) | `sponsors_enabled` | Parachute gifts for favoured tributes in need. With `--no-sponsors`, `gifts.csv` is empty |
| `--days` | int | `24` (`defaults.max_days`) | `max_days` | The game ends in a draw once `max_days * ticks_per_day` ticks have passed |

Returns nothing; it mutates the parser you pass in.

The values in the table are the current `SimulationConfig` defaults. If they look different in `--help`, someone changed `config.py`, and `--help` is the one telling the truth. `--help` also prints the on/off pairs as `--gamemaker, --no-gamemaker` and `--sponsors, --no-sponsors`.

### `config_from_arguments`

```python
def config_from_arguments(args: argparse.Namespace) -> SimulationConfig
```

Builds the config field by field: `width=args.size`, `height=args.size`, `shape=ArenaShape(args.shape)`, `layout=LayoutName(args.layout)`, `num_players=args.players`, `chaos=args.chaos`, `seed=args.seed`, `brain_name=args.brain`, `gamemaker_enabled=args.gamemaker`, `sponsors_enabled=args.sponsors`, `max_days=args.days`. Because both on/off pairs already hold a boolean, no `not` is needed. Every other config field (`ticks_per_day`, `vision_radius`, `thirst_days`, `allow_water_podiums`, `career_districts`, `endgame_instinct`, the noise, terrain, neural and reward settings, and so on) keeps its default. Returns the `SimulationConfig`.

```python
import argparse
from hunger_games.__main__ import add_config_arguments, config_from_arguments

parser = argparse.ArgumentParser()
add_config_arguments(parser)
config = config_from_arguments(parser.parse_args(["--shape", "round", "--seed", "3", "--no-gamemaker"]))
print(config.shape, config.seed, config.gamemaker_enabled, config.sponsors_enabled)
# ArenaShape.ROUND 3 False True
```

### `main`

```python
def main() -> None
```

Builds the parser, parses `sys.argv`, and dispatches.

1. `argparse.ArgumentParser(prog="hunger_games", description="Infinite Hunger Games simulator")`.
2. `commands = parser.add_subparsers(dest="command", required=True)`, so running with no verb prints usage and exits.
3. Defines the four subcommands (tables below).
4. `args = parser.parse_args()`.
5. `if args.command == "watch"`: builds `Game(config_from_arguments(args))`, wraps it in `Renderer(game, ticks_per_frame=args.speed)`, then calls `renderer.save(args.save)` if `--save` was given, otherwise `renderer.show()`.
6. `elif args.command == "simulate"`: builds `Runner(config, num_games=args.games, workers=args.workers, output_dir=args.output)`, calls `runner.run()`, and prints how many elimination rows, player rows and games were written and where. The gifts table is written too but not mentioned in the message.
7. `elif args.command == "analyze"`: calls `make_report(args.output, show=args.show)` and prints the path of the PNG.
8. `elif args.command == "ui"`: imports `launch` from `hunger_games.ui` and calls it. This blocks until the dashboard window is closed.

#### `watch` subcommand

Animate a single game. Accepts all the shared flags plus:

| Flag | Type | Default | Effect |
| --- | --- | --- | --- |
| `--speed` | int | `1` | Simulation ticks per animation frame. Higher is faster and skips intermediate positions |
| `--save` | str | `None` | A file path. If set, writes a GIF (or MP4 when the path ends in `.mp4` and ffmpeg is installed) instead of opening a window |

```bash
python -m hunger_games watch
python -m hunger_games watch --shape round --layout cornucopia --seed 5
python -m hunger_games watch --save output/game.gif --speed 3
python -m hunger_games watch --no-gamemaker --no-sponsors
```

#### `simulate` subcommand

Play many games and write CSVs. Accepts all the shared flags plus:

| Flag | Type | Default | Effect |
| --- | --- | --- | --- |
| `--games` | int | `100` | Number of games. Game `i` uses seed `(seed + i) % 2**31` |
| `--workers` | int | `1` | CPU cores. Values above 1 use a `ProcessPoolExecutor` (see [runner.md](runner.md)) |
| `--output` | str | `output` | Folder for `eliminations.csv`, `players.csv`, `games.csv` and `gifts.csv`. Created if missing |

```bash
python -m hunger_games simulate --games 500 --workers 4
python -m hunger_games simulate --games 200 --layout cornucopia --output output/cornucopia
python -m hunger_games simulate --games 200 --no-gamemaker --output output/no_gm
python -m hunger_games simulate --games 200 --no-sponsors --output output/no_sponsors
```

#### `analyze` subcommand

Draw the chapter 3 charts from saved CSVs. Does not accept the shared config flags, because it never runs a game.

| Flag | Type | Default | Effect |
| --- | --- | --- | --- |
| `--output` | str | `output` | Folder holding the CSV files. `report.png` is written there too |
| `--show` | flag | off | Also open the chart in a window |

```bash
python -m hunger_games analyze
python -m hunger_games analyze --output output/cornucopia --show
```

#### `ui` subcommand

Open the game makers' dashboard. It takes no flags at all: every setting is edited inside the window, and the dashboard starts from `SimulationConfig()` defaults.

```bash
python -m hunger_games ui
```

Requires Dear PyGui and a display. The import happens only inside this branch, so `watch`, `simulate` and `analyze` work on machines without Dear PyGui installed.

### The `__name__` guard

```python
if __name__ == "__main__":
    main()
```

Runs `main()` only when the file is executed, not when it is imported. It also matters for multiprocessing: with `--workers` above 1, each spawned worker re-imports this module, and the guard stops the workers from starting their own runs.

## How to use it / experiment

### A full experiment from the shell

```bash
# 1. Play two batches with different layouts, same seed.
MPLBACKEND=Agg python -m hunger_games simulate --games 200 --workers 4 --seed 1 --layout cornucopia --output output/cornucopia
MPLBACKEND=Agg python -m hunger_games simulate --games 200 --workers 4 --seed 1 --layout ring --output output/ring

# 2. Draw the charts for each.
MPLBACKEND=Agg python -m hunger_games analyze --output output/cornucopia
MPLBACKEND=Agg python -m hunger_games analyze --output output/ring
```

Then compare `output/cornucopia/report.png` with `output/ring/report.png`, or load both CSV sets in pandas as shown in [analysis.md](analysis.md).

### Game makers on versus off

```bash
python -m hunger_games simulate --games 100 --seed 1 --output output/with_gm
python -m hunger_games simulate --games 100 --seed 1 --no-gamemaker --output output/no_gm
```

Compare the `interventions` column of `games.csv` and how many games end with a `winner_id`.

### Sponsors on versus off

```bash
python -m hunger_games simulate --games 100 --seed 1 --output output/sponsors
python -m hunger_games simulate --games 100 --seed 1 --no-sponsors --output output/no_sponsors
```

Then compare the `natural_causes` share of `method` in the two `eliminations.csv` files, and count rows in `output/sponsors/gifts.csv`.

### Test the chaos dial

```bash
python -m hunger_games watch --chaos 0 --seed 5     # replays identically every time
python -m hunger_games watch --chaos 1 --seed 5     # same seed, much wilder
```

### Smaller and faster while developing

```bash
python -m hunger_games simulate --games 10 --size 60 --days 5 --output output/quick
```

### Reuse the parser in your own script

```python
import argparse
from hunger_games import Game
from hunger_games.__main__ import add_config_arguments, config_from_arguments

parser = argparse.ArgumentParser()
add_config_arguments(parser)
parser.add_argument("--repeat", type=int, default=3)
args = parser.parse_args()
for game_id in range(args.repeat):
    result = Game(config_from_arguments(args), game_id=game_id).run()
    print(game_id, result.winner_name, len(result.gifts), "gifts")
```

## Gotchas

- **A subcommand is required.** `python -m hunger_games` alone prints usage and exits with an error.
- **Shared flags go after the verb.** `python -m hunger_games --seed 1 watch` fails; write `python -m hunger_games watch --seed 1`.
- **The game makers are on by default.** Pass `--no-gamemaker` for a quiet arena. `--gamemaker` is accepted too and simply says "on" explicitly. Older docs that describe `--gamemaker` as the only way to switch them on are out of date.
- **`--sponsors` and `--no-sponsors` are a pair.** Either may be given; the last one on the line wins, as with any argparse flag.
- **`BooleanOptionalAction` needs Python 3.9 or newer.** The project already requires a newer Python for its type hints, so this is not a new constraint.
- **`analyze` and `ui` take no config flags.** Passing `--seed` to either is an error.
- **Defaults are read from `SimulationConfig()` at parse time.** Building that object is cheap, but it means `add_config_arguments` constructs a config every time it is called. Do not add heavy work to `SimulationConfig.__init__`.
- **Only ten fields are exposed.** `ticks_per_day`, `allow_water_podiums`, `start_thirst_min`, `career_districts`, `endgame_instinct` and the nested noise, terrain, neural and reward settings keep their defaults. Use Python or the dashboard for those.
- **`--size` sets both width and height.** For a non-square arena build `SimulationConfig(width=..., height=...)` in Python.
- **`--brain` is not validated by argparse.** A typo like `--brain votng` only fails when the first game is built, with a `KeyError` listing the valid names.
- **`--save` picks the writer by extension.** `.mp4` needs ffmpeg; anything else uses Pillow and produces a GIF regardless of the extension you typed.
- **`watch` without `--save` needs a display.** Under SSH or in CI, use `--save` and set `MPLBACKEND=Agg`.
- **`ui` needs Dear PyGui.** It is not in the import list at the top of the file, so a missing install only errors when you run `ui`.
- **`--workers` above 1 spawns processes.** Keep your own scripts under an `if __name__ == "__main__":` guard on macOS (see [runner.md](runner.md)).
- **Output folders are overwritten.** The four CSVs and `report.png` replace any earlier files with the same names.
