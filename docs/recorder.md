# `recorder.py`

**Source:** [hunger_games/recorder.py](../hunger_games/recorder.py)
**Depends on:** `dataclasses`, `pathlib.Path`, `pickle` (standard library), `numpy`; project modules [config.md](config.md) (`SimulationConfig`), [game.md](game.md) (`Game`), [records.md](records.md) (`Elimination`, `GameResult`), `scenario.py` (`Scenario`), `sponsors.py` (`SponsorGift`).
**Used by:** [renderer.md](renderer.md) (`Renderer` drives a `Recorder`; `export_recording_gif` draws a `Recording`), `ui/session.py` (the dashboard records every game, saves and loads `.replay` files), `tests/test_recorder_training.py`.

## Purpose

A game runs in a fraction of a second. To watch it, scrub back and forth, click a tribute mid-game, or export a GIF, every tick has to be saved. This file does that. `Recorder` watches a `Game` and, after each tick, copies the interesting state into a `Frame`. A `Recording` is the list of frames plus everything needed to redraw them: the config, the scenario, the terrain and height grids, and a roster of fixed facts about each tribute. Recordings save to disk with `pickle`.

The recorder is deliberately separate from the referee. `Game` knows nothing about frames; it just runs. The recorder reads the game's public attributes after each step. That means any code that can call `game.step()` can be recorded, and the renderer, the dashboard and the tests all get frames of the same shape.

## Concepts you need

**Snapshots versus references.** A `PlayerSnapshot` copies numbers out of a `Player`. It does not hold the `Player`. Later changes to the game do not change old frames. The grids are copied with `.copy()` for the same reason.

**Dataclasses with `field(default_factory=list)`.** Each frame gets its own fresh `eliminations` and `gifts` lists.

**Slicing to find new items.** `game.eliminations[self._eliminations_seen:]` is every elimination added since the last capture. Remembering the count is cheaper than comparing lists.

**Class methods.** `Recording.load` is a `@classmethod`: you call it on the class, `Recording.load(path)`, and it returns a new instance.

**pickle.** `pickle.dumps(obj)` turns almost any Python object into bytes and `pickle.loads` turns them back. It is easy and complete, but loading runs code embedded in the file. Never load a pickle you did not make.

**`Path.write_bytes` / `read_bytes`.** One-line file I/O from `pathlib`.

## Walkthrough

### `class PlayerSnapshot` (dataclass)

```python
PlayerSnapshot(player_id: int, x: int, y: int, alive: bool, thirst: float, hunger: float,
               health: float, food: int, medicine: int, weapon_quality: float, kills: int,
               favor: float, last_action: str)
```

One tribute's state at one tick. Every field is a copy of the matching `Player` attribute, except `last_action`, which is the action kind's string value (`"move"`, `"attack"`, and so on) or `""` if the tribute has not acted yet.

### `class Frame` (dataclass)

```python
Frame(tick: int, day: int, players: list[PlayerSnapshot], resource_kind: np.ndarray,
      safe_radius: float, circle_visible: bool,
      eliminations: list[Elimination] = [], gifts: list[SponsorGift] = [])
```

| Field | Meaning |
| --- | --- |
| `tick` | `game.tick` at capture time. |
| `day` | `game.day_number`. |
| `players` | One snapshot per tribute, alive or dead, in `game.players` order. |
| `resource_kind` | A copy of `arena.resources.kind`, one byte per cell. Enough to draw the supplies. |
| `safe_radius` | `gamemaker.safe_radius`. |
| `circle_visible` | `gamemaker.is_active`. |
| `eliminations` | Rows that appeared since the previous frame. |
| `gifts` | Parachutes that appeared since the previous frame. |

### `class RosterEntry` (dataclass)

```python
RosterEntry(player_id: int, name: str, district: int, sex: str, training_score: int,
            survival_score: float, brain: str)
```

The facts about a tribute that never change during a game. Stored once per recording rather than once per frame. The renderer looks up `district` for the colour and `sex` for the marker shape.

### `class Recording` (dataclass)

```python
Recording(config: SimulationConfig, scenario: Scenario | None, terrain: np.ndarray,
          heights: np.ndarray, roster: list[RosterEntry],
          frames: list[Frame] = [], result: GameResult | None = None)
```

Everything needed to replay a game without the `Game` object. `frames[0]` is the state before anyone moves. `result` is `None` until the game ends.

#### `length` (property) `-> int`

`len(self.frames)`.

#### `save(self, path: str | Path) -> None`

`Path(path).write_bytes(pickle.dumps(self))`. The dashboard uses the `.replay` extension by convention, but nothing enforces it.

#### `load(cls, path: str | Path) -> Recording` (classmethod)

`pickle.loads(Path(path).read_bytes())`. The docstring carries the warning: only open replay files you made yourself, because pickle runs code. A `.replay` from a stranger can do anything your Python process can do.

### `class Recorder`

#### `__init__(self, game: Game) -> None`

Builds a `Recording` from the game's config, scenario, copies of `arena.terrain` and `arena.heights`, and a `RosterEntry` for every player. Sets `_eliminations_seen` and `_gifts_seen` to 0. Then calls `capture()` once, so frame 0 is the starting positions.

| Attribute | Meaning |
| --- | --- |
| `game` | The game being watched. |
| `recording` | The `Recording` being built. |
| `_eliminations_seen` | How many of `game.eliminations` are already in a frame. |
| `_gifts_seen` | Same for `game.gifts`. |

#### `capture(self) -> Frame`

Snapshots the game right now. Takes the eliminations and gifts added since the last capture, updates the seen counts, builds a `Frame` with a `PlayerSnapshot` per player and a copy of the supply grid, appends it to `recording.frames`, and returns it. Calling it twice in a row without stepping gives two identical frames with empty event lists in the second.

#### `step(self) -> Frame`

`game.step()`, then `capture()`. If the game is now over, `recording.result = game.result()`. Returns the new frame.

#### `record_all(self) -> Recording`

`step()` until `game.is_over`, then makes sure `result` is filled in even for a game that was over at tick 0. Returns the recording.

```python
from hunger_games.config import SimulationConfig
from hunger_games.game import Game
from hunger_games.recorder import Recorder, Recording

game = Game(SimulationConfig(seed=7, width=50, height=50, max_days=4))
recording = Recorder(game).record_all()
print(recording.length == game.tick + 1)          # True: frame 0 plus one per tick
print(recording.result.winner_name)
recording.save("game7.replay")
same = Recording.load("game7.replay")
print(same.frames[-1].players[0].x == recording.frames[-1].players[0].x)   # True
```

## How to use it / experiment

**Scrub through a game.** Once recorded, any tick is just an index.

```python
frame = recording.frames[100]
alive = [p for p in frame.players if p.alive]
print(frame.day, len(alive), frame.safe_radius)
for p in alive[:3]:
    entry = recording.roster[p.player_id]
    print(entry.name, entry.sex, p.health, p.last_action)
```

**Find every event.** Eliminations and gifts are attached to the frame they appeared in, so a single pass finds them all with their surrounding state.

```python
for frame in recording.frames:
    for e in frame.eliminations:
        print(f"frame {frame.tick}: {e.victim_name} died of {e.weapon}")
    for g in frame.gifts:
        print(f"frame {frame.tick}: {g.player_name} received {g.kind}")
```

**Record while stepping by hand.** `Recorder.step()` is a drop-in replacement for `game.step()`. The dashboard's play button does exactly this.

**Export a GIF.** Hand the recording to `export_recording_gif` from [renderer.md](renderer.md).

**Add a field to the snapshot.** Append it to `PlayerSnapshot`, add the matching argument in `capture()`, and old `.replay` files will fail to load with a `TypeError` because the pickled objects have a different shape. Recordings are not a stable file format.

## Gotchas

- Frame numbers are one ahead of event ticks. `Recorder.step()` captures after `game.tick` has advanced, so an elimination whose row says `tick=30` sits in the frame whose `tick` is 31. Frame `n` is the state after `n` ticks have been played.
- Frame 0 is captured in `__init__`, before any step. A recording of a game that ends at tick 100 has 101 frames.
- Every frame copies the supply grid. For a 120 by 120 arena that is 14,400 bytes per frame, so a long game costs several megabytes in memory and on disk. Downsample with `export_recording_gif(step=...)` for GIFs, and consider fewer `max_days` for experiments.
- `Recording.load` is a pickle load. Treat replay files like executables: only open your own.
- Pickles are tied to the class definitions. Renaming a field in any of these dataclasses, or in `SimulationConfig`, `Scenario`, `Elimination` or `SponsorGift`, breaks old replays.
- `capture()` copies numbers, not `Player` objects. `frame.players[0]` has no `brain`, no `name` and no `sex`; those live in `recording.roster[0]`.
- `PlayerSnapshot.last_action` is `""` on frame 0 and for any tribute whose brain has not been asked yet. It is the action *chosen*, which may have failed (a blocked move still reads `"move"`).
- `record_all()` on a game that is already over adds no frames but still fills in `result`.
- `Recorder` and the `Renderer` share the same recording object. Drawing does not copy frames.
