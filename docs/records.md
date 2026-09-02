# `records.py`

**Source:** [hunger_games/records.py](../hunger_games/records.py)
**Depends on:** the standard library only (`dataclasses.asdict`, `dataclass`, `field`, `enum.Enum`).
**Used by:** [game.py](game.md) (fills in `Elimination`, `PlayerResult`, `GameResult` and uses `EliminationMethod`), [runner.py](runner.md) (`GameResult`, reads the rows and `telemetry`), [recorder.py](recorder.md) (`Elimination` per frame and `GameResult` at the end).

## Purpose

Chapter 3 of the video catalogues every elimination by day, method and weapon, and every player by kills, placing and training score. These dataclasses are those spreadsheet rows. `Game` fills them in, and `Runner` stacks them into pandas tables and CSV files.

Keeping them as plain dataclasses with `to_row()` methods means the results pickle cleanly across worker processes, convert to dictionaries with one call, and never depend on pandas inside the simulator.

## Concepts you need

**Dataclasses as records.** No behaviour, just named fields with types. `asdict` turns one into a dictionary keyed by field name, which is exactly what `pd.DataFrame([...])` wants.

**Enums stored as strings.** `EliminationMethod.PLAYER.value` is `"player_vs_player"`. The `Elimination.method` field holds the string, not the enum, so the CSV is readable and `groupby` works without conversion.

**Optional fields.** `killer_id: int | None` is `None` for natural and game maker deaths. pandas shows those as `NaN` in the CSV.

**Default factories for lists.** `field(default_factory=list)` gives each `GameResult` its own empty lists.

## Walkthrough

### `EliminationMethod`

```python
class EliminationMethod(Enum):
    PLAYER = "player_vs_player"
    GAMEMAKER = "gamemaker"
    NATURAL = "natural_causes"
```

Chapter 3's three categories. `Game._resolve_attack` uses `PLAYER`; `Game._environment_tick` uses `GAMEMAKER` for the circle and `NATURAL` for dehydration, starvation and untreated wounds.

### `Elimination`

```python
@dataclass
class Elimination:
```

One row of `eliminations.csv`.

| Field | Type | Meaning |
| --- | --- | --- |
| `game_id` | `int` | Which game. |
| `day` | `int` | In-game day, 1 is the first. |
| `tick` | `int` | Exact tick. |
| `victim_id` | `int` | Who died. |
| `victim_name` | `str` | Their name. |
| `victim_district` | `int` | Their district. |
| `victim_training_score` | `int` | Their 1..12 score. |
| `method` | `str` | One of the three category strings. |
| `weapon` | `str` | `"knife"`, `"dehydration"`, `"arena hazard"`, and so on. |
| `killer_id` | `int \| None` | The killer, if a player. |
| `killer_name` | `str \| None` | Their name. |
| `x` | `int` | Column where it happened. |
| `y` | `int` | Row. |
| `placement` | `int` | Victim's placing: 24 is first out, 2 is runner-up. |

`x` and `y` feed the death heatmap in `research/plots.py`.

### `Elimination.to_row`

```python
def to_row(self) -> dict
```

`asdict(self)`.

### `PlayerResult`

```python
@dataclass
class PlayerResult:
```

One row of `players.csv`: how one tribute's game went.

| Field | Type | Meaning |
| --- | --- | --- |
| `game_id` | `int` | Which game. |
| `player_id` | `int` | Who. |
| `name` | `str` | Name. |
| `district` | `int` | District. |
| `sex` | `str` | `"F"` or `"M"`. |
| `training_score` | `int` | 1..12. |
| `survival_score` | `float` | 0..1 aptitude. |
| `brain` | `str` | The brain's `name`. |
| `favor` | `float` | Sponsors' final opinion, 0..1. |
| `gifts_received` | `int` | Parachutes received. |
| `placement` | `int` | Final placing, 1 is the victor. |
| `kills` | `int` | Eliminations credited. |
| `days_survived` | `float` | Days lasted. |
| `cause_of_death` | `str \| None` | How they died, or `None`. |
| `alive_at_end` | `bool` | Still alive when the game stopped. |

### `PlayerResult.to_row`

```python
def to_row(self) -> dict
```

`asdict(self)`.

### `GameResult`

```python
@dataclass
class GameResult:
```

Everything worth keeping from one finished game.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `game_id` | `int` | required | Which game. |
| `seed` | `int` | required | The seed that reproduces it. |
| `days` | `int` | required | Days lasted. |
| `ticks` | `int` | required | Ticks lasted. |
| `winner_id` | `int \| None` | required | The victor, or `None` for a draw. |
| `winner_name` | `str \| None` | required | Their name. |
| `interventions` | `int` | required | Times the game makers stepped in. |
| `eliminations` | `list[Elimination]` | `[]` | Every elimination in order. |
| `players` | `list[PlayerResult]` | `[]` | One per tribute. |
| `gifts` | `list[dict]` | `[]` | Every sponsor gift as a plain dictionary. |
| `telemetry` | `dict \| None` | `None` | Behaviour summary from `BehaviorTelemetry.summary()`, when the runner asked for it. |

`telemetry` is filled by `run_single_game` in [runner.md](runner.md) when `collect_telemetry` is true. It is a plain dictionary of lists and numbers so it survives pickling from worker processes and `json.dumps`. `Runner` merges these across games with `BehaviorTelemetry.merge` and writes `telemetry.json`. The `games.csv` row does not include it.

### `GameResult.elimination_rows`

```python
def elimination_rows(self) -> list[dict]
```

Every elimination as a dictionary. `test_same_seed_reproduces_the_same_game` compares two games with this.

### `GameResult.player_rows`

```python
def player_rows(self) -> list[dict]
```

Every player result as a dictionary.

## How to use it / experiment

Turn one game into tables without the runner:

```python
import pandas as pd
from hunger_games.config import SimulationConfig
from hunger_games.game import Game

result = Game(SimulationConfig(width=60, height=60, seed=11, max_days=10)).run()
eliminations = pd.DataFrame(result.elimination_rows())
players = pd.DataFrame(result.player_rows())
print(eliminations["method"].value_counts())
print(players.sort_values("placement")[["name", "training_score", "placement", "kills"]].head())
print(result.winner_name, result.interventions, result.telemetry)
```

To add a column to a CSV, add a field here and fill it in `Game.result` (for players) or `Game._eliminate` (for eliminations). Because the runner builds tables from `to_row()`, no runner change is needed for those two. The `games.csv` columns are listed explicitly in `Runner.run`, so a new `GameResult` field needs a line there too.

## Gotchas

- `method` is a string, not an `EliminationMethod`. Compare with `"player_vs_player"` or `EliminationMethod.PLAYER.value`.
- `placement` in `Elimination` is the number alive at the moment of death, so it never equals 1. Survivors' placings only appear in `PlayerResult`.
- In a draw every survivor shares the same placing (`len(survivors)`), and `winner_id` is `None`.
- `gifts` are dictionaries, not `SponsorGift` objects; see [sponsors.md](sponsors.md) for the keys.
- `telemetry` is `None` unless telemetry was requested, and it is not written to any CSV, only to `telemetry.json` by the runner.
- `days_survived` for the living is measured at the tick the result was built, which is the final tick.
