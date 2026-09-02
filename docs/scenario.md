# `scenario.py`

**Source:** [hunger_games/scenario.py](../hunger_games/scenario.py)
**Depends on:** only the standard library (`dataclasses`, `json`, `pathlib`). It does not import anything from the package, so it is pure data.
**Used by:** [game.py](game.md) (`Scenario`, `TributeSpec`), `recorder.py` (stores the scenario with a replay), `training/genetic.py` (trains on a painted map), the dashboard (`ui/session.py` builds and saves one; `ui/app.py` has the Save / Load buttons), and `tests/test_scenario.py`.

## Purpose

A `Scenario` is everything a game maker can customise before the games begin. It is the saved state of the dashboard: a painted map, hand-placed loot, and a roster of tributes with names, districts, training scores, brains, starting gear and podiums. It is plain data with no behaviour of its own beyond saving and loading as JSON. Anything left as `None` falls back to the normal generated behaviour, so an empty `Scenario()` gives exactly the same game as no scenario at all.

This is what the dashboard's **Save scenario** button writes and **Load scenario** reads. `ui/session.py` keeps a live `Scenario` while you paint and edit, then builds a fresh one with `_scenario_for_game()` (the painted map plus the edited roster and loot) each time you press play. The recorder stores that scenario inside a replay so a saved game can be reopened with the same map and names.

`Game` consumes a scenario in four places, all in `Game.__init__` and its helpers (see [game.md](game.md)):

1. **Map.** If `scenario.terrain` is not `None`, it becomes a numpy `int8` grid and is passed to `Arena(config, rng, terrain=painted)`. The arena adopts it instead of generating one (see [arena.md](arena.md)).
2. **Loot.** The layout's own loot is scattered only if `scenario.use_layout_loot` is `True`. Then every `LootSpec` in `scenario.loot` is placed on top with `arena.resources.place`, skipping cells that are not walkable.
3. **Roster.** If `scenario.tributes` is a non-empty list, `Game._create_players` builds one `Player` per `TributeSpec` instead of rolling random tributes. Each spec's `weapon_quality`, `food`, `medicine` and `favor_bonus` are copied straight onto the player, and `start_thirst` / `start_hunger` / `start_health` override the config's random range when they are not `None`.
4. **Podiums.** After the layout places everyone, `Game._place_players` moves any tribute whose spec has a `podium` to that cell, snapped with `arena.snap_to_podium`.

The neural trainer uses the map part alone: `GeneticTrainer` is given `Scenario(terrain=...)` so every evaluation game runs on the painted arena while the roster is generated.

## Concepts you need

**Dataclasses with defaults.** Fields without a default must come first (`player_id`, `name`, ...); fields with defaults follow. `field(default_factory=list)` gives each scenario its own empty loot list rather than one shared list.

**`Optional` values.** `float | None = None` means "unset". `Game` checks `is not None` and only then uses the value. That is how "leave it to the config" is expressed.

**`dataclasses.asdict`.** Recursively turns a dataclass, and any dataclasses nested inside lists, into plain dictionaries and lists. `Scenario.to_dict` is one line because of it.

**JSON.** `json.dumps` writes a dictionary as text, `json.loads` reads it back. JSON has no tuples, so `(5, 5)` becomes `[5, 5]` and must be converted back on load. `None` becomes `null`.

**`**` unpacking.** `LootSpec(**item)` calls the constructor with each dictionary key as a keyword argument. It fails with `TypeError` if the dictionary has a key the dataclass does not know.

**`pathlib.Path`.** `Path(path).write_text(...)` and `.read_text()` handle opening and closing the file. Accepting `str | Path` means callers can pass either.

**`classmethod` constructors.** `from_dict` and `load` are alternative ways to build a `Scenario`, called on the class.

## Walkthrough

### `TributeSpec`

```python
@dataclass
class TributeSpec:
```

One tribute as edited in the dashboard.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `player_id` | `int` | required | Unique id, `0` upward. |
| `name` | `str` | required | Display name. |
| `district` | `int` | required | District, `1` to `12`. |
| `sex` | `str` | required | `"F"` or `"M"`. |
| `training_score` | `int` | required | The `1` to `12` training score. |
| `survival_score` | `float` | required | The `0.05` to `0.95` survival aptitude. |
| `brain_name` | `str` | `"voting"` | `"voting"`, `"random"` or `"neural"`. |
| `genome` | `list[float] \| None` | `None` | A saved genome for that brain, or `None` for a fresh one. |
| `weapon_quality` | `float` | `0.0` | A weapon granted before the games (`0.0` = none). |
| `food` | `int` | `0` | Rations granted before the games. |
| `medicine` | `int` | `0` | Medkits granted before the games. |
| `favor_bonus` | `float` | `0.0` | Extra sponsor favour, `0.0` to `1.0` (see [sponsors.md](sponsors.md)). |
| `start_thirst` | `float \| None` | `None` | Starting thirst bar; `None` means the config's random range. |
| `start_hunger` | `float \| None` | `None` | Starting hunger bar. |
| `start_health` | `float \| None` | `None` | Starting health bar. |
| `podium` | `tuple[int, int] \| None` | `None` | Podium `(x, y)`, or `None` to use the layout's podium. |

`Game._generated_spec` builds one of these for every slot when there is no roster, so the same class describes both generated and hand-edited tributes. `Game._start_value(minimum, override)` returns `clip(override, 0.01, 1.0)` when an override is given, otherwise a random draw between the config's `start_*_min` and `1.0`.

### `LootSpec`

```python
@dataclass
class LootSpec:
```

One stack of supplies placed by hand.

| Field | Type | Meaning |
| --- | --- | --- |
| `x` | `int` | Column. |
| `y` | `int` | Row. |
| `kind` | `int` | `1` food, `2` weapon, `3` medicine (the `ResourceKind` values from [resources.md](resources.md)). |
| `quantity` | `int` | How many. |
| `quality` | `float` | How good, `0.0` to `1.0`. |

All five fields are required. `kind` is stored as a plain `int` so the file stays JSON-friendly; `Game` wraps it back with `ResourceKind(loot.kind)`.

### `Scenario`

```python
@dataclass
class Scenario:
```

A complete custom setup. Every field is optional.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `terrain` | `list[list[int]] \| None` | `None` | Rows of `TerrainType` integers, or `None` to generate a map. |
| `use_layout_loot` | `bool` | `True` | Whether the config's layout still scatters its own loot. |
| `loot` | `list[LootSpec]` | `[]` | Hand-placed stacks. |
| `tributes` | `list[TributeSpec] \| None` | `None` | The roster, or `None` to generate one. |
| `title` | `str` | `"Untitled scenario"` | A free-text label for the dashboard. |

`terrain` is a list of lists rather than a numpy array so it serialises directly. `Game` converts it with `np.array(scenario.terrain, dtype=np.int8)`; the dashboard converts the other way with `.tolist()`.

### `Scenario.to_dict`

```python
def to_dict(self) -> dict
```

`asdict(self)`. Nested `LootSpec` and `TributeSpec` objects become dictionaries automatically.

### `Scenario.from_dict`

```python
@classmethod
def from_dict(cls, data: dict) -> "Scenario"
```

Rebuilds from a dictionary. Loot entries become `LootSpec(**item)`. If `tributes` is present and not `None`, each entry is copied, its `podium` list is turned back into a tuple, and a `TributeSpec` is built. Missing keys use the defaults (`use_layout_loot` `True`, `title` `"Untitled scenario"`, `terrain` `None`). Unknown keys inside a loot or tribute entry raise `TypeError`.

### `Scenario.save`

```python
def save(self, path: str | Path) -> None
```

`Path(path).write_text(json.dumps(self.to_dict()))`. No indentation, so a 120 by 120 map is one long line.

### `Scenario.load`

```python
@classmethod
def load(cls, path: str | Path) -> "Scenario"
```

Reads the file, parses the JSON and calls `from_dict`.

### `Scenario.tribute`

```python
def tribute(self, player_id: int) -> TributeSpec | None
```

Linear search of `self.tributes or []` for a matching `player_id`. Returns `None` if the roster is empty or the id is absent. The dashboard's roster editor uses it to find the selected row.

## How to use it / experiment

Build a small custom game entirely in code:

```python
import numpy as np
from hunger_games.config import SimulationConfig
from hunger_games.game import Game
from hunger_games.resources import ResourceKind
from hunger_games.scenario import LootSpec, Scenario, TributeSpec
from hunger_games.terrain import TerrainType

size = 40
terrain = np.full((size, size), int(TerrainType.GRASS), dtype=np.int8)
terrain[18:22, :] = int(TerrainType.WATER)          # a river across the middle

roster = [
    TributeSpec(0, "Katniss", 12, "F", 11, 0.8, "voting", podium=(5, 5), weapon_quality=0.95, food=3),
    TributeSpec(1, "Peeta", 12, "M", 8, 0.5, "voting", podium=(30, 30), start_health=0.4),
    TributeSpec(2, "Cato", 2, "M", 10, 0.7, "voting", podium=(5, 30), favor_bonus=0.2),
]
scenario = Scenario(
    terrain=terrain.tolist(),
    use_layout_loot=False,
    loot=[LootSpec(20, 10, int(ResourceKind.MEDICINE), 2, 0.9)],
    tributes=roster,
    title="River test",
)
game = Game(SimulationConfig(seed=1, width=size, height=size), scenario=scenario)
print([(p.name, p.position, p.weapon_quality) for p in game.players])
result = game.run()
```

Save, load and check the round trip:

```python
scenario.save("river.json")
again = Scenario.load("river.json")
assert again.tributes[0].podium == (5, 5)
assert again.loot == scenario.loot
```

Use only a painted map and let everything else generate:

```python
game = Game(SimulationConfig(seed=1, width=size, height=size), scenario=Scenario(terrain=terrain.tolist()))
```

Edit a roster entry before playing:

```python
spec = scenario.tribute(1)
spec.brain_name = "random"
spec.medicine = 2
```

## Gotchas

- **The map size must match the config.** `Scenario.terrain` carries no width or height. `Game` passes `config.width` and `config.height` to the arena, so a 40 by 40 painted map with a default 120 by 120 config will fail. The dashboard keeps them in sync for you.
- **A painted map is never carved round.** `config.shape` is ignored when `terrain` is given. Paint `VOID` (value `0`) cells yourself for a round arena.
- **`use_layout_loot=False` with an empty `loot` list means an arena with nothing in it.** Tributes can still drink and hunt, but there are no weapons.
- **Hand-placed loot on a non-walkable cell is silently dropped.** `Game` checks `arena.is_walkable` and skips it.
- **Hand-placed loot overwrites layout loot.** `ResourceGrid.place` replaces the cell.
- **`num_players` is ignored when a roster is given.** `Game` builds exactly `len(scenario.tributes)` players. The dashboard updates `config.num_players` to match when you add or remove a tribute.
- **`player_id` values should be unique and match the `podium` lookup.** `Game._place_players` looks podiums up by `player_id`; duplicates make one tribute win.
- **`podium` comes back as a list from JSON only if you bypass `from_dict`.** `from_dict` converts it; hand-built dictionaries passed to `TributeSpec(**item)` do not get that treatment.
- **`start_*` of `0` is not "unset".** The dashboard maps a slider at `0` to `None`; in code write `None` explicitly. A literal `0.0` is clipped to `0.01`, a nearly dead tribute.
