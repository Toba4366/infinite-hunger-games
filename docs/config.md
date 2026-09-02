# `config.py`

**Source:** [hunger_games/config.py](../hunger_games/config.py)
**Depends on:** the standard library only (`dataclasses.dataclass`, `dataclasses.field`, `enum.Enum`). Inside `to_dict_raw` and `to_dict` it imports `dataclasses.MISSING`, `fields` and `asdict` lazily.
**Used by:** almost everything. [arena.py](arena.md) (`ArenaShape`, `SimulationConfig`), [terrain.py](terrain.md) (`TerrainConfig`), [resources.py](resources.md) (`LayoutName`), [sponsors.py](sponsors.md), [gamemaker.py](gamemaker.md), [game.py](game.md), [runner.py](runner.md), [recorder.py](recorder.md), [init.md](init.md), [main.md](main.md), [brain/init.md](brain/init.md) and [brain/neural.md](brain/neural.md) (`NeuralConfig`), `research/experiments.py`, [training/genetic.md](training/genetic.md) (`NeuralConfig`, `SimulationConfig`), `training/reinforce.py` (reads `config.reward`), `ui/app.py`, `ui/session.py`, `ui/painter.py`, the three `experiments/run_*.py` scripts, and every test file.

## Purpose

This file is the control panel. It does nothing on its own. It only declares settings, with defaults, that every other module reads. If you want the games to behave differently, change a value here (or pass a different value when you build a `SimulationConfig`) instead of editing the simulation code.

There are two enums (fixed multiple-choice settings), four small nested dataclasses (noise, terrain, neural network, reward function) and one master dataclass, `SimulationConfig`, that holds everything. The dashboard in `hunger_games/ui` edits these same objects, and the command line in [main.md](main.md) reads its defaults from them so nothing drifts.

## Concepts you need

**Dataclasses.** `@dataclass` looks at the annotated fields of a class and writes `__init__`, `__repr__` and `__eq__` for you. `SimulationConfig(width=60, seed=3)` works because the decorator generated an `__init__` with one keyword argument per field.

**Default factories.** A mutable default (a nested dataclass, a list) must not be shared between instances. `field(default_factory=NoiseConfig)` builds a fresh `NoiseConfig` for every `SimulationConfig`.

**Enums.** `ArenaShape.ROUND` is a named constant with a `.value` of `"round"`. `ArenaShape("round")` goes the other way. JSON and the command line use the string; the code uses the enum.

**Properties.** `@property` turns a method into something you read like an attribute: `config.ticks_per_game`, not `config.ticks_per_game()`. The three properties here derive numbers from fields so nobody has to repeat the arithmetic.

**Pickling.** The dashboard and the multi-core runner send configs to worker processes by pickling them. A config saved by an older version of this file may be missing fields that were added later. `to_dict_raw` guards against that.

## Walkthrough

### `ArenaShape`

```python
class ArenaShape(Enum):
    OPEN_FIELD = "open_field"
    ROUND = "round"
```

The outline of the arena. `OPEN_FIELD` is the square forest of the 74th games. `ROUND` carves a circle out of the grid, like the 75th games' clock arena.

### `LayoutName`

```python
class LayoutName(Enum):
    CORNUCOPIA = "cornucopia"
    RING = "ring"
```

How supplies are scattered. `CORNUCOPIA` piles everything in the centre. `RING` is the video's redesign: cheap supplies at the edge, weapons in the centre.

### `NoiseConfig`

```python
@dataclass
class NoiseConfig:
```

| Field | Default | Meaning |
| --- | --- | --- |
| `scale` | `40.0` | Cells per hill. Bigger means smoother, wider hills. |
| `octaves` | `5` | Layers of detail stacked together. |
| `persistence` | `0.5` | How much quieter each extra layer is. |
| `lacunarity` | `2.0` | How much finer each extra layer is. |

Read by [noise.md](noise.md) through [arena.md](arena.md).

### `TerrainConfig`

```python
@dataclass
class TerrainConfig:
```

| Field | Default | Meaning |
| --- | --- | --- |
| `water_threshold` | `0.25` | Heights below this are water. |
| `sand_size` | `0.10` | Height band above the water that is sand. |
| `grass_size` | `0.50` | Height band above the sand that is grass. |

Whatever height is left at the top becomes rock, so rock needs no setting. Thresholds are relative: shrink `sand_size` and grass starts lower. Read by [terrain.md](terrain.md).

### `NeuralConfig`

```python
@dataclass
class NeuralConfig:
```

| Field | Default | Meaning |
| --- | --- | --- |
| `hidden_layers` | `(16,)` | Width of each hidden layer in order. `(32, 16)` is two layers. |
| `activation` | `"tanh"` | Squashing function: `tanh`, `relu`, `leaky_relu`, `sigmoid` or `selu`. |
| `initializer` | `"xavier_uniform"` | How starting weights are drawn (see [brain/initializers.md](brain/initializers.md)). |
| `init_scale` | `0.05` | Constant or spread for the constant, uniform and normal initializers. |
| `sparsity` | `0.1` | Fraction of non-zero weights for the sparse initializer. |

Read by [brain/neural.md](brain/neural.md).

### `RewardConfig`

```python
@dataclass
class RewardConfig:
```

The per-tick score a reinforcement-learning tribute receives. Only `training/reinforce.py` reads it. The genetic algorithm ignores it and scores whole games by placing.

| Field | Default | Meaning |
| --- | --- | --- |
| `survive_tick` | `0.01` | Reward for every tick survived. |
| `win` | `5.0` | Bonus for winning the games. |
| `death` | `-3.0` | Penalty for dying. |
| `kill` | `1.0` | Bonus per elimination. |
| `damage_taken` | `-2.0` | Penalty per point of health lost. A full bar lost costs this much. |
| `need_gain` | `0.5` | Bonus per point of thirst or hunger restored while that bar was below half. |
| `placement` | `2.0` | End-of-game bonus scaled by placing: this much for first, nothing for last. |
| `discount` | `0.98` | How much a reward one tick later is worth compared to now. |

Design reasoning: the small `survive_tick` gives a constant pull toward staying alive, the large `win` and `death` values anchor the ends, and `need_gain` only pays out below half a bar so drinking when already full is not rewarded.

### `SimulationConfig`

```python
@dataclass
class SimulationConfig:
```

The master object handed to `Game`, `Runner`, `Renderer` and the dashboard. Every field with its default:

| Field | Default | Meaning |
| --- | --- | --- |
| `width` | `120` | Arena width in cells. |
| `height` | `120` | Arena height in cells. |
| `shape` | `ArenaShape.OPEN_FIELD` | Arena outline. |
| `layout` | `LayoutName.RING` | Supply layout. |
| `allow_water_podiums` | `True` | Whether podiums may stand in water. |
| `num_players` | `24` | Tributes entering the arena (two per district). |
| `brain_name` | `"voting"` | Default brain: `"voting"`, `"random"` or `"neural"`. |
| `start_thirst_min` | `1.0` | Starting thirst is drawn between this and 1.0. |
| `start_hunger_min` | `1.0` | Same for hunger. |
| `start_health_min` | `1.0` | Same for health. |
| `career_districts` | `(1, 2, 4)` | Districts whose tributes attract sponsors. |
| `chaos` | `0.5` | Randomness dial, 0.0 deterministic to 1.0 chaotic. |
| `seed` | `None` | Random seed. `None` picks a fresh one each game. |
| `ticks_per_day` | `24` | Simulation steps per in-game day. |
| `max_days` | `24` | Strict cutoff. Longer games end in a draw. |
| `vision_radius` | `8` | Cells a player can see other players and supplies. |
| `landmark_radius` | `30` | Cells a player can spot a lake or a meadow. |
| `thirst_days` | `3.0` | Days from a full thirst bar to death. |
| `hunger_days` | `7.0` | Days from a full hunger bar to death. |
| `sponsors_enabled` | `True` | Whether sponsors may send gifts. |
| `sponsor_gift_chance` | `0.5` | Daily chance a fully favoured tribute in need gets a gift. |
| `gamemaker_enabled` | `True` | Whether the safe circle may shrink when the games go quiet. |
| `quiet_days_before_intervention` | `1.0` | Days without an elimination before the game makers step in. |
| `intervention_days` | `6.0` | Days of shrinking for the circle to close from the edge to the centre. |
| `cannon_and_sky` | `True` | Tributes know how many remain and how strong they are. |
| `endgame_instinct` | `False` | Bold tributes head for the centre once fewer than half remain. |
| `noise` | `NoiseConfig()` | Perlin noise settings. |
| `terrain` | `TerrainConfig()` | Terrain thresholds. |
| `neural` | `NeuralConfig()` | Neural brain architecture. |
| `reward` | `RewardConfig()` | Reinforcement-learning reward function. |

Why `gamemaker_enabled` is `True`: measured over 20 seeded games, a strict day cutoff with neither the circle nor the endgame instinct ends no game with a victor. The circle is slow (`intervention_days` of 6) and rarely kills, so it is on by default. See [gamemaker.md](gamemaker.md) for the full trade-off.

### `SimulationConfig.ticks_per_game`

```python
@property
def ticks_per_game(self) -> int
```

`max_days * ticks_per_day`. With defaults that is 576 ticks. `Game.is_over` compares the clock against this.

### `SimulationConfig.thirst_per_tick`

```python
@property
def thirst_per_tick(self) -> float
```

`1.0 / (thirst_days * ticks_per_day)`. With defaults, 1/72 of the bar per tick, so a full bar lasts exactly three days.

### `SimulationConfig.hunger_per_tick`

```python
@property
def hunger_per_tick(self) -> float
```

`1.0 / (hunger_days * ticks_per_day)`. With defaults, 1/168 per tick.

### `SimulationConfig.to_dict_raw`

```python
def to_dict_raw(self) -> dict
```

A shallow dictionary with one entry per field, values untouched (enums stay enums, nested configs stay objects). It is the "copy with changes" helper:

```python
from hunger_games.config import SimulationConfig

base = SimulationConfig(seed=7)
faster = SimulationConfig(**{**base.to_dict_raw(), "max_days": 10})
```

It tolerates old pickles. For each declared field it checks `hasattr(self, name)`. If an older config lacks the field, it falls back to the field's default, or calls the default factory. This is why a dashboard launched before a field was added no longer crashes its worker processes.

### `SimulationConfig.to_dict`

```python
def to_dict(self) -> dict
```

A deep, JSON-friendly dictionary. `dataclasses.asdict` recurses into the nested configs, then `shape` and `layout` are replaced by their string values. Tuples (`hidden_layers`, `career_districts`) are left as tuples; `json.dumps` writes them as lists.

### `SimulationConfig.from_dict`

```python
@classmethod
def from_dict(cls, data: dict) -> "SimulationConfig"
```

The inverse of `to_dict`. Copies the input, turns the two strings back into enums (defaulting to the class defaults if missing), rebuilds `NoiseConfig`, `TerrainConfig`, `NeuralConfig` and `RewardConfig` from their sub-dictionaries (an empty dictionary gives defaults), and converts `hidden_layers` and `career_districts` back from lists to tuples.

```python
import json
from hunger_games.config import SimulationConfig

text = json.dumps(SimulationConfig(max_days=12).to_dict())
again = SimulationConfig.from_dict(json.loads(text))
assert again.max_days == 12 and again.neural.hidden_layers == (16,)
```

## How to use it / experiment

Build a config with keyword arguments and hand it to a game:

```python
from hunger_games.config import ArenaShape, SimulationConfig
from hunger_games.game import Game

config = SimulationConfig(shape=ArenaShape.ROUND, seed=42, endgame_instinct=True)
result = Game(config).run()
print(result.winner_name, result.days)
```

Things worth trying:

- Set `gamemaker_enabled=False` and `endgame_instinct=True` to see whether the instinct alone ends games.
- Set `cannon_and_sky=False` to hide the field from the tributes. The four field slots of the perception vector become zeros (and `my_rank` 0.5).
- Shrink `intervention_days` to make the circle bite faster, or raise `quiet_days_before_intervention` to give tributes more time.
- Change `reward` and retrain with `experiments/run_rl.py` to see what behaviour a different reward buys.

Sweeps over any field, including nested ones written as `"terrain.water_threshold"`, are handled by `research/experiments.py`.

## Gotchas

- The command line in [main.md](main.md) exposes `gamemaker_enabled` as the flag pair `--gamemaker` / `--no-gamemaker`, defaulting to the config value (on). Pass `--no-gamemaker` to test a design with no intervention.
- `to_dict_raw` is shallow. The copy shares its `noise`, `terrain`, `neural` and `reward` objects with the original. Mutating `copy.neural.hidden_layers` would affect both (though tuples cannot be mutated, `activation` can be reassigned).
- `from_dict` requires every key it does not default to be present with the right name; extra keys raise `TypeError` from the generated `__init__`.
- `max_days` is a strict cutoff. A game that reaches it ends in a draw, with every survivor sharing a placing.
- `seed=None` means a new random seed per `Game`. To reproduce a batch, set a seed; `Game` adds its `game_id` to it so each game in the batch still differs.
- `start_*_min` values above 1.0 are clamped by `Game._start_value` (`min(minimum, 1.0)`), so they cannot make bars start above full.
