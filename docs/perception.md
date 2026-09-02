# `perception.py`

**Source:** [hunger_games/perception.py](../hunger_games/perception.py)
**Depends on:** [resources.py](resources.md) (`ResourceKind`), [terrain.py](terrain.md) (`TerrainType`), `numpy`, and `dataclasses`.
**Used by:** [player.py](player.md) (builds one per tick in `perceive`), [game.py](game.md) (the type of the second hook argument), [brain/base.md](brain/base.md), [brain/voting.md](brain/voting.md), [brain/random_brain.md](brain/random_brain.md), [brain/neural.md](brain/neural.md) (`Perception`, `VECTOR_SIZE`), `training/reinforce.py` (`VECTOR_SIZE` for the value network), `research/telemetry.py`, `ui/app.py` (`VECTOR_NAMES`, `VECTOR_SIZE`), `ui/visualizer.py` (`VECTOR_NAMES` to label input nodes), [tests/test_brains.md](tests/test_brains.md) and [tests/test_initializers.md](tests/test_initializers.md).

## Purpose

A `Perception` is everything one tribute can sense during one tick: their own bars and pack, the ground under them, the nearest water, grass, supplies and people, the game makers' circle, the clock, and what the cannon and the nightly sky have told them about the rest of the field.

This is the only thing a brain ever sees. The body gathers it, the brain returns an `Action`, and the body obeys. Because the interface is one object, you can swap the voting brain for a neural network or your own code without touching the simulator.

`to_vector()` flattens the whole object into 50 numbers, each scaled to roughly -1..1, so a neural network can consume it directly. `VECTOR_SIZE` is that length and `VECTOR_NAMES` names each slot.

## Concepts you need

**Dataclasses with required and defaulted fields.** Fields without defaults (`thirst`, `hunger`, ...) must be given when constructing. Fields with defaults (`nearby_players`, `in_danger_zone`, ...) come after them; a dataclass does not allow a required field after a defaulted one.

**Chebyshev distance.** `max(abs(dx), abs(dy))`: the number of king moves. Every distance here is measured this way.

**Direction tuples.** `(dx, dy)` with each component in `{-1, 0, 1}`. `(0, 0)` means "no direction" (nothing in sight, or already there).

**Infinity.** `water_distance` and friends can be `float("inf")` when nothing is known. `to_vector` maps infinity to 1.0, meaning "as far as I can see".

**One-hot encoding.** The terrain underfoot becomes four slots, exactly one of them 1.0. Networks handle this better than a single "terrain number".

**Fixed layout.** The order of the vector is a contract. Trained networks store weights by position. Adding a field means retraining and updating `VECTOR_SIZE`, `VECTOR_NAMES` and the count comment.

## Walkthrough

### `NearbyPlayer`

```python
@dataclass
class NearbyPlayer:
    player_id: int
    dx: int
    dy: int
    distance: float
    threat: float
    health: float
```

What can be told about another tribute in sight. `player_id` is needed to target them with `ATTACK`. `dx`, `dy` are offsets from me (negative is left or up). `threat` is their `Player.threat_level` (survival skill, weapon and health blended) from 0.0 to 1.0. `health` is their bar.

### `NearbyPlayer.direction_toward`

```python
def direction_toward(self) -> tuple[int, int]
```

The sign of each offset: one step toward them. Uses the `(a > 0) - (a < 0)` trick, which subtracts two booleans to get -1, 0 or 1.

### `NearbyPlayer.direction_away`

```python
def direction_away(self) -> tuple[int, int]
```

The opposite of `direction_toward`. The voting brain's flee vote uses it.

### `Perception`

```python
@dataclass
class Perception:
```

Required fields, in order:

| Field | Type | Meaning |
| --- | --- | --- |
| `thirst` | `float` | 1.0 hydrated, 0.0 dead. |
| `hunger` | `float` | 1.0 full, 0.0 dead. |
| `health` | `float` | 1.0 unhurt, 0.0 dead. |
| `survival_score` | `float` | Fixed aptitude 0.0..1.0. |
| `training_score` | `float` | My own training score scaled to 0.0..1.0 (the 1..12 score divided by 12). |
| `weapon_quality` | `float` | Best weapon carried, 0.0 for bare hands. |
| `reach` | `int` | Cells my weapon can strike, 1 to 3. |
| `food_count` | `int` | Rations carried. |
| `medicine_count` | `int` | Medkits carried. |
| `terrain_here` | `TerrainType` | Ground under my feet. |
| `in_water` | `bool` | Standing in water. |
| `hunt_difficulty` | `float` | How hard hunting is here. |
| `downhill` | `tuple[int, int]` | Steepest downhill step. |
| `water_direction` | `tuple[int, int]` | Step toward the nearest water, `(0, 0)` if out of landmark range. |
| `water_distance` | `float` | Steps to water, may be `inf`. |
| `grass_direction` | `tuple[int, int]` | Step toward the nearest grass. |
| `grass_distance` | `float` | Steps to grass, may be `inf`. |
| `center_direction` | `tuple[int, int]` | Step toward the arena centre. |
| `center_distance` | `float` | 0.0 at the middle, 1.0 at the edge. |
| `resource_here_kind` | `ResourceKind` | Supply in my cell. |
| `resource_here_quantity` | `int` | How many. |
| `resource_here_quality` | `float` | How good. |
| `nearby_resource_direction` | `tuple[int, int]` | Step toward the nearest visible supply. |
| `nearby_resource_distance` | `float` | Steps to it, `inf` if none. |
| `nearby_resource_kind` | `ResourceKind` | Its kind. |

Defaulted fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `nearby_players` | `[]` | Living players within vision, nearest first. |
| `in_danger_zone` | `False` | Standing outside the safe circle. |
| `hazard_distance` | `999.0` | Cells of safe ground to the lethal edge; negative when outside. |
| `hazard_closing` | `False` | The circle is shrinking right now. |
| `safe_direction` | `(0, 0)` | Step toward safety. |
| `day_fraction` | `0.0` | Fraction of the maximum game length elapsed. |
| `alive_fraction` | `1.0` | Fraction of tributes still alive (the cannon). |
| `field_known` | `False` | Whether the next three fields mean anything. |
| `field_strength` | `0.0` | Mean training score of the other living tributes, 0..1. |
| `strongest_remaining` | `0.0` | The strongest other living tribute's score, 0..1. |
| `my_rank` | `0.5` | Fraction of other living tributes weaker than me. |
| `vision_radius` | `8` | Used to normalise distances. |

The four field values come from `Game.field_knowledge` (see [game.md](game.md)). With `cannon_and_sky` off they stay at their defaults, which is why `to_vector` produces zeros for them and 0.5 for rank.

### `Perception.nearest_threat`

```python
@property
def nearest_threat(self) -> NearbyPlayer | None
```

The first entry of `nearby_players`, or `None`. Works because `Player.perceive` sorts the list nearest first.

### `Perception.to_vector`

```python
def to_vector(self) -> np.ndarray
```

Builds a Python list of 50 floats in a fixed order and returns it as a numpy array. The local helper `scaled(distance)` returns `1.0` for infinity, else `min(1.0, distance / vision_radius)`. The nearest threat contributes five values, all zero when nobody is in sight except the distance, which is 1.0. The terrain underfoot is one-hot over water, sand, grass and rock.

The full index table:

| Index | Name | Source and scaling |
| --- | --- | --- |
| 0 | thirst | `thirst` |
| 1 | hunger | `hunger` |
| 2 | health | `health` |
| 3 | survival score | `survival_score` |
| 4 | training score | `training_score` (already 0..1) |
| 5 | weapon quality | `weapon_quality` |
| 6 | reach | `reach / 3.0` |
| 7 | food carried | `min(1.0, food_count / 5.0)` |
| 8 | medkits carried | `min(1.0, medicine_count / 3.0)` |
| 9 | in water | `float(in_water)` |
| 10 | hunt difficulty | `hunt_difficulty` |
| 11 | downhill dx | `downhill[0]` |
| 12 | downhill dy | `downhill[1]` |
| 13 | water dx | `water_direction[0]` |
| 14 | water dy | `water_direction[1]` |
| 15 | water distance | `scaled(water_distance)` |
| 16 | grass dx | `grass_direction[0]` |
| 17 | grass dy | `grass_direction[1]` |
| 18 | grass distance | `scaled(grass_distance)` |
| 19 | centre dx | `center_direction[0]` |
| 20 | centre dy | `center_direction[1]` |
| 21 | centre distance | `center_distance` |
| 22 | loot here kind | `int(resource_here_kind) / 3.0` (NONE 0, FOOD 1, WEAPON 2, MEDICINE 3) |
| 23 | loot here qty | `min(1.0, resource_here_quantity / 5.0)` |
| 24 | loot here quality | `resource_here_quality` |
| 25 | nearby loot dx | `nearby_resource_direction[0]` |
| 26 | nearby loot dy | `nearby_resource_direction[1]` |
| 27 | nearby loot distance | `scaled(nearby_resource_distance)` |
| 28 | nearby loot kind | `int(nearby_resource_kind) / 3.0` |
| 29 | threat dx | `threat.dx / vision_radius`, 0.0 if nobody |
| 30 | threat dy | `threat.dy / vision_radius`, 0.0 if nobody |
| 31 | threat distance | `scaled(threat.distance)`, 1.0 if nobody |
| 32 | threat level | `threat.threat`, 0.0 if nobody |
| 33 | threat health | `threat.health`, 0.0 if nobody |
| 34 | players in sight | `min(1.0, len(nearby_players) / 5.0)` |
| 35 | in danger zone | `float(in_danger_zone)` |
| 36 | hazard distance | `hazard_distance / vision_radius` clamped to -1..1 |
| 37 | hazard closing | `float(hazard_closing)` |
| 38 | safe dx | `safe_direction[0]` |
| 39 | safe dy | `safe_direction[1]` |
| 40 | day fraction | `day_fraction` |
| 41 | alive fraction | `alive_fraction` |
| 42 | field known | `float(field_known)` |
| 43 | field strength | `field_strength` |
| 44 | strongest remaining | `strongest_remaining` |
| 45 | my rank | `my_rank` |
| 46 | on water | `terrain_here is TerrainType.WATER` |
| 47 | on sand | `terrain_here is TerrainType.SAND` |
| 48 | on grass | `terrain_here is TerrainType.GRASS` |
| 49 | on rock | `terrain_here is TerrainType.ROCK` |

The source comment counts it the same way: 11 body and terrain, 2 downhill, 3 water, 3 grass, 3 centre, 3 here, 4 nearby supply, 5 threat, 1 crowd, 3 hazard, 2 safe, 2 clock, 4 field, 4 one-hot.

### `VECTOR_SIZE`

```python
VECTOR_SIZE = 50
```

The length of `to_vector()`. `NeuralBrain` sizes its input layer with it, and the RL value network does the same.

### `VECTOR_NAMES`

```python
VECTOR_NAMES = ["thirst", "hunger", ..., "on rock"]
```

Fifty human-readable names in vector order. The dashboard's Network tab labels input nodes with them and lists them as `index: name`. `ui/visualizer.py` only uses them when the first layer size equals `len(VECTOR_NAMES)`.

## How to use it / experiment

Get a real perception from a game and look at the vector:

```python
from hunger_games.config import SimulationConfig
from hunger_games.game import Game
from hunger_games.perception import VECTOR_NAMES, VECTOR_SIZE

game = Game(SimulationConfig(width=60, height=60, seed=5))
player = game.players[0]
perception = player.perceive(game.arena, game.players, False, 0.0, 1.0, game.config.vision_radius)
vector = perception.to_vector()
assert vector.shape == (VECTOR_SIZE,)
for name, value in zip(VECTOR_NAMES, vector, strict=True):
    print(f"{name:22s} {value:+.2f}")
```

After a game has stepped, `player.last_perception` holds the perception the brain last saw (see [player.md](player.md)), so you can inspect what a tribute knew when it made a decision.

To add a sense: add a field to `Perception`, fill it in `Player.perceive`, append it to the `values` list in `to_vector`, bump `VECTOR_SIZE`, add a name to `VECTOR_NAMES`, and expect saved neural genomes to stop loading.

## Gotchas

- `hazard_distance` defaults to `999.0`, so a tribute built without game maker information looks "very safe" (slot 36 clamps to 1.0).
- `safe_direction` is filled by `Player.perceive` with the direction to the centre, regardless of whether the circle is closing. Check `hazard_closing` or `in_danger_zone` first.
- `training_score` here is 0..1, but `Player.training_score` is the raw 1..12 integer. `perceive` does the division.
- `my_rank` is `0.5` when the field is unknown and `1.0` when the tribute is alone (no one to compare against).
- `nearby_players` only contains living tributes within `vision_radius`; dead ones vanish immediately.
- `int(resource_here_kind) / 3.0` relies on `ResourceKind` being an `IntEnum` with values 0..3. A new kind would push the scale past 1.0.
- Directions from the arena's distance fields are orthogonal steps, while `downhill` may be diagonal.
