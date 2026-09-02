# `sponsors.py`

**Source:** [hunger_games/sponsors.py](../hunger_games/sponsors.py)
**Depends on:** [config.py](config.md) (`SimulationConfig`), [districts.py](districts.md) (`is_career_district`), `numpy`, and the standard library (`dataclasses`, `typing`). It mentions `Player` from [player.py](player.md) in type hints only.
**Used by:** [game.py](game.md) (`SponsorPool`, `SponsorGift`), `recorder.py` (stores each `SponsorGift` in a replay), [renderer.py](renderer.md) and the dashboard's `ui/canvas.py` (draw parachutes from `game.gifts`), and `tests/test_scenario.py`.

## Purpose

In the films a wounded or starving tribute with rich sponsors gets a silver parachute: medicine, food or water, dropped from the sky. Sponsors back the tributes they like. This module is that mechanic. `SponsorPool` scores every living tribute's **favour** from `0.0` to `1.0`, works out what each one needs most, and once a day rolls to see whether a parachute arrives. `SponsorGift` records each delivery so the renderer can draw it and the recorder can save it.

Why does this module exist at all? The video points out that in the arena even a cut is deadly. `Player` models that: health below `WOUND_THRESHOLD` (`0.5`) is a serious wound that bleeds `BLEED_PER_TICK` (`0.004`) every step and will not close by resting. Only a medkit (`HEAL_AMOUNT`, `0.40`) can pull a tribute back. If medkits were common loot, every fight would be followed by a quick patch-up and the danger would vanish. So the layouts in [resources.md](resources.md) make medicine rare (5% of the Cornucopia pile, a 3% roll in the ring) and sponsors become the main path to healing.

That also makes the games feel like the films. Help comes by parachute to the tributes the Capitol favours: a high training score, the career districts 1, 2 and 4, and a record of kills. Favour is a weighted sum of exactly those three things, plus any bonus a game maker grants in the dashboard. A District 12 tribute with a training score of 1 has almost no chance of a parachute; a career with a score of 12 and three kills has close to the maximum.

`Game.__init__` builds one `SponsorPool(config)` and writes each player's starting favour onto `player.favor` so the dashboard can show it. `Game.step` calls `daily_gifts` at the start of every new day (when `tick % ticks_per_day == 0` and `tick > 0`) and appends the results to `game.gifts`.

## Concepts you need

**Dataclass as a record.** `SponsorGift` has no methods. It is a typed row, like the classes in [records.md](records.md).

**Class constants.** `FOOD_PARCEL`, `NEEDS_MEDICINE` and friends live on the class. `need_of` is a `@classmethod` because it only needs those constants, not a configured instance.

**Weighted sums and clamping.** Favour adds four parts that are designed to sum to at most `1.0`, then `np.clip(..., 0.0, 1.0)` guards against a large `favor_bonus` pushing it over.

**`min` as a cap.** `min(0.25, 0.08 * kills)` grows with kills but never passes `0.25`.

**Scaling a probability.** `rng.random() < gift_chance * favor` is true with probability `gift_chance * favor`. Favour does not change *what* you get, only *how likely* you are to get it.

**Ordered `if` checks as priority.** `need_of` returns on the first match, so medicine beats water beats food.

**`TYPE_CHECKING`.** `player.py` does not import this file, but this file only needs `Player` for hints, so the import is guarded to avoid any chance of a circle.

## Walkthrough

### `SponsorGift`

```python
@dataclass
class SponsorGift:
    game_id: int
    day: int
    tick: int
    player_id: int
    player_name: str
    kind: str
    favor: float
```

| Field | Meaning |
| --- | --- |
| `game_id` | Which game in a batch. |
| `day` | The in-game day the parachute landed. |
| `tick` | The exact tick. |
| `player_id` | Who received it. |
| `player_name` | Their name at the time. |
| `kind` | `"medicine"`, `"food"` or `"water"`. |
| `favor` | The receiver's favour when the roll was made, `0.0` to `1.0`. |

### `SponsorPool`

```python
class SponsorPool:
    FOOD_PARCEL = 3
    WATER_PARCEL = 0.6
    NEEDS_MEDICINE = 0.6
    NEEDS_FOOD = 0.35
    NEEDS_WATER = 0.35
```

| Constant | Value | Meaning |
| --- | --- | --- |
| `FOOD_PARCEL` | `3` | Rations in a food parcel. |
| `WATER_PARCEL` | `0.6` | How much of the thirst bar a water parcel restores. |
| `NEEDS_MEDICINE` | `0.6` | Health below this counts as needing medicine. |
| `NEEDS_FOOD` | `0.35` | Hunger below this counts as needing food. |
| `NEEDS_WATER` | `0.35` | Thirst below this counts as needing water. |

`NEEDS_MEDICINE` is above `Player.WOUND_THRESHOLD` (`0.5`) on purpose: sponsors start trying to help *before* a wound becomes a bleeding one.

### `SponsorPool.__init__`

```python
def __init__(self, config: SimulationConfig) -> None
```

Copies three settings out of the config:

| Attribute | From | Default |
| --- | --- | --- |
| `enabled` | `config.sponsors_enabled` | `True` |
| `gift_chance` | `config.sponsor_gift_chance` | `0.5` |
| `career_districts` | `config.career_districts` | `(1, 2, 4)` |

### `SponsorPool.favor`

```python
def favor(self, player: "Player") -> float
```

How much the sponsors like this tribute.

| Part | Formula | Maximum |
| --- | --- | --- |
| Training score | `0.5 * training_score / 12.0` | `0.5` at a score of 12 |
| Career district | `0.25` if `is_career_district(district, career_districts)` else `0.0` | `0.25` |
| Kills | `min(0.25, 0.08 * kills)` | `0.25` at 4 or more kills |
| Game maker bonus | `player.favor_bonus` | whatever was granted |

The sum is clipped to `0.0..1.0` and returned as a float. Some worked values:

| Tribute | Score | Career | Kills | Favour |
| --- | --- | --- | --- | --- |
| District 12 outsider | 1 | no | 0 | `0.042` |
| Average tribute | 6 | no | 0 | `0.25` |
| Career at the start | 10 | yes | 0 | `0.667` |
| Career after three kills | 12 | yes | 3 | `0.99` |

Kills are the only part that changes during a game, so a tribute's favour rises as they eliminate others. Audiences love a killer.

### `SponsorPool.need_of`

```python
@classmethod
def need_of(cls, player: "Player") -> str | None
```

What the tribute most needs, checked in this order:

1. `health < 0.6` gives `"medicine"` (wounds come first because untreated ones bleed).
2. `thirst < 0.35` gives `"water"` (thirst kills fastest).
3. `hunger < 0.35` gives `"food"`.
4. Otherwise `None`: comfortable tributes get nothing.

Sponsors do not send gifts to tributes who are fine, however famous. Only one need is answered per day.

### `SponsorPool.deliver`

```python
def deliver(self, player: "Player", kind: str) -> None
```

Puts the parcel's contents into the tribute's hands:

| Kind | Effect |
| --- | --- |
| `"medicine"` | `player.medicine += 1`. The brain decides when to use it (the `HEAL` action). |
| `"food"` | `player.food += 3`. |
| `"water"` | `player.thirst = min(1.0, player.thirst + 0.6)`. Drunk on the spot. |

Medicine and food go into the pack rather than being applied, so a badly wounded tribute with a brain that never chooses `HEAL` can still die holding a medkit. Water is applied immediately because there is nothing to decide.

### `SponsorPool.daily_gifts`

```python
def daily_gifts(self, players: list["Player"], rng: np.random.Generator, game_id: int, day: int, tick: int) -> list[SponsorGift]
```

Once a day, for every player:

1. If `enabled` is `False`, return `[]` at once.
2. Skip the dead.
3. `kind = need_of(player)`; skip if `None`.
4. `favor = self.favor(player)` and store it on `player.favor` for the dashboard.
5. If `rng.random() < gift_chance * favor`, call `deliver` and append a `SponsorGift`.

Returns the list of today's gifts. With the defaults, a fully favoured tribute in need has a 50% chance per day; a `0.25` favour tribute has a 12.5% chance. `Game.step` extends `game.gifts` with the result, and the renderer draws a parachute over each receiver for that day.

## How to use it / experiment

Compare favour for two tributes:

```python
from hunger_games.config import SimulationConfig
from hunger_games.game import Game

game = Game(SimulationConfig(seed=3, width=40, height=40))
pool = game.sponsors
star, nobody = game.players[0], game.players[1]
star.district, star.training_score, star.kills = 2, 12, 3
nobody.district, nobody.training_score, nobody.kills = 12, 1, 0
print(pool.favor(star), pool.favor(nobody))   # about 0.99 and 0.04
```

Force a gift and watch it land:

```python
game = Game(SimulationConfig(seed=3, width=40, height=40, sponsor_gift_chance=1.0))
player = game.players[0]
player.health = 0.3
player.favor_bonus = 1.0
gifts = game.sponsors.daily_gifts(game.players, game.rng, 0, 1, 24)
print([(g.player_name, g.kind) for g in gifts if g.player_id == player.player_id])
print(player.medicine)   # 1
```

Run a full game and count parachutes by kind:

```python
from collections import Counter
game = Game(SimulationConfig(seed=8))
result = game.run()
print(Counter(g.kind for g in game.gifts))        # SponsorGift objects on the game
print(Counter(g["kind"] for g in result.gifts))   # plain dictionaries on the GameResult
```

Turn sponsors off and see how much deadlier wounds become, or make everyone a career:

```python
Game(SimulationConfig(sponsors_enabled=False))
Game(SimulationConfig(career_districts=tuple(range(1, 13))))
```

Grant favour to a favourite via a scenario rather than by editing the player (see [scenario.md](scenario.md)):

```python
from hunger_games.scenario import Scenario, TributeSpec
roster = [TributeSpec(0, "Katniss", 12, "F", 11, 0.8, favor_bonus=0.3), TributeSpec(1, "Peeta", 12, "M", 8, 0.5)]
game = Game(SimulationConfig(seed=1, width=40, height=40), scenario=Scenario(tributes=roster))
print(game.players[0].favor)   # 0.458 + 0.3
```

## Gotchas

- **Gifts arrive once a day, not every tick.** `Game.step` only calls `daily_gifts` when `tick % ticks_per_day == 0` and `tick > 0`, so nothing lands on day one's first tick. A tribute who is badly hurt right after a fight may bleed for most of a day before the first roll.
- **`player.favor` is only refreshed on the daily roll.** `Game.__init__` sets it once, then `daily_gifts` updates it only for tributes who are *in need*. A comfortable tribute's displayed favour can lag behind their kills.
- **Medicine is delivered to the pack, not applied.** The brain must choose `HEAL`. The random brain often does not.
- **One need per day.** A tribute who is wounded, thirsty and starving gets a medicine roll only; water and food wait for tomorrow.
- **`favor_bonus` can exceed the natural maximum.** The sum is clipped at `1.0`, so a bonus of `1.0` makes anyone fully favoured regardless of score.
- **`sponsor_gift_chance` is a ceiling, not a rate.** The actual daily chance is `gift_chance * favor`, so the default `0.5` gives most tributes well under a coin flip.
- **Career districts come from the config.** Changing `DISTRICT_INDUSTRIES` or the colours in [districts.md](districts.md) has no effect on favour; only `SimulationConfig.career_districts` does.
- **The rolls consume the game's random stream.** Turning sponsors on or off changes every later dice roll in the game, so seeds are not comparable across that setting.
