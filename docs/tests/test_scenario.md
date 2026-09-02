# `test_scenario.py`

**Source:** [tests/test_scenario.py](../../tests/test_scenario.py)
**Tests:** `hunger_games/scenario.py` (`Scenario`, `TributeSpec`, `LootSpec`), [../game.md](../game.md) (`Game` built from a scenario, start bars), `hunger_games/sponsors.py` (`SponsorPool`), [../player.md](../player.md) (`rest`, `tick_needs`), [../resources.md](../resources.md) (`ResourceKind`, layouts), [../config.md](../config.md) (`SimulationConfig`), [../terrain.md](../terrain.md) (`TerrainType`)

## Purpose

A `Scenario` is everything a game maker can set up by hand in the dashboard: a painted terrain grid, hand-placed loot, and a roster of tributes with names, scores, brains, podiums and starting gear. It is plain data saved as JSON, and `Game` knows how to build a game from it. This file proves that the JSON round trip is lossless, that a game really adopts every part of a scenario, and that three features which arrived with scenarios work: configurable starting bars, sponsor gifts, and wounds that bleed.

The first test writes a scenario to disk and reads it back. The second builds a game on a flat painted map with two hand-edited tributes and checks the map, the loot, the podiums, the names, the gear and the brains. The third checks that tributes start with full bars by default and that `start_thirst_min` spreads them. The fourth checks the sponsor maths: a career with a high score and kills is favoured, and when wounded gets medicine. The fifth checks the wound rules on a `Player`. The sixth checks that medkits are rare in generated layouts, because sponsors are now meant to be the main source of healing.

## Concepts you need

**Test discovery.** pytest collects the six `test_*` functions. `flat_map` is a helper and is not collected.

**The `tmp_path` fixture.** Name a test parameter `tmp_path` and pytest hands you a fresh, empty `pathlib.Path` folder that is deleted later. `test_scenario_json_round_trip` uses it so the JSON file never lands in your project.

**Dataclass equality.** `@dataclass` generates `__eq__`, so `loaded.loot == scenario.loot` compares every field of every `LootSpec` in order. That is why one line can check the whole loot list.

**JSON has no tuples.** `json.dumps((2, 2))` writes `[2, 2]`. `Scenario.from_dict` converts the podium list back to a tuple. The test checks that specifically.

**Direct attribute pokes.** Several tests set `player.health = 0.4` or `star.kills = 3` straight on the object. That is fine in tests: it puts the object in the exact state you want without playing a game to get there.

**Running a subset.** `python -m pytest tests/test_scenario.py -k sponsor` runs one test.

## Walkthrough

### `flat_map(size: int = 40, kind: TerrainType = TerrainType.GRASS) -> list[list[int]]`

```python
def flat_map(size: int = 40, kind: TerrainType = TerrainType.GRASS) -> list[list[int]]:
    return np.full((size, size), int(kind), dtype=np.int8).tolist()
```

Builds a square grid where every cell is the same terrain and returns it as nested lists, which is the format `Scenario.terrain` stores. Grass everywhere means no water, no rock, no void, so podiums and loot land exactly where the test puts them.

### `test_scenario_json_round_trip(tmp_path)`

**Setup.** A `Scenario` with a 10 by 10 grass map, `use_layout_loot=False`, one `LootSpec(3, 4, WEAPON, 1, 0.9)`, one `TributeSpec` named Katniss from district 12 with `podium=(2, 2)` and `weapon_quality=0.9`, and `title="test"`. It is saved to `tmp_path / "s.json"` and loaded back.

**`assert loaded.terrain == scenario.terrain`.** Nested lists of ints survive JSON unchanged.

**`assert loaded.loot == scenario.loot`.** Dataclass equality over the whole list.

**`assert loaded.tributes[0].podium == (2, 2)`.** The list-to-tuple conversion in `from_dict` ran. Without it this would be `[2, 2]`, and `[2, 2] == (2, 2)` is `False` in Python.

**`assert loaded.tributes[0].name == "Katniss"`.** The roster came back.

**`assert loaded.use_layout_loot is False`.** A `False` must not be lost to the `data.get("use_layout_loot", True)` default.

### `test_game_uses_painted_map_loot_and_roster()`

**Setup.** Two tributes. Katniss: district 12, score 11, voting brain, podium `(5, 5)`, weapon 0.95, 3 food, 1 medicine, `favor_bonus=0.2`, `start_thirst=0.4`. Peeta: district 12, score 8, random brain, podium `(30, 30)`. A 40 by 40 grass map, `use_layout_loot=False`, and one hand-placed `LootSpec(10, 10, MEDICINE, 2, 0.5)`. Then `Game(SimulationConfig(seed=1, width=40, height=40), scenario=scenario)`.

**`assert (game.arena.terrain == int(TerrainType.GRASS)).all()`.** The painted map replaced the Perlin map entirely.

**`assert game.arena.resources.peek(10, 10) == (ResourceKind.MEDICINE, 2, 0.5)`.** The hand-placed stack is there with the right kind, quantity and quality.

**`assert (game.arena.resources.kind != 0).sum() == 1`.** Exactly one occupied cell. `use_layout_loot=False` stopped the ring layout from scattering anything else.

**`katniss, peeta = game.players`.** Two specs, two players, in roster order.

**`assert (katniss.x, katniss.y) == (5, 5) and (peeta.x, peeta.y) == (30, 30)`.** Roster podiums override the layout's podiums. On flat grass `snap_to_podium` leaves them alone.

**`assert katniss.name == "Katniss" and katniss.sex == "F" and katniss.weapon_quality == 0.95`.** Name, sex and pre-game weapon all copied.

**`assert katniss.food == 3 and katniss.medicine == 1 and katniss.favor_bonus == 0.2`.** Rations, medkits and the sponsor bonus.

**`assert katniss.thirst == 0.4 and peeta.thirst == 1.0`.** Katniss's `start_thirst=0.4` is used as an exact value. Peeta has no override, so `_start_value` draws from `uniform(start_thirst_min, 1.0)`, and with the default minimum of 1.0 that is exactly 1.0.

**`assert peeta.brain.name == "random"`.** Each tribute's `brain_name` is honoured, even when it differs from the config's `brain_name`.

### `test_start_bars_follow_config_minimums()`

**Setup.** Two games with `seed=2` on a 40 by 40 arena. The first uses defaults, the second sets `start_thirst_min=0.5`.

**`assert all(p.thirst == 1.0 and p.hunger == 1.0 and p.health == 1.0 for p in full.players)`.** With every minimum at 1.0, every bar starts full. This is the documented default and what the README's "everyone starts full" promise rests on.

**`assert all(0.5 <= p.thirst <= 1.0 for p in spread.players)`.** Thirst is drawn between the minimum and full.

**`assert any(p.thirst < 0.99 for p in spread.players)`.** At least one of the 24 draws is clearly below full, proving the minimum actually spreads values instead of being ignored. With seed 2 the lowest is about 0.50 and the highest about 0.94.

### `test_sponsor_favor_and_gifts()`

**Setup.** `Game(SimulationConfig(seed=3, width=40, height=40, sponsor_gift_chance=1.0))`, then `pool = game.sponsors`. Player 0 is made a star: district 2 (a career), score 12, 3 kills. Player 1 is made a nobody: district 12, score 1, no kills.

**`assert pool.favor(star) > pool.favor(nobody)`.** Favour is `0.5 * score / 12` plus 0.25 for a career plus `min(0.25, 0.08 * kills)` plus the bonus, clamped to 0 to 1. The star scores 0.5 + 0.25 + 0.24 = 0.99. The nobody scores 0.5 / 12, about 0.04.

**`star.health = 0.3` then `gifts = pool.daily_gifts([star, nobody], np.random.default_rng(0), 0, 2, 48)`.** Health below 0.6 means the star needs medicine. The nobody's bars are all full, so `need_of` returns `None` and no roll is made for them. The arguments are `game_id=0, day=2, tick=48`.

**`assert [g.kind for g in gifts] == ["medicine"] and star.medicine == 1`.** Exactly one gift, of kind medicine, and it was delivered: the medkit is in the star's pack. With `gift_chance=1.0` the roll is `random() < 0.99`, and the first draw from `default_rng(0)` is about 0.64, so the gift is certain for this seed.

**Why chance 1.0.** It removes luck from the test. The default chance of 0.5 would make the outcome depend on the seed.

### `test_deep_wounds_bleed_and_do_not_heal_by_resting()`

**Setup.** `Game(SimulationConfig(seed=4, width=40, height=40))`, `player = game.players[0]`.

**`player.health = 0.4; player.rest(); assert player.health == 0.4`.** Below `WOUND_THRESHOLD` (0.5) resting does nothing. A deep wound needs medicine.

**`player.tick_needs(0.0, 0.0); assert player.health < 0.4`.** With zero thirst and hunger drain, the only effect of `tick_needs` is bleeding, `BLEED_PER_TICK` = 0.004. The health drops to 0.396.

**`player.health = 0.9; player.rest(); assert player.health > 0.9`.** Above the threshold, resting adds `REST_AMOUNT` (0.02). Minor wounds close on their own.

### `test_medicine_is_rare_in_layouts()`

**Setup.** `Game(SimulationConfig(seed=5))`: the full default 120 by 120 ring layout.

**`assert (kinds == int(ResourceKind.MEDICINE)).sum() / total < 0.06`.** Medkits must be less than 6 percent of all supply cells. In the ring layout a non-weapon cell only becomes medicine on a 3 percent roll, and in the Cornucopia only 5 percent of the pile is medicine. With seed 5 the ring gives about 2.5 percent. A failure would mean someone made medkits common again, undoing the design where sponsors are the main way to survive a wound.

## How to run and extend

```bash
python -m pytest tests/test_scenario.py
python -m pytest tests/test_scenario.py -v
python -m pytest tests/test_scenario.py -k "round_trip or painted"
python -m pytest tests/test_scenario.py::test_sponsor_favor_and_gifts
```

**1. Loot outside the arena is dropped.** `Game` only places loot on walkable cells.

```python
def test_loot_in_the_void_is_ignored():
    terrain = flat_map()
    terrain[0][0] = int(TerrainType.VOID)
    scenario = Scenario(terrain=terrain, use_layout_loot=False, loot=[LootSpec(0, 0, int(ResourceKind.FOOD), 5, 0.5)])
    game = Game(SimulationConfig(seed=1, width=40, height=40), scenario=scenario)
    assert (game.arena.resources.kind != 0).sum() == 0
```

**2. A podium in the void is snapped inside.**

```python
def test_roster_podium_is_snapped_onto_the_arena():
    terrain = flat_map()
    terrain[0][0] = int(TerrainType.VOID)
    roster = [TributeSpec(0, "A", 1, "F", 5, 0.5, podium=(0, 0)), TributeSpec(1, "B", 1, "M", 5, 0.5, podium=(20, 20))]
    game = Game(SimulationConfig(seed=1, width=40, height=40), scenario=Scenario(terrain=terrain, tributes=roster))
    assert game.arena.is_walkable(game.players[0].x, game.players[0].y)
```

**3. Sponsors can be switched off.**

```python
def test_no_sponsors_means_no_gifts():
    game = Game(SimulationConfig(seed=3, width=40, height=40, sponsors_enabled=False, sponsor_gift_chance=1.0))
    star = game.players[0]
    star.health = 0.1
    assert game.sponsors.daily_gifts([star], np.random.default_rng(0), 0, 2, 48) == []
```

**4. Water need beats food need.** `need_of` checks medicine, then water, then food.

```python
def test_need_order():
    game = Game(SimulationConfig(seed=3, width=40, height=40))
    p = game.players[0]
    p.thirst, p.hunger = 0.1, 0.1
    assert SponsorPool.need_of(p) == "water"
```

## Gotchas

**`game.players` is roster order, not podium order.** `_spread_high_scorers` decides who stands where, but the list itself keeps the spec order. That is why `katniss, peeta = game.players` works.

**Roster fields are copied at construction.** Changing a `TributeSpec` after `Game(...)` has been built does nothing to the running game. The dashboard rebuilds the game for that reason.

**`favor` is recalculated once a day.** `pool.favor(player)` is a pure calculation; `player.favor` is only updated by `daily_gifts` (and once at game start). The test calls `pool.favor` directly to avoid that timing.

**`sponsor_gift_chance=1.0` still multiplies by favour.** A tribute with favour 0.04 only has a 4 percent chance even at "certain" gift chance. The nobody in the sponsor test never rolls at all, because they are not in need.

**Painted maps get synthetic heights.** `Arena._heights_from_terrain` gives grass a base height of 0.5 plus a little noise, so "flat" grass still has tiny slopes. Nothing here depends on that, but `downhill_direction` is not `(0, 0)` everywhere.

**The medicine test uses the full-size arena.** It is the slowest test in the file, about 0.13 seconds, because the ring layout visits 14400 cells in Python.
