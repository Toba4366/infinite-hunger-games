# `test_game.py`

**Source:** [tests/test_game.py](../../tests/test_game.py)
**Tests:** [../game.md](../game.md) (`Game`, including `run`, `_eliminate`, `_finish`, `_spread_high_scorers`), [../records.md](../records.md) (`GameResult`, `PlayerResult`, `Elimination`), [../config.md](../config.md) (`SimulationConfig`, `ArenaShape`, `LayoutName`)

## Purpose

`Game` is the referee. It builds the arena, scatters supplies, creates 24 tributes, stands them on podiums, and then ticks: every living player senses, decides and acts, time passes, the game makers may shrink the safe circle, and deaths are recorded. When one player is left or the day limit is hit, it assigns final placings and returns a `GameResult`. This file plays whole games and checks the bookkeeping.

These are integration tests. They do not check any single formula. Instead they prove that the whole machine runs without crashing, that the records it produces are internally consistent, and that the same seed replays the same game. The final test checks the chapter 2 podium rule in isolation: the top third of training scores are spread evenly around the ring so the strongest tributes do not start next to each other.

The bugs these catch are the ones that only show up when everything runs together. A player eliminated twice would give duplicate placements. A survivor left with `placement=None` would break the results CSV. A stray unseeded `np.random` call anywhere in the package would make replays diverge. A shape and layout combination that raises an exception on tick 40 would never be seen by the unit tests.

## Concepts you need

**Test discovery.** pytest runs every `test_*` function in the file. `small_config` is a helper because its name has no `test_` prefix.

**Helper with overrides.** `small_config(**overrides)` starts from a dictionary of settings, updates it with whatever the caller passes, and builds a `SimulationConfig`. This is the same idea as `make_arena` in `test_arena.py`, written with an explicit `dict` so you can see the two steps.

**List comprehensions as filters.** `[player for player in result.players if not player.alive_at_end]` builds the list of dead tributes in one line. The first test splits the roster into `dead` and `survivors` this way and checks each group separately.

**Conditional assertions.** `if result.winner_id is not None:` wraps an assertion that only makes sense when there was a sole victor. A draw skips it. Both outcomes are legitimate, so the test is written to be strict in each case rather than assuming one.

**Calling a private static method.** `Game._spread_high_scorers(game.players)` calls a `@staticmethod` directly. The leading underscore is a convention meaning "internal", but tests may still call it to check one rule without running a whole game.

**`zip(list, list[1:], strict=False)`.** Pairs each element with the next one, a standard way to check gaps between consecutive values. `strict=False` tells Python it is fine that the second list is one shorter; the pairing simply stops at the end of the shorter one. The keyword was added by the ruff formatter and does not change the behaviour.

**Running a subset.** `python -m pytest tests/test_game.py -k seed`.

## Walkthrough

### `small_config(**overrides) -> SimulationConfig`

```python
def small_config(**overrides) -> SimulationConfig:
    settings = dict(width=60, height=60, seed=11, max_days=10)
    settings.update(overrides)
    return SimulationConfig(**settings)
```

A 60 by 60 arena, `seed=11`, and a 10-day limit instead of the default 24. Everything else keeps its default: 24 players, the voting brain, `chaos=0.5`, the ring layout, the open field, sponsors on, and the game makers on (the slow safe circle that starts after a quiet day). Ten days is 240 ticks, which keeps each game around a sixth of a second.

### `test_game_runs_to_completion_with_consistent_bookkeeping()`

```python
def test_game_runs_to_completion_with_consistent_bookkeeping():
```

**Setup.** `Game(small_config()).run()` plays until `is_over` and returns a `GameResult`.

**`dead` and `survivors`.** The players are split by `alive_at_end` into those who died and those still standing when the game stopped. Everything below is checked separately for the two groups, so the test is strict whether the game ends with a victor or in a draw.

**`assert len(result.eliminations) == len(dead)`.** Every death writes exactly one `Elimination` row, and every dead player has `alive_at_end=False`. If these numbers differ, either `_eliminate` ran twice for one player (its `if not player.alive: return` guard is what prevents this) or a death path forgot to record a row.

**`assert sorted(player.placement for player in dead) == list(range(len(survivors) + 1, 25))`.** Each dead tribute's placement is the number of players alive at the moment they died: 24 for the first out, 2 for the runner-up. So the dead placements, sorted, must be an unbroken run from one more than the survivor count up to 24. With one survivor that is 2 through 24; with three survivors it is 4 through 24. A failure would mean a placement was skipped, duplicated, or computed after the alive flag was cleared.

**`assert all(player.placement == len(survivors) for player in survivors)`.** `_finish` gives every survivor the same placement, equal to how many survived. A sole victor gets 1. In a draw with three survivors all three get 3.

**`if result.winner_id is not None:` then `assert len(survivors) == 1 and survivors[0].placement == 1`.** `result()` names a winner only when exactly one survivor is left, so a winner id must go together with a single survivor placed first.

**Why `seed=11`.** It is the seed the helper fixes for every game test. With the current settings it runs to the 240-tick limit and ends in a draw with three survivors, after one game maker intervention, so the `if` branch is skipped and the draw arithmetic is what gets exercised. Changing the seed to one that produces a sole victor exercises the winner branch instead; both are covered.

### `test_same_seed_reproduces_the_same_game()`

```python
def test_same_seed_reproduces_the_same_game():
```

**Setup.** Two games with identical configs at `chaos=1.0`, the most random setting. Chaos scales the hunt luck, fight luck, terrain roughness, and how often a brain picks a lower-voted action. All of that randomness flows from one `np.random.default_rng(seed)` inside `Game`.

**`assert first.elimination_rows() == second.elimination_rows()`.** `elimination_rows` turns every `Elimination` into a dictionary with the day, tick, victim, method, weapon, killer, coordinates and placement. Two lists of dictionaries compare equal only if every field of every death matches. If any code path used an unseeded generator, such as a stray `np.random.random()` or Python's `random` module, the two games would diverge within a few ticks and this would fail. It is the strongest single guarantee in the suite, because the README promises that a seed replays a game exactly.

**Why `chaos=1.0`.** Maximum chaos makes the most random draws, so any leak has the most chances to show up.

### `test_all_shapes_and_layouts_run()`

```python
def test_all_shapes_and_layouts_run():
```

**Setup.** Nested loops over `ArenaShape` (open field, round) and `LayoutName` (cornucopia, ring), four combinations. Each gets `max_days=4` to keep the test quick.

**`assert result.ticks > 0`.** The game must have advanced at least one tick. The real check is that `run()` returned at all. Any exception inside the arena carve, the layout, `spawn_positions`, `edge_positions` or the tick loop would propagate up and fail the test with a traceback. The round arena is the interesting case: void cells must be handled by `is_walkable`, `snap_to_podium` and the distance fields, and the cornucopia podiums must be nudged inside the circle if its radius of 10 reaches the void.

**Why `max_days=4`.** Four games at 96 ticks each finish in under half a second in total. The test is about crashes, not outcomes, so a short game is enough.

### `test_strong_players_spread_apart()`

```python
def test_strong_players_spread_apart():
```

**Setup.** Builds a game (which creates players with seeded training scores) and calls `Game._spread_high_scorers(game.players)` directly to get the podium order. Then it recomputes the top third independently: sort by `training_score` descending and take the first `len(players) // 3`, which is 8 of 24.

**`top_indices = [index for index, player in enumerate(ordered) if player in top]`.** The positions in the podium order that belong to top scorers.

**`assert all(b - a >= 3 for a, b in zip(top_indices, top_indices[1:], strict=False))`.** Consecutive top scorers must be at least 3 podiums apart. The method places the top third at `int(index * stride)` where `stride = 24 / 8 = 3`, so the expected positions are 0, 3, 6, ..., 21, and the rest fill the gaps. With `seed=11` the ordered scores begin `11, 7, 7, 10, 7, 6, 10, 6, 6, 8, ...`, and the top indices are exactly `[0, 3, 6, 9, 12, 15, 18, 21]`. A failure would mean the stride was computed wrongly, the fill loop overwrote a reserved slot, or the sort direction was reversed so the weakest were spread instead.

**Why `len // 3`.** It matches `max(1, count // 3)` in the method. The test recomputes it rather than importing it so that a change in the method's definition of "top third" would be noticed.

## How to run and extend

```bash
python -m pytest tests/test_game.py
python -m pytest tests/test_game.py::test_same_seed_reproduces_the_same_game
python -m pytest tests/test_game.py -k "shapes or spread"
python -m pytest tests/test_game.py -v
python -m pytest tests/test_game.py --durations=0   # show how long each test took
```

Ideas for new tests in this area:

**1. Chaos 0 gives identical games.** At chaos 0 the hunt and fight luck are zero, so the only randomness left is the terrain, the act order and the sponsor rolls, and all of it comes from the seeded generator.

```python
def test_chaos_zero_is_deterministic():
    first = Game(small_config(chaos=0.0)).run()
    second = Game(small_config(chaos=0.0)).run()
    assert first.elimination_rows() == second.elimination_rows()
    assert first.player_rows() == second.player_rows()
```

**2. Different `game_id` values give different games from one seed.** `Game` offsets the seed by `game_id` so a batch is varied but reproducible.

```python
def test_game_id_varies_the_seed():
    first = Game(small_config(), game_id=0).run()
    second = Game(small_config(), game_id=1).run()
    assert first.seed != second.seed
    assert first.elimination_rows() != second.elimination_rows()
```

**3. Disabling the game makers removes hazard deaths.**

```python
def test_no_gamemaker_means_no_hazard_deaths():
    result = Game(small_config(gamemaker_enabled=False)).run()
    assert result.interventions == 0
    assert all(row.method != "gamemaker" for row in result.eliminations)
```

**4. Research hooks see every decision.** `Game.decision_hooks` and `Game.tick_hooks` are how the telemetry in `research/` plugs in.

```python
def test_hooks_fire():
    game = Game(small_config(max_days=1))
    decisions, ticks = [], []
    game.decision_hooks.append(lambda player, perception, action: decisions.append(action))
    game.tick_hooks.append(lambda g: ticks.append(g.tick))
    game.run()
    assert ticks == list(range(1, game.tick + 1)) and len(decisions) >= len(ticks)
```

## Gotchas

**These are the slow tests.** The reproducibility test and the four-combination test take about 0.4 seconds each on a 60 by 60 arena. At the default 120 by 120 they would take several seconds, because the layouts loop over every cell in Python and each tick runs 24 perceptions. Keep test arenas small and day limits short.

**The draw case is real.** `seed=11` with these settings ends at the day limit with three survivors. If you change the seed, `max_days`, or any default such as `thirst_days` or `intervention_days`, the first test may switch to the sole-victor branch. Both are valid; the `if` exists for this reason.

**Reproducibility depends on the whole package.** `test_same_seed_reproduces_the_same_game` will fail if any module, even one not obviously random, calls an unseeded generator. If it starts failing after you add code, search for `np.random.` calls that do not go through `self.rng` or a passed-in generator.

**Placement numbering.** Placement is the number of players alive at the moment of death, so the first tribute out is placed 24 and the victor 1. Survivors in a draw share the survivor count. Do not expect `None` anywhere after `run()`; `_finish` fills every placement.

**Default config values matter.** All four tests rely on `num_players=24` (for the `range(1, 25)` and the `// 3` maths), `ticks_per_day=24`, and `brain_name="voting"`. Changing those defaults in `config.py` will change what these tests see.

**The game makers are on by default.** `gamemaker_enabled` defaults to `True`, so the seed-11 game includes one intervention. Pass `gamemaker_enabled=False` through `small_config` if a new test needs a quiet arena.
