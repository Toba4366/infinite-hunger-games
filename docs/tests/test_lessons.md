# `test_lessons.py`

**Source:** [tests/test_lessons.py](../../tests/test_lessons.py)
**Tests:** [../training/common.md](../training/common.md) (`Stage`, `CurriculumConfig.lessons`, `Curriculum.stages`, `stage_spec`, `describe`, `observe`, `apply_overrides`, `stage_config`, `episode_config`), with [../config.md](../config.md) (`SimulationConfig`)

## Purpose

The lesson curriculum changes what a stage is: not just a number of opponents but a set of rules and a promotion metric of its own, with per-episode variation in the last lesson. These tests pin down four things: the old ladder still behaves exactly as before, the lessons are the eight the research guide describes and are judged on the right metric, a stage's config has the right roster and rules without touching the base config, and the generalisation lesson varies the rules by seed and reproducibly.

## Concepts you need

**Stage.** `Stage(name, opponents, overrides, variants, metric, threshold)`. `overrides` apply to every game of the stage; `variants` are alternative override sets, one of which is picked per training episode.

**Promotion metrics.** `"win_rate"` (share of validation games won), `"survival"` (share of the game the learner's copies stayed alive, 0 to 1), `"score"` (mean return).

**Running it.** `python -m pytest tests/test_lessons.py -q`. Under a second; no games are played.

## Walkthrough

### `test_opponent_ladder_is_unchanged()`

`Curriculum(CurriculumConfig())` has stages with 1, 3, 7, 11 and 23 opponents judged on wins, starts against one, and is promoted on the fifth majority-win iteration, exactly as before the lessons existed.

### `test_lessons_start_with_survival_and_end_with_generalisation()`

`CurriculumConfig.lessons()` gives `survive`, `survive the rules`, five `beat N` lessons and `generalise`, with 0, 0, 1, 3, 7, 11, 23 and 23 opponents. The first lesson is judged on survival: five iterations with a perfect win rate but half-game survival do not promote; five with 95 percent survival do. `describe()` then reads `survive the rules (0 opponents)`, and the curriculum is not finished.

### `test_stage_config_applies_opponents_and_rules()`

The `survive` stage on a 40 by 40 base config with six learners gives a 6-player game with the circle and sponsors off, while the base config still has them on. `beat 7` gives 13 players. `apply_overrides` reaches a nested setting (`neural.hidden_layers`) and accepts an enum as its string value (`layout: "cornucopia"`).

### `test_episode_config_picks_a_variant_by_seed()`

Sixty seeds through the `generalise` stage produce at least four distinct rule sets, the same seed always gives the same rules, and a stage without variants returns the stage config object itself.

### `test_comparison_rejects_an_unknown_curriculum_name()`

`MethodComparison._build` on a `Variant` with `curriculum="lesson"` (a typo for `"lessons"`) raises `ValueError`. Without the guard the string would be truthy and silently select the opponent ladder.

## Gotchas

- **Survival is a share, not ticks.** Trainers divide the mean ticks survived by `config.ticks_per_game` before calling `observe`.
- **Validation games are not varied.** Only training episodes go through `episode_config`; validation plays the stage's base rules on fixed seeds so the win rate stays comparable across iterations.
