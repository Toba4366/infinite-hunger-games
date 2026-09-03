# `test_runner_flags.py`

**Source:** [tests/test_runner_flags.py](../../tests/test_runner_flags.py)
**Tests:** [../experiments/run_comparison.md](../experiments/run_comparison.md) (`parse_settings`, `apply_side_settings`, `check_known`), with [../research/comparison.md](../research/comparison.md) (`Variant`)

## Purpose

The comparison runner's setting flags decide which trainer settings each variant trains with, and after the sensitivity sweep the cold and warm sides of a comparison get different ones. A flag that silently applied to nothing, or to the wrong side, would confound the warm-against-cold question the experiment exists to answer. These tests pin down the parsing, the side selection and the two loud failures.

## Concepts you need

**Importing a script.** `experiments/` is not a package, so the test inserts that folder into `sys.path` and imports `run_comparison` by name; nothing in it runs at import time.

**Sides.** A variant with `warm_from` is warm; one without is cold; imitation is neither and is never changed.

**Running it.** `python -m pytest tests/test_runner_flags.py -q`. Under a second.

## Walkthrough

### `test_parse_settings_reads_literals_and_strings()`

`learning_rate=1e-3,3e-3` gives two float pairs, `record_showcase=False` a boolean, `name=abc` a string; a spec without `=` raises `SystemExit`. Commas separate values, so a tuple cannot be passed through these flags.

### `test_side_settings_reach_only_their_side()`

Four variants: imitation, cold REINFORCE, warm REINFORCE and cold PPO. `--cold-set episodes_per_epoch=16` and `--cold-set reinforce.learning_rate=3e-3` with `--warm-set entropy_bonus=0.02`. Afterwards imitation has no settings object; both cold variants have 16 episodes per epoch; only cold REINFORCE has the learning rate 3e-3 (cold PPO keeps 1e-3); the warm variant keeps 4 episodes and gets the 0.02 entropy bonus.

### `test_check_known_rejects_unknown_fields_and_methods()`

Valid specs pass. `nonsense=1` raises `SystemExit`, and so does `reinforce.learning_rate=3e-3` when only `ppo` is being run.

## Gotchas

- **Settings objects are created on demand.** A variant with `settings=None` gets its method's default settings dataclass the first time a side setting reaches it; the defaults are the same ones the trainer would have used.
- **`--set` and the side flags compose.** `--set` builds variants first; the side flags then adjust them, so a `--set` value can be overridden by a side flag naming the same field.
