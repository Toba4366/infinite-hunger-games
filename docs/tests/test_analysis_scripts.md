# `test_analysis_scripts.py`

**Source:** [tests/test_analysis_scripts.py](../../tests/test_analysis_scripts.py)
**Tests:** [../experiments/analyze_comparison.md](../experiments/analyze_comparison.md) (`wilson_interval`, `fisher_exact`, `analyze`)

## Purpose

The statistics in the results pages come from `experiments/analyze_comparison.py`. These tests check the two hand-written statistics against known values and against scipy, and run the whole analysis on a fake run folder so the file layout it expects and the files it writes are pinned down.

## Concepts you need

**Importing a script.** `experiments/` is not a package, so the test inserts that folder into `sys.path` and imports `analyze_comparison` by name.

**A fake run folder.** `results.csv` with the variant names, `summary.json` with tournament numbers, and one `runs/<variant>_x/learning.json` per variant with twenty made-up rows whose survival rises by exactly 2 per iteration.

**Running it.** `python -m pytest tests/test_analysis_scripts.py -q`. A few seconds; it writes three small charts.

## Walkthrough

### `test_wilson_interval_matches_known_values()`

13 wins of 75 gives an interval of about 0.104 to 0.278 (the standard Wilson result at 95 percent), and 0 wins of 75 starts at exactly 0 and ends near 0.05.

### `test_fisher_exact_matches_scipy()`

Four tables, including 0 wins on both sides, agree with `scipy.stats.fisher_exact` to nine decimals.

### `test_analyze_writes_stats_for_a_tiny_run(tmp_path)`

Three variants (`imitation`, `ppo_cold`, `ppo_warm`) with 12, 5 and 2 wins of 75. After `analysis.analyze(run, "analysis", window=5)`:

- `stats.md` and `tournament_win_rate_ci.png` exist.
- `pair_tests.csv` has exactly one "warm against cold" row and two "against imitation" rows.
- `trends.csv` reports a survival slope of 200 per 100 iterations for the first variant with a p-value below `1e-6`, because the fake survival rises by exactly 2 per iteration.

## Gotchas

- **scipy is required** for `fisher_exact` cross-checks and for `trend`; it is part of the project's environment.
- **The charts are real PNGs**, so a matplotlib backend problem shows here first.
