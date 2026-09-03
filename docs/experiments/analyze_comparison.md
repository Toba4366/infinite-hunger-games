# `analyze_comparison.py`

**Source:** [experiments/analyze_comparison.py](../../experiments/analyze_comparison.py)
**Depends on:** `argparse`, `json`, `math`, `sys`, `pathlib` (standard library); `numpy`, `pandas`, `matplotlib`, `scipy` (`scipy.stats.linregress`, imported inside `trend`); [../research/plots.md](../research/plots.md) (`overlay_curves`)
**Used by:** the results pages under [../results/README.md](../results/README.md); `tests/test_analysis_scripts.py`

## Purpose

A comparison run ends with a table of win rates and a folder of learning curves. Those are raw data, not evidence. This script asks the questions a reader will ask: how wide is the uncertainty on a win rate measured over 75 games, is the difference between two champions more than luck, and was a variant still improving when its budget ran out. It answers with standard tools: Wilson confidence intervals, Fisher exact tests and straight-line regression slopes with p-values. It also draws smoothed learning curves, because validation win rates of 0, 0.5 and 1 are unreadable as raw lines.

## Concepts you need

**Wilson interval.** A confidence interval for a proportion that behaves near 0 and 1 (the plain "plus or minus two standard errors" interval goes negative for 0 wins). With 13 wins in 75 games the 95 percent interval is 0.10 to 0.28: the point estimate of 0.17 is far less certain than two decimals suggest.

**Fisher exact test.** Given two champions' wins and losses as a 2 by 2 table, the probability of a table at least as extreme under the hypothesis that both have the same true win rate. It is exact for small counts, which matters when one champion has 0 wins. The script computes it from the hypergeometric distribution with `math.comb`, and the test file checks it against scipy.

**Regression slope.** A straight line through survival (or score, entropy, validation wins) against the iteration. The slope per 100 iterations says how fast the variant was changing; the p-value says whether that trend is distinguishable from a flat line given the noise. A slope near zero with a large p-value means "stalled".

**Rolling mean.** Each value averaged with the previous `window - 1` values. The same smoothing the comparison uses for its win-rate charts.

## Walkthrough

### `wilson_interval(wins, games, z=1.96)`

The Wilson centre `(p + z²/2n) / (1 + z²/n)` and half-width, clamped to `[0, 1]`. `games == 0` returns `(0, 0)`.

### `fisher_exact(a, b, c, d)`

Rows are the two champions, columns are wins and losses. `probability(x)` is the hypergeometric probability of `x` wins for the first champion given the margins. The two-sided p-value sums the probabilities of every table at most as likely as the observed one (with a `1e-12` tolerance for floating-point ties), over the feasible range of `x`.

### `trend(values)`

`scipy.stats.linregress` on the values against `0, 1, 2, ...`. Returns the slope times 100, the two-sided p-value and R squared. Fewer than three values returns a flat line with p = 1.

### `load_run(folder)`

Reads `results.csv` and `summary.json`, then every variant's `runs/<variant>_*/learning.json` (the last one when there are several, because the extension phase writes a second folder under `runs/`), falling back to the learning rows stored in the summary.

### `tournament_intervals(table, tournament)`

Recovers the win count from the stored win rate and game count, then the Wilson interval, one row per variant.

### `pair_tests(intervals)`

Two families of comparisons: every `<method>_warm` against its `<method>_cold`, and every champion against `imitation` (or the first variant whose name starts with `imitation`). Each row has both win counts, the difference in win rate, the Fisher p-value and a `significant` flag at 0.05.

### `trends(learning)`

Per variant, the slope, p-value and R squared for survival (`mean_length`), score (`mean_score`), entropy and `val_win_rate` over the whole run, plus the first and last survival.

### `draw_charts(intervals, learning, window, folder)`

A bar chart of win rates with asymmetric error bars from the Wilson intervals (`tournament_win_rate_ci.png`), and four smoothed overlay curves (`survival_smoothed.png`, `score_smoothed.png`, `entropy_smoothed.png`, `val_win_smoothed.png`).

### `write_stats(folder, intervals, tests, slopes, window)`

`stats.md`: the three tables as Markdown, ready to paste into a report.

### `analyze(run_folder, out="analysis", window=10)` and `main()`

Runs everything, writes `tournament_ci.csv`, `pair_tests.csv`, `trends.csv`, the charts and `stats.md` under `<run folder>/<out>/`, and `main` prints `stats.md`.

## How to use it / experiment

```bash
python experiments/analyze_comparison.py results/full_methods_20260903_025758
python experiments/analyze_comparison.py results/sizes_20260903_135744 --window 5
```

- The pair tests are the answer to "is warm really better than cold": if the Fisher p-value is above 0.05, the run does not show a difference, whatever the point estimates say.
- A trend row with a positive survival slope and a small p-value for a variant that missed the criterion means "more iterations would have helped"; a flat slope means the budget was not the problem.
- Change `--window` to match the criterion window of the run you are analysing.

## Gotchas

- **Win counts are recovered by rounding** `win_rate * games`; that is exact for the rates the comparison writes.
- **Trends are whole-run slopes.** A variant that rose and then fell gets a small slope; look at the smoothed chart before quoting the number.
- **One seed per variant** means the intervals describe the uncertainty of 75 tournament games, not the variance between training runs. Repeating a run with other seeds is the only way to measure that.
