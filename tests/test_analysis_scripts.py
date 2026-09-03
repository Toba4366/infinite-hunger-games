"""Tests for the post-run analysis scripts: the statistics and the run-folder reader."""

import json
import sys
from pathlib import Path

import pandas as pd

# The experiments folder is not a package; import the scripts by path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

import analyze_comparison as analysis  # noqa: E402


def test_wilson_interval_matches_known_values():
    """13 wins of 75 gives the textbook Wilson interval, and 0 of 75 starts at zero."""
    low, high = analysis.wilson_interval(13, 75)
    assert abs(low - 0.104) < 0.005 and abs(high - 0.278) < 0.005
    low, high = analysis.wilson_interval(0, 75)
    assert low == 0.0 and 0.04 < high < 0.06


def test_fisher_exact_matches_scipy():
    """The hand-written Fisher test agrees with scipy on a few tables."""
    from scipy import stats

    for table in ((13, 62, 0, 75), (5, 70, 2, 73), (12, 63, 13, 62), (0, 10, 0, 10)):
        a, b, c, d = table
        assert abs(analysis.fisher_exact(a, b, c, d) - stats.fisher_exact([[a, b], [c, d]])[1]) < 1e-9


def test_analyze_writes_stats_for_a_tiny_run(tmp_path):
    """A fake run folder with two variants gets intervals, pair tests, trends, charts and stats.md."""
    run = tmp_path / "fake_run"
    (run / "runs" / "imitation_x").mkdir(parents=True)
    (run / "runs" / "ppo_cold_x").mkdir(parents=True)
    (run / "runs" / "ppo_warm_x").mkdir(parents=True)
    names = ["imitation", "ppo_cold", "ppo_warm"]
    pd.DataFrame({"variant": names}).to_csv(run / "results.csv", index=False)
    tournament = {
        "imitation": {"win_rate": 12 / 75, "games": 75},
        "ppo_cold": {"win_rate": 5 / 75, "games": 75},
        "ppo_warm": {"win_rate": 2 / 75, "games": 75},
    }
    (run / "summary.json").write_text(json.dumps({"tournament": tournament, "learning": {}}))
    for name in names:
        rows = [
            {
                "iteration": i,
                "mean_length": 100 + 2 * i,
                "mean_score": -1 + 0.01 * i,
                "entropy": 2.5 - 0.01 * i,
                "val_win_rate": 0.0,
            }
            for i in range(20)
        ]
        (run / "runs" / f"{name}_x" / "learning.json").write_text(json.dumps(rows))
    target = analysis.analyze(run, "analysis", window=5)
    assert (target / "stats.md").exists() and (target / "tournament_win_rate_ci.png").exists()
    tests = pd.read_csv(target / "pair_tests.csv")
    # One warm-against-cold pair and two comparisons against imitation.
    assert list(tests["comparison"]).count("warm against cold") == 1
    assert list(tests["comparison"]).count("against imitation") == 2
    trends = pd.read_csv(target / "trends.csv")
    # Survival rises by exactly 2 per iteration: 200 per 100 iterations, with a tiny p-value.
    assert abs(trends["survival_slope_per_100"].iloc[0] - 200) < 1e-6 and trends["survival_p"].iloc[0] < 1e-6
