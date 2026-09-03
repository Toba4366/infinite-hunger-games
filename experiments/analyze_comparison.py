"""Statistical analysis of a finished method comparison run.

`experiments/run_comparison.py` writes a run folder with `results.csv`, `summary.json`
and one `runs/<variant>_<stamp>/learning.json` per variant. This script turns those
raw numbers into the evidence a write-up needs:

* Wilson confidence intervals for every tournament win rate (75 games is not many).
* Fisher exact tests between pairs of champions (cold against warm, each against
  imitation), because a difference of a few games out of 75 may be noise.
* Regression slopes of survival, score and entropy against the iteration for every
  variant, with p-values, so "it was still learning" or "it had stalled" is a
  number rather than an impression.
* Smoothed learning curves (rolling means) and a tournament chart with error bars.

Usage:
    python experiments/analyze_comparison.py results/full_methods_20260903_025758
    python experiments/analyze_comparison.py <run folder> --window 10 --out analysis

Everything is written under `<run folder>/<out>/`: `stats.md` (tables ready to paste
into a report), `tournament_ci.csv`, `pair_tests.csv`, `trends.csv` and the charts.
"""

# Command-line flags.
import argparse

# JSON for the summary and learning files.
import json

# Square roots for the confidence intervals.
import math

# Paths.
import sys
from pathlib import Path

# Arrays and tables.
import numpy as np
import pandas as pd

# The project root, so the script runs from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Drawing (the research plots module already selects the non-interactive Agg backend).
import matplotlib.pyplot as plt  # noqa: E402

# The shared overlay chart.
from hunger_games.research import plots  # noqa: E402

# ------------------------------------------------------------------ statistics


def wilson_interval(wins: int, games: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval for a proportion: better than the normal approximation near 0 and 1."""
    # No games, no interval.
    if games == 0:
        return (0.0, 0.0)
    # The observed proportion.
    p = wins / games
    # The Wilson centre and half-width.
    denominator = 1 + z * z / games
    centre = (p + z * z / (2 * games)) / denominator
    half = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denominator
    # Clamp to [0, 1].
    return (max(0.0, centre - half), min(1.0, centre + half))


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test p-value for the 2x2 table [[a, b], [c, d]] (wins/losses of two champions)."""
    # Row and column totals.
    row1, row2 = a + b, c + d
    col1 = a + c
    total = row1 + row2

    # The probability of a table with `x` in the top-left cell, given the margins (hypergeometric).
    def probability(x: int) -> float:
        return math.comb(row1, x) * math.comb(row2, col1 - x) / math.comb(total, col1)

    # The observed table's probability.
    observed = probability(a)
    # Sum every table at most as likely as the observed one.
    low, high = max(0, col1 - row2), min(row1, col1)
    return float(sum(probability(x) for x in range(low, high + 1) if probability(x) <= observed + 1e-12))


def trend(values: list[float]) -> tuple[float, float, float]:
    """Slope per 100 iterations, its p-value, and R squared of a straight line through the values."""
    # Need at least three points.
    if len(values) < 3:
        return (0.0, 1.0, 0.0)
    # scipy's linear regression gives the slope and its two-sided p-value.
    from scipy import stats  # noqa: PLC0415 - optional dependency, imported only when needed

    xs = np.arange(len(values), dtype=float)
    result = stats.linregress(xs, np.asarray(values, dtype=float))
    return (float(result.slope * 100), float(result.pvalue), float(result.rvalue**2))


def rolling(values: list[float], window: int) -> list[float]:
    """Each value averaged with the ones before it, up to `window` long."""
    out = []
    for i in range(len(values)):
        out.append(float(np.mean(values[max(0, i - window + 1) : i + 1])))
    return out


# ------------------------------------------------------------------ the analysis


def load_run(folder: Path) -> tuple[pd.DataFrame, dict, dict[str, list[dict]]]:
    """Read the results table, the tournament numbers and every variant's learning rows."""
    # The table.
    table = pd.read_csv(folder / "results.csv")
    # Tournament numbers per variant.
    summary = json.loads((folder / "summary.json").read_text())
    tournament = summary["tournament"]
    # Learning rows: prefer the run folders (they hold the final state), fall back to the summary.
    learning: dict[str, list[dict]] = {}
    for name in table["variant"]:
        candidates = sorted((folder / "runs").glob(f"{name}_*/learning.json")) if (folder / "runs").exists() else []
        if candidates:
            learning[name] = json.loads(candidates[-1].read_text())
        else:
            learning[name] = summary["learning"].get(name, [])
    return table, tournament, learning


def tournament_intervals(table: pd.DataFrame, tournament: dict) -> pd.DataFrame:
    """Wins, games and the Wilson interval per champion."""
    rows = []
    for name in table["variant"]:
        games = int(tournament[name]["games"])
        wins = int(round(tournament[name]["win_rate"] * games))
        low, high = wilson_interval(wins, games)
        rows.append(
            {"variant": name, "wins": wins, "games": games, "win_rate": wins / games, "ci_low": low, "ci_high": high}
        )
    return pd.DataFrame(rows)


def pair_tests(intervals: pd.DataFrame) -> pd.DataFrame:
    """Fisher exact tests for cold-against-warm pairs and for every champion against imitation."""
    by_name = {row.variant: row for row in intervals.itertuples()}
    pairs: list[tuple[str, str, str]] = []
    # Cold against warm.
    for name in by_name:
        if name.endswith("_cold") and name.replace("_cold", "_warm") in by_name:
            pairs.append(("warm against cold", name.replace("_cold", "_warm"), name))
    # Everyone against the imitation champion (or the first imitation variant).
    baseline = next((n for n in by_name if n == "imitation"), None) or next(
        (n for n in by_name if n.startswith("imitation")), None
    )
    if baseline is not None:
        for name in by_name:
            if name != baseline:
                pairs.append(("against imitation", name, baseline))
    rows = []
    for kind, first, second in pairs:
        a, b = by_name[first], by_name[second]
        p = fisher_exact(a.wins, a.games - a.wins, b.wins, b.games - b.wins)
        rows.append(
            {
                "comparison": kind,
                "first": first,
                "second": second,
                "first_wins": a.wins,
                "second_wins": b.wins,
                "games": a.games,
                "difference": a.win_rate - b.win_rate,
                "fisher_p": p,
                "significant": p < 0.05,
            }
        )
    return pd.DataFrame(rows)


def trends(learning: dict[str, list[dict]]) -> pd.DataFrame:
    """Slopes of survival, score and entropy against the iteration, per variant, over the whole run."""
    rows = []
    for name, records in learning.items():
        if not records:
            continue
        entry = {"variant": name, "iterations": len(records)}
        for key, label in (
            ("mean_length", "survival"),
            ("mean_score", "score"),
            ("entropy", "entropy"),
            ("val_win_rate", "val_win"),
        ):
            slope, p, r2 = trend([r[key] for r in records])
            entry[f"{label}_slope_per_100"] = slope
            entry[f"{label}_p"] = p
            entry[f"{label}_r2"] = r2
        entry["first"] = records[0]["mean_length"]
        entry["last"] = records[-1]["mean_length"]
        rows.append(entry)
    return pd.DataFrame(rows)


def draw_charts(intervals: pd.DataFrame, learning: dict[str, list[dict]], window: int, folder: Path) -> list[Path]:
    """The tournament chart with error bars and the smoothed learning curves."""
    written = []
    # Tournament win rates with Wilson intervals.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(len(intervals))
    lows = intervals["win_rate"] - intervals["ci_low"]
    highs = intervals["ci_high"] - intervals["win_rate"]
    ax.bar(xs, intervals["win_rate"], color="slateblue", yerr=[lows, highs], capsize=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(intervals["variant"], rotation=30, ha="right")
    ax.set_ylabel("tournament win rate")
    ax.set_title("Tournament win rate with 95% Wilson intervals")
    path = folder / "tournament_win_rate_ci.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path)
    # Smoothed curves.
    for key, label, ylabel in (
        ("mean_length", "survival", "ticks survived"),
        ("mean_score", "score", "mean score"),
        ("entropy", "entropy", "nats"),
        ("val_win_rate", "val_win", "validation win rate"),
    ):
        series = {
            name: ([r["iteration"] for r in rows], rolling([r[key] for r in rows], window))
            for name, rows in learning.items()
            if rows
        }
        written.append(
            plots.overlay_curves(
                series,
                f"{label} per iteration (rolling mean over {window})",
                "iteration",
                ylabel,
                folder / f"{label}_smoothed.png",
            )
        )
    return written


def write_stats(folder: Path, intervals: pd.DataFrame, tests: pd.DataFrame, slopes: pd.DataFrame, window: int) -> Path:
    """A Markdown file with every table, ready to paste into a report."""
    lines = ["# Statistical analysis", ""]
    lines.append("## Tournament win rates with 95% Wilson intervals")
    lines.append("")
    lines.append("| variant | wins / games | win rate | 95% interval |")
    lines.append("| --- | --- | --- | --- |")
    for row in intervals.itertuples():
        lines.append(
            f"| {row.variant} | {row.wins} / {row.games} | {row.win_rate:.3f} | {row.ci_low:.3f} to {row.ci_high:.3f} |"
        )
    lines.append("")
    lines.append("## Pairwise Fisher exact tests (two-sided)")
    lines.append("")
    lines.append("| comparison | first | second | wins | difference | p | significant at 0.05 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in tests.itertuples():
        lines.append(
            f"| {row.comparison} | {row.first} | {row.second} | {row.first_wins} against {row.second_wins} of {row.games} | "
            f"{row.difference:+.3f} | {row.fisher_p:.4f} | {'yes' if row.significant else 'no'} |"
        )
    lines.append("")
    lines.append("## Learning trends (straight-line slopes per 100 iterations, whole run)")
    lines.append("")
    lines.append(
        "| variant | iterations | survival slope | p | score slope | p | entropy slope | p | validation win slope | p |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in slopes.itertuples():
        lines.append(
            f"| {row.variant} | {row.iterations} | {row.survival_slope_per_100:+.1f} | {row.survival_p:.3g} | "
            f"{row.score_slope_per_100:+.3f} | {row.score_p:.3g} | {row.entropy_slope_per_100:+.3f} | {row.entropy_p:.3g} | "
            f"{row.val_win_slope_per_100:+.3f} | {row.val_win_p:.3g} |"
        )
    lines.append("")
    lines.append(
        f"Smoothed curves use a rolling mean over {window} iterations. Charts: `tournament_win_rate_ci.png`, `survival_smoothed.png`, `score_smoothed.png`, `entropy_smoothed.png`, `val_win_smoothed.png`."
    )
    path = folder / "stats.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def analyze(run_folder: str | Path, out: str = "analysis", window: int = 10) -> Path:
    """Run every analysis on one comparison run folder and return the output folder."""
    folder = Path(run_folder)
    target = folder / out
    target.mkdir(parents=True, exist_ok=True)
    table, tournament, learning = load_run(folder)
    intervals = tournament_intervals(table, tournament)
    tests = pair_tests(intervals)
    slopes = trends(learning)
    intervals.to_csv(target / "tournament_ci.csv", index=False)
    tests.to_csv(target / "pair_tests.csv", index=False)
    slopes.to_csv(target / "trends.csv", index=False)
    draw_charts(intervals, learning, window, target)
    write_stats(target, intervals, tests, slopes, window)
    return target


def main() -> None:
    """Parse the flags and run the analysis."""
    parser = argparse.ArgumentParser(description="Statistics and smoothed charts for a comparison run folder.")
    parser.add_argument("run_folder", help="a results/<name>_<timestamp> folder written by run_comparison.py")
    parser.add_argument("--out", default="analysis", help="subfolder to write into (default: analysis)")
    parser.add_argument("--window", type=int, default=10, help="rolling-mean window for the smoothed curves")
    args = parser.parse_args()
    target = analyze(args.run_folder, args.out, args.window)
    print((target / "stats.md").read_text())
    print(f"written to {target}")


if __name__ == "__main__":
    main()
