"""analysis.py - the charts from chapter 3, rebuilt from simulated data.

Chapter 3 plotted eliminations per day, eliminations by method, weapons
used, eliminations per training-score point, and placement by score, and
then complained about only having three games of data. These functions do
the same maths over as many games as the runner produced.
"""

# JSON for the telemetry file.
import json

# Filesystem paths.
from pathlib import Path

# matplotlib for the charts.
import matplotlib.pyplot as plt

# pandas for the tables.
import pandas as pd


def load_results(output_dir: str | Path = "output") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the three CSV files the runner wrote."""
    # Where to look.
    folder = Path(output_dir)
    # Eliminations table.
    eliminations = pd.read_csv(folder / "eliminations.csv")
    # Players table.
    players = pd.read_csv(folder / "players.csv")
    # Games table.
    games = pd.read_csv(folder / "games.csv")
    # Hand them back.
    return eliminations, players, games


def eliminations_per_day(eliminations: pd.DataFrame, num_games: int) -> pd.Series:
    """Average number of eliminations on each day (chapter 2's exponential decay)."""
    # Count eliminations per day, then divide by the number of games.
    return eliminations.groupby("day").size() / num_games


def eliminations_by_method(eliminations: pd.DataFrame) -> pd.Series:
    """How many eliminations fell into each of the three categories."""
    # Count rows per method.
    return eliminations["method"].value_counts()


def weapons_used(eliminations: pd.DataFrame) -> pd.Series:
    """Which weapons did the killing in player-vs-player eliminations."""
    # Only the fights.
    fights = eliminations[eliminations["method"] == "player_vs_player"]
    # Count rows per weapon.
    return fights["weapon"].value_counts()


def kills_per_training_point(players: pd.DataFrame) -> float:
    """Chapter 3's statistic: average of (kills / training score) across all players."""
    # Divide each player's kills by their score, then average.
    return float((players["kills"] / players["training_score"]).mean())


def placement_by_training_score(players: pd.DataFrame) -> pd.Series:
    """Average final placing for each training score (lower is better)."""
    # Group by score and average the placings.
    return players.groupby("training_score")["placement"].mean()


def game_lengths(games: pd.DataFrame) -> pd.Series:
    """How many games lasted each number of days."""
    # Count games per length, sorted by length.
    return games["days"].value_counts().sort_index()


def make_report(output_dir: str | Path = "output", show: bool = False) -> Path:
    """Draw the chapter 3 charts: one PNG per chart under `<output_dir>/plots/`
    (for papers), plus a single combined `report.png` (for a quick look).
    If the runner collected behaviour telemetry (`telemetry.json`), the
    behaviour charts are written too.
    """
    # Load the tables.
    eliminations, players, games = load_results(output_dir)
    # Individual charts.
    from hunger_games.research import plots

    # Arena size for the death heatmap: one more than the largest coordinate seen.
    width = int(eliminations["x"].max()) + 1 if len(eliminations) else 120
    height = int(eliminations["y"].max()) + 1 if len(eliminations) else 120
    # Write them.
    written = plots.batch_plots(eliminations, players, games, Path(output_dir) / "plots", width, height)
    # Behaviour charts if telemetry was collected.
    telemetry_path = Path(output_dir) / "telemetry.json"
    if telemetry_path.exists():
        written += plots.behaviour_plots(
            json.loads(telemetry_path.read_text()), Path(output_dir) / "plots" / "behaviour"
        )
    # Say how many.
    print(f"wrote {len(written)} individual charts to {Path(output_dir) / 'plots'}/")
    # How many games were played.
    num_games = len(games)
    # A 2-by-3 grid of charts.
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    # Chart 1: eliminations per day.
    eliminations_per_day(eliminations, num_games).plot.bar(ax=axes[0, 0], color="firebrick")
    # Label it.
    axes[0, 0].set_title("Eliminations per day (average per game)")
    # Chart 2: by method.
    eliminations_by_method(eliminations).plot.bar(ax=axes[0, 1], color="slateblue")
    # Label it.
    axes[0, 1].set_title("Eliminations by method")
    # Chart 3: weapons.
    weapons_used(eliminations).plot.bar(ax=axes[0, 2], color="darkorange")
    # Label it.
    axes[0, 2].set_title("Weapons used (player vs player)")
    # Chart 4: placement by score.
    placement_by_training_score(players).plot(ax=axes[1, 0], marker="o", color="seagreen")
    # Label it.
    axes[1, 0].set_title("Average placing by training score (lower = better)")
    # Invert so "better" is up.
    axes[1, 0].invert_yaxis()
    # Chart 5: kills by score.
    players.groupby("training_score")["kills"].mean().plot(ax=axes[1, 1], marker="o", color="steelblue")
    # Label it.
    axes[1, 1].set_title("Average kills by training score")
    # Chart 6: game lengths.
    game_lengths(games).plot.bar(ax=axes[1, 2], color="grey")
    # Label it.
    axes[1, 2].set_title("Game length in days")
    # Tidy the layout.
    fig.tight_layout()
    # Where to save.
    report_path = Path(output_dir) / "report.png"
    # Save.
    fig.savefig(report_path, dpi=120)
    # Print the headline numbers.
    print(f"games: {num_games}")
    # The statistic chapter 3 computed as 0.134.
    print(f"eliminations per training point: {kills_per_training_point(players):.3f}")
    # Share of each method.
    print("eliminations by method (%):")
    # Percentages.
    print((eliminations_by_method(eliminations) / len(eliminations) * 100).round(1).to_string())
    # How often the game makers stepped in.
    print(f"average interventions per game: {games['interventions'].mean():.2f}")
    # Optionally pop the window.
    if show:
        plt.show()
    # Hand back the path.
    return report_path
