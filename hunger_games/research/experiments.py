"""research/experiments.py - parameter sweeps that write timestamped run folders.

A researcher's question is usually "what happens to X if I change Y?".
`Sweep` takes one config field (even a nested one, like
"terrain.water_threshold"), a list of values, and a number of games per
value, plays them all, and writes a folder like

    results/water_threshold_20260902_1530/
        config.json     the base config and the sweep settings
        results.csv     one row per value with every metric
        summary.json    the same, plus merged behaviour telemetry per value
        plots/          one PNG per metric against the swept value

The same folder layout is used by the trainers (see `make_run_dir`).
"""

# Copying configs.
import copy

# JSON.
import json

# Type hints.
from collections.abc import Callable

# Settings.
from dataclasses import asdict, dataclass

# Timestamps for run folders.
from datetime import datetime

# Paths.
from pathlib import Path

# numpy.
import numpy as np

# pandas.
import pandas as pd

# Settings.
from hunger_games.config import SimulationConfig

# Plots.
from hunger_games.research import plots

# Telemetry merging.
from hunger_games.research.telemetry import BehaviorTelemetry

# The batch runner.
from hunger_games.runner import Runner

# Custom setups.
from hunger_games.scenario import Scenario


def make_run_dir(base: str | Path, name: str) -> Path:
    """Create `base/<name>_<timestamp>/plots` and return the run folder."""
    # Timestamp.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Folder.
    folder = Path(base) / f"{name}_{stamp}"
    # Make it, with the plots subfolder.
    (folder / "plots").mkdir(parents=True, exist_ok=True)
    # Done.
    return folder


def set_field(config: SimulationConfig, dotted: str, value) -> SimulationConfig:
    """Return a copy of the config with a (possibly nested) field set, e.g. 'terrain.water_threshold'."""
    # Deep copy so the original is untouched.
    updated = copy.deepcopy(config)
    # Walk the path.
    target = updated
    # All but the last part select nested objects.
    parts = dotted.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    # Set the last part.
    setattr(target, parts[-1], value)
    # Done.
    return updated


def batch_metrics(eliminations: pd.DataFrame, players: pd.DataFrame, games: pd.DataFrame) -> dict:
    """The headline numbers for a batch of games."""
    # Method shares.
    shares = eliminations["method"].value_counts(normalize=True) if len(eliminations) else pd.Series(dtype=float)
    # Assemble.
    return {
        "games": int(len(games)),
        "victor_rate": float(games["winner_id"].notna().mean()) if len(games) else 0.0,
        "mean_days": float(games["days"].mean()) if len(games) else 0.0,
        "player_vs_player_share": float(shares.get("player_vs_player", 0.0)),
        "natural_share": float(shares.get("natural_causes", 0.0)),
        "gamemaker_share": float(shares.get("gamemaker", 0.0)),
        "eliminations_per_point": float((players["kills"] / players["training_score"]).mean()) if len(players) else 0.0,
        "mean_interventions": float(games["interventions"].mean()) if len(games) else 0.0,
        "eliminations_per_game": float(len(eliminations) / max(1, len(games))),
    }


@dataclass
class SweepConfig:
    """What to sweep and how hard."""

    # A label for the run folder.
    name: str
    # The config field to change, dotted for nested fields.
    parameter: str
    # The values to try.
    values: list
    # Games per value.
    games_per_value: int = 50
    # CPU cores.
    workers: int = 1
    # Base seed (game N of every value uses seed + N, so values are compared on the same games).
    seed: int = 1000
    # Whether to collect behaviour telemetry (slower, but gives the behaviour charts).
    telemetry: bool = True
    # Where run folders go.
    results_dir: str = "results"


class Sweep:
    """Runs one batch per value and writes the run folder."""

    def __init__(self, config: SimulationConfig, sweep: SweepConfig, scenario: Scenario | None = None) -> None:
        """Remember the base config and the sweep settings."""
        # Base settings.
        self.config = config
        # Sweep settings.
        self.sweep = sweep
        # Optional custom setup.
        self.scenario = scenario
        # Results per value.
        self.rows: list[dict] = []
        # Telemetry per value.
        self.summaries: list[dict] = []
        # The run folder, created by run().
        self.run_dir: Path | None = None
        # Stop flag.
        self._stop = False

    def run(
        self, on_value: Callable[[dict], None] | None = None, on_progress: Callable[[int, int], None] | None = None
    ) -> Path:
        """Play every value and write the folder."""
        # Reset.
        self._stop = False
        # The folder.
        self.run_dir = make_run_dir(self.sweep.results_dir, self.sweep.name)
        # Save the setup.
        (self.run_dir / "config.json").write_text(
            json.dumps({"base_config": self.config.to_dict(), "sweep": asdict(self.sweep)}, indent=2, default=str)
        )
        # Each value.
        for index, value in enumerate(self.sweep.values):
            # Stop if asked.
            if self._stop:
                break
            # The config for this value.
            config = set_field(self.config, self.sweep.parameter, value)
            # Seeded so values share games.
            config.seed = self.sweep.seed
            # Play; the runner writes its CSVs under batches/<value>/ so every batch can be inspected.
            runner = Runner(
                config,
                num_games=self.sweep.games_per_value,
                workers=self.sweep.workers,
                output_dir=self.run_dir / "batches" / str(value),
                collect_telemetry=self.sweep.telemetry,
                scenario=self.scenario,
            )
            # Tables.
            eliminations, players, games = runner.run(show_progress=False)
            # Metrics.
            row = {"value": value, **batch_metrics(eliminations, players, games)}
            # Telemetry.
            summary = runner.telemetry_summary
            # Behaviour numbers into the row.
            if summary is not None:
                row.update(
                    {
                        "entropy": summary["entropy"],
                        "mean_survival_ticks": summary["mean_survival_ticks"],
                        "kill_rate": summary["kill_rate"],
                    }
                )
                self.summaries.append(summary)
            # Keep.
            self.rows.append(row)
            # Report.
            if on_value is not None:
                on_value(row)
            if on_progress is not None:
                on_progress(index + 1, len(self.sweep.values))
        # Write the results.
        self.write()
        # Done.
        return self.run_dir

    def stop(self) -> None:
        """Ask a running sweep to stop after the current value."""
        # Flag.
        self._stop = True

    def write(self) -> None:
        """Write results.csv, summary.json and one plot per metric."""
        # Need a folder.
        if self.run_dir is None or not self.rows:
            return
        # Table.
        table = pd.DataFrame(self.rows)
        # CSV.
        table.to_csv(self.run_dir / "results.csv", index=False)
        # JSON with telemetry.
        (self.run_dir / "summary.json").write_text(
            json.dumps({"rows": self.rows, "telemetry": self.summaries}, indent=2, default=str)
        )
        # One plot per metric.
        xs = [str(row["value"]) for row in self.rows]
        # Metrics are every numeric column except the value itself.
        for column in table.columns:
            # Skip the swept value.
            if column == "value" or not np.issubdtype(table[column].dtype, np.number):
                continue
            # Draw.
            plots.curves(
                xs,
                {column: table[column].tolist()},
                f"{column} vs {self.sweep.parameter}",
                self.sweep.parameter,
                column,
                self.run_dir / "plots" / f"{column}.png",
            )
        # Behaviour charts for the last value (and a stacked area across values).
        if self.summaries:
            plots.stacked_area_over_training(
                self.summaries, self.run_dir / "plots" / "action_distribution_by_value.png", self.sweep.parameter
            )
            plots.behaviour_plots(BehaviorTelemetry.merge(self.summaries), self.run_dir / "plots" / "behaviour")
