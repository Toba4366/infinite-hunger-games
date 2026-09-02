"""runner.py - the "infinite" part: run many games and collect spreadsheets.

Chapter 3 wanted hundreds of games to find reliable trends. `Runner` plays
as many as you ask for, optionally across several CPU cores, and writes
three CSV files: one row per elimination, one row per player-game, and one
row per game.
"""

# Parallel execution across CPU cores.
# JSON for the telemetry file.
import json
from concurrent.futures import ProcessPoolExecutor

# Filesystem paths.
from pathlib import Path

# pandas turns lists of dictionaries into tables and writes CSV.
import pandas as pd

# The settings.
from hunger_games.config import SimulationConfig

# The single-game runner.
from hunger_games.game import Game

# The result type.
from hunger_games.records import GameResult

# Behaviour measurement.
from hunger_games.research.telemetry import BehaviorTelemetry

# Custom setups.
from hunger_games.scenario import Scenario


def run_single_game(
    config: SimulationConfig, game_id: int, collect_telemetry: bool = False, scenario: Scenario | None = None
) -> GameResult:
    """Play one game to completion (a top-level function so it can run in a worker process)."""
    # Build.
    game = Game(config, game_id, scenario=scenario)
    # Optionally measure behaviour.
    telemetry = BehaviorTelemetry(game.arena.width, game.arena.height).attach(game) if collect_telemetry else None
    # Run.
    result = game.run()
    # Attach the summary.
    if telemetry is not None:
        result.telemetry = telemetry.summary()
    # Done.
    return result


class Runner:
    """Plays a batch of games and saves the results as CSV."""

    def __init__(
        self,
        config: SimulationConfig,
        num_games: int = 100,
        workers: int = 1,
        output_dir: str | Path = "output",
        collect_telemetry: bool = False,
        scenario: Scenario | None = None,
    ) -> None:
        """Remember what to run and where to put the results."""
        # Whether to measure behaviour in every game.
        self.collect_telemetry = collect_telemetry
        # Optional custom setup every game uses.
        self.scenario = scenario
        # The merged telemetry summary after run(), if collected.
        self.telemetry_summary: dict | None = None
        # The shared settings for every game.
        self.config = config
        # How many games to play.
        self.num_games = num_games
        # How many CPU cores to use.
        self.workers = max(1, workers)
        # Where the CSV files go.
        self.output_dir = Path(output_dir)

    def run(self, show_progress: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Play every game and return (eliminations, players, games) tables; gifts are saved to CSV too."""
        # Collect results here.
        results: list[GameResult] = []
        # Multi-core path.
        if self.workers > 1:
            # A pool of worker processes.
            with ProcessPoolExecutor(max_workers=self.workers) as pool:
                # Hand each game id to a worker, keeping results in order.
                for result in pool.map(
                    run_single_game,
                    [self.config] * self.num_games,
                    range(self.num_games),
                    [self.collect_telemetry] * self.num_games,
                    [self.scenario] * self.num_games,
                ):
                    # Store the finished game.
                    results.append(result)
                    # Show progress.
                    self._progress(len(results), show_progress)
        # Single-core path.
        else:
            # Play the games one after another.
            for game_id in range(self.num_games):
                # Store the finished game.
                results.append(run_single_game(self.config, game_id, self.collect_telemetry, self.scenario))
                # Show progress.
                self._progress(len(results), show_progress)
        # Finish the progress line.
        if show_progress:
            print()
        # One table row per elimination across all games.
        eliminations = pd.DataFrame([row for result in results for row in result.elimination_rows()])
        # One table row per player per game.
        players = pd.DataFrame([row for result in results for row in result.player_rows()])
        # One table row per game.
        games = pd.DataFrame(
            [
                {
                    "game_id": result.game_id,
                    "seed": result.seed,
                    "days": result.days,
                    "ticks": result.ticks,
                    "winner_id": result.winner_id,
                    "winner_name": result.winner_name,
                    "interventions": result.interventions,
                }
                for result in results
            ]
        )
        # One table row per sponsor gift across all games.
        gifts = pd.DataFrame([row for result in results for row in result.gifts])
        # Merge the behaviour telemetry, if collected.
        if self.collect_telemetry:
            self.telemetry_summary = BehaviorTelemetry.merge(
                [result.telemetry for result in results if result.telemetry is not None]
            )
        # Write the CSV files.
        self.save(eliminations, players, games, gifts)
        # Hand the three main tables back for further analysis.
        return eliminations, players, games

    def _progress(self, done: int, show: bool) -> None:
        """Print a one-line progress counter that overwrites itself."""
        # Only if asked.
        if show:
            # `\r` returns to the start of the line so the counter updates in place.
            print(f"\rgames finished: {done}/{self.num_games}", end="", flush=True)

    def save(
        self, eliminations: pd.DataFrame, players: pd.DataFrame, games: pd.DataFrame, gifts: pd.DataFrame | None = None
    ) -> None:
        """Write the tables to CSV files in the output directory."""
        # Make sure the directory exists.
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Eliminations table.
        eliminations.to_csv(self.output_dir / "eliminations.csv", index=False)
        # Players table.
        players.to_csv(self.output_dir / "players.csv", index=False)
        # Games table.
        games.to_csv(self.output_dir / "games.csv", index=False)
        # Gifts table (may be empty when sponsors are off).
        if gifts is not None:
            gifts.to_csv(self.output_dir / "gifts.csv", index=False)
        # Telemetry, if collected.
        if self.telemetry_summary is not None:
            (self.output_dir / "telemetry.json").write_text(json.dumps(self.telemetry_summary))
