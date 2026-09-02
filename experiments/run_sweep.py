"""Sweep one config value and write a results folder with one chart per metric.

python experiments/run_sweep.py --parameter chaos --values 0,0.25,0.5,0.75,1 --games 50 --workers 4
python experiments/run_sweep.py --parameter terrain.water_threshold --values 0.1,0.25,0.4
python experiments/run_sweep.py --parameter gamemaker_enabled --values false,true
"""

# Command-line parsing.
import argparse

# Make the package importable when run from the repo root.
import sys
from pathlib import Path

# The repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Settings, the sweep.
from hunger_games.config import SimulationConfig  # noqa: E402
from hunger_games.research.experiments import Sweep, SweepConfig  # noqa: E402


def parse_value(text: str):
    """Turn a command-line token into a bool, int, float or string."""
    # Booleans.
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    # Numbers.
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def main() -> None:
    """Parse arguments, sweep, save."""
    # Arguments.
    parser = argparse.ArgumentParser(description="Parameter sweep")
    parser.add_argument("--parameter", required=True, help="config field, dotted for nested (terrain.water_threshold)")
    parser.add_argument("--values", required=True, help="comma-separated values")
    parser.add_argument("--games", type=int, default=50, help="games per value")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--no-telemetry", action="store_true")
    parser.add_argument("--size", type=int, default=120)
    parser.add_argument("--name", default=None)
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    # Values.
    values = [parse_value(v) for v in args.values.split(",")]
    # Configs.
    config = SimulationConfig(width=args.size, height=args.size)
    sweep = SweepConfig(
        name=args.name or args.parameter.replace(".", "_"),
        parameter=args.parameter,
        values=values,
        games_per_value=args.games,
        workers=args.workers,
        seed=args.seed,
        telemetry=not args.no_telemetry,
        results_dir=args.results,
    )
    # Run.
    folder = Sweep(config, sweep).run(
        on_value=lambda row: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
    )
    print(f"saved to {folder}")


# Needed for multiprocessing on macOS.
if __name__ == "__main__":
    main()
