"""Train every method under the same budget, then run the champions' tournament.

python experiments/run_comparison.py --iterations 20 --workers 4
python experiments/run_comparison.py --methods imitation,ppo,neat --iterations 30 --games 75
python experiments/run_comparison.py --sizes 16,64x32 --methods ppo        # network size comparison
python experiments/run_comparison.py --initializers xavier_uniform,he_uniform,zeros --methods ppo
"""

# Command-line parsing.
import argparse

# Make the package importable when run from the repo root.
import sys
from pathlib import Path

# The repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Settings, the comparison.
from hunger_games.config import SimulationConfig  # noqa: E402
from hunger_games.research.comparison import ComparisonConfig, MethodComparison, Variant  # noqa: E402


def parse_layers(text: str) -> tuple[int, ...]:
    """'64x32' -> (64, 32)."""
    return tuple(int(part) for part in text.split("x") if part)


def main() -> None:
    """Parse arguments, compare, save."""
    parser = argparse.ArgumentParser(description="Compare training methods")
    parser.add_argument(
        "--methods", default="imitation,genetic,neat,reinforce,ppo", help="comma-separated method names"
    )
    parser.add_argument("--iterations", type=int, default=20, help="iterations per variant")
    parser.add_argument("--time-budget", type=float, default=None, help="seconds per variant (optional)")
    parser.add_argument("--games", type=int, default=75, help="tournament games per champion")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--curriculum", action="store_true", help="train the reward methods with the opponent curriculum"
    )
    parser.add_argument(
        "--warm",
        action="store_true",
        help="warm-start the neural methods from the imitation champion (requires imitation in --methods)",
    )
    parser.add_argument("--sizes", default=None, help="hidden-layer variants to compare, e.g. 16,64x32,128x64")
    parser.add_argument(
        "--initializers", default=None, help="initializer variants to compare, e.g. xavier_uniform,he_uniform,zeros"
    )
    parser.add_argument("--size", type=int, default=120, help="arena size")
    parser.add_argument("--days", type=int, default=SimulationConfig.max_days)
    parser.add_argument("--name", default="comparison")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    # Base config.
    config = SimulationConfig(width=args.size, height=args.size, max_days=args.days)
    # Variants: methods, crossed with sizes or initializers when asked.
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    variants = []
    # Imitation variants must come first so the others can warm-start from them.
    ordered = sorted(methods, key=lambda m: m != "imitation")
    for method in ordered:
        curriculum = args.curriculum and method in ("reinforce", "ppo", "genetic", "neat")
        # A warm start comes from the imitation variant with the same size or initializer suffix.
        can_warm = args.warm and method not in ("imitation", "neat") and "imitation" in methods
        if args.sizes:
            for size in args.sizes.split(","):
                variants.append(
                    Variant(
                        f"{method}_{size}",
                        method,
                        config_overrides={"neural.hidden_layers": parse_layers(size)},
                        curriculum=curriculum,
                        warm_from=f"imitation_{size}" if can_warm else None,
                    )
                )
        elif args.initializers:
            for init in args.initializers.split(","):
                variants.append(
                    Variant(
                        f"{method}_{init}",
                        method,
                        config_overrides={"neural.initializer": init},
                        curriculum=curriculum,
                        warm_from=f"imitation_{init}" if can_warm else None,
                    )
                )
        else:
            variants.append(Variant(method, method, curriculum=curriculum, warm_from="imitation" if can_warm else None))
    # Run.
    comparison = MethodComparison(
        config,
        ComparisonConfig(
            name=args.name,
            iterations=args.iterations,
            time_budget=args.time_budget,
            tournament_games=args.games,
            workers=args.workers,
            seed=args.seed,
            results_dir=args.results,
        ),
        variants,
    )
    folder = comparison.run(on_progress=lambda name, what: print(f"{name}: {what}"))
    print(comparison.table().to_string(index=False))
    print(f"saved to {folder}")


# Needed for multiprocessing on macOS.
if __name__ == "__main__":
    main()
