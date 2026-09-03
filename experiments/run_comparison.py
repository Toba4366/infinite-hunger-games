"""Train every method under the same budget, then run the champions' tournament.

python experiments/run_comparison.py --iterations 20 --workers 4
python experiments/run_comparison.py --methods imitation,ppo,neat --iterations 30 --games 75
python experiments/run_comparison.py --sizes 16,64x32 --methods ppo        # network size comparison
python experiments/run_comparison.py --initializers xavier_uniform,he_uniform,zeros --methods ppo
"""

# Command-line parsing.
import argparse
import ast

# Make the package importable when run from the repo root.
import sys
from pathlib import Path

# The repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Settings, the comparison.
from hunger_games.config import SimulationConfig  # noqa: E402
from hunger_games.research.comparison import METHODS, ComparisonConfig, MethodComparison, Variant  # noqa: E402


def parse_layers(text: str) -> tuple[int, ...]:
    """'64x32' -> (64, 32)."""
    return tuple(int(part) for part in text.split("x") if part)


def parse_settings(specs: list[str]) -> list[tuple[str, object]]:
    """Turn `["learning_rate=1e-3,3e-3", "entropy_bonus=0.01"]` into `[("learning_rate", 0.001), ...]`."""
    # Every (name, value) pair, in the order given.
    pairs: list[tuple[str, object]] = []
    for spec in specs:
        # The field name and its comma-separated values.
        name, _, values = spec.partition("=")
        if not name or not values:
            raise SystemExit(f"--set expects NAME=V1,V2,... (got {spec!r})")
        for raw in values.split(","):
            # Numbers, booleans, tuples and lists parse as Python literals; anything else stays a string.
            try:
                value = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                value = raw
            pairs.append((name.strip(), value))
    return pairs


def check_known(specs: list[str], methods: list[str]) -> None:
    """Stop with an error when a setting name matches no chosen method, instead of silently doing nothing."""
    for name, _ in parse_settings(specs):
        # An optional "method." prefix narrows the candidates.
        method, _, field = name.rpartition(".")
        # A prefixed method must be one of the methods being run, or the setting would apply to nothing.
        if method and method not in methods:
            raise SystemExit(f"{name}: {method!r} is not one of the methods being run ({', '.join(methods)})")
        candidates = [method] if method else methods
        if not any(m in METHODS and hasattr(METHODS[m][1](), field) for m in candidates):
            raise SystemExit(f"{name}: none of {', '.join(candidates)} has a setting called {field!r}")


def apply_side_settings(variants: list[Variant], cold_specs: list[str], warm_specs: list[str]) -> list[Variant]:
    """Apply `--cold-set` to variants without a warm start and `--warm-set` to those with one (imitation untouched).

    The sensitivity sweep showed cold starts want different settings (more episodes per update, a smaller entropy
    bonus) from warm starts, so a cold-against-warm comparison must be allowed to give each side its own.
    """
    for variant in variants:
        # Imitation has no warm or cold side.
        if variant.method == "imitation":
            continue
        specs = warm_specs if variant.warm_from else cold_specs
        for name, value in parse_settings(specs):
            # "reinforce.learning_rate" applies to REINFORCE variants only; "learning_rate" to every method that has it.
            method, _, field = name.rpartition(".")
            if method and method != variant.method:
                continue
            if variant.settings is None:
                variant.settings = METHODS[variant.method][1]()
            if hasattr(variant.settings, field):
                setattr(variant.settings, field, value)
    return variants


def main() -> None:
    """Parse arguments, compare, save."""
    parser = argparse.ArgumentParser(description="Compare training methods")
    parser.add_argument(
        "--methods", default="imitation,genetic,neat,reinforce,ppo", help="comma-separated method names"
    )
    parser.add_argument("--iterations", type=int, default=20, help="iterations per variant")
    parser.add_argument("--time-budget", type=float, default=None, help="seconds per variant (optional)")
    parser.add_argument("--games", type=int, default=75, help="tournament games per champion")
    parser.add_argument(
        "--until-win",
        type=float,
        default=0.5,
        help="stop a variant once it wins this share of validation games over --window iterations (negative: never)",
    )
    parser.add_argument("--window", type=int, default=5, help="iterations averaged for the win criterion")
    parser.add_argument(
        "--extend-iterations",
        type=int,
        default=0,
        help="after every variant has had --iterations, keep training those short of the criterion for up to this many more",
    )
    parser.add_argument(
        "--extend-hours", type=float, default=None, help="wall-clock cap per variant for that extension"
    )
    parser.add_argument(
        "--save-replays-every",
        type=int,
        default=0,
        help="save a replay of every Nth iteration's training game per variant (0 = never)",
    )
    parser.add_argument(
        "--pairs",
        action="store_true",
        help="train a cold and a warm-started variant of every reward or evolution method (requires imitation in --methods)",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--curriculum",
        nargs="?",
        const="opponents",
        default=None,
        choices=["opponents", "lessons"],
        help="train the reward methods with a curriculum: 'opponents' (1, 3, 7, 11, 23; the default when the flag "
        "is given bare) or 'lessons' (survive, survive the rules, beat 1 to 23, generalise)",
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
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=V1,V2,...",
        help="sweep one trainer setting: one variant per value for every method whose settings have that field "
        "(imitation keeps a single plain variant so the others can warm-start from it); repeatable",
    )
    parser.add_argument(
        "--cold-set",
        action="append",
        default=[],
        metavar="[METHOD.]NAME=VALUE",
        help="set one trainer setting on every cold-started variant (no warm start; imitation excluded), optionally "
        "for one method only, e.g. --cold-set episodes_per_epoch=16 --cold-set reinforce.learning_rate=3e-3; repeatable",
    )
    parser.add_argument(
        "--warm-set",
        action="append",
        default=[],
        metavar="[METHOD.]NAME=VALUE",
        help="the same for every warm-started variant; repeatable",
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
    # A --set field no method's settings have would otherwise create no variants and fail silently.
    check_known(args.set, methods)
    check_known(args.cold_set, [m for m in methods if m != "imitation"])
    check_known(args.warm_set, [m for m in methods if m != "imitation"])
    # Imitation variants must come first so the others can warm-start from them.
    ordered = sorted(methods, key=lambda m: m != "imitation")
    for method in ordered:
        curriculum = args.curriculum if method in ("reinforce", "ppo", "genetic", "neat") else False
        curriculum = curriculum or False
        # A warm start comes from the imitation variant with the same size or initializer suffix.
        can_warm = args.warm and method not in ("imitation", "neat") and "imitation" in methods
        if args.set:
            # One variant per (setting, value) for the methods whose settings dataclass has that field.
            if method == "imitation":
                variants.append(Variant(method, method, curriculum=curriculum))
            for name, value in parse_settings(args.set):
                settings = METHODS[method][1]()
                if not hasattr(settings, name):
                    continue
                setattr(settings, name, value)
                variants.append(
                    Variant(
                        f"{method}_{name}_{value}",
                        method,
                        settings,
                        curriculum=curriculum,
                        warm_from="imitation" if can_warm else None,
                    )
                )
        elif args.sizes:
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
        elif args.pairs and method not in ("imitation", "neat") and "imitation" in methods:
            variants.append(Variant(f"{method}_cold", method, curriculum=curriculum))
            variants.append(Variant(f"{method}_warm", method, curriculum=curriculum, warm_from="imitation"))
        else:
            variants.append(Variant(method, method, curriculum=curriculum, warm_from="imitation" if can_warm else None))
    # Each side of a cold-against-warm comparison may need its own settings.
    apply_side_settings(variants, args.cold_set, args.warm_set)
    # Run.
    comparison = MethodComparison(
        config,
        ComparisonConfig(
            name=args.name,
            iterations=args.iterations,
            time_budget=args.time_budget,
            until_win_rate=None if args.until_win < 0 else args.until_win,
            win_window=args.window,
            extended_iterations=args.extend_iterations,
            extended_time_budget=None if args.extend_hours is None else args.extend_hours * 3600,
            save_replays_every=args.save_replays_every,
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
