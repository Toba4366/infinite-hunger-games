"""Evolve brains with the genetic algorithm and write a results folder.

python experiments/run_ga.py --brain neural --population 48 --generations 20 --workers 4
"""

# Command-line parsing.
import argparse

# Make the package importable when run from the repo root.
import sys
from pathlib import Path

# The repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Settings and the trainer.
from hunger_games.config import NeuralConfig, SimulationConfig  # noqa: E402
from hunger_games.training import GeneticTrainer, TrainingConfig, save_run  # noqa: E402


def main() -> None:
    """Parse arguments, train, save."""
    # Arguments.
    parser = argparse.ArgumentParser(description="Genetic-algorithm training run")
    parser.add_argument("--brain", default="neural", choices=["neural", "voting"], help="which brain to evolve")
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=2, help="games per genome per generation")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", default="16", help="hidden layer widths, e.g. 32,16")
    parser.add_argument("--activation", default="tanh")
    parser.add_argument("--initializer", default="xavier_uniform")
    parser.add_argument("--size", type=int, default=120, help="arena size")
    parser.add_argument("--days", type=int, default=SimulationConfig.max_days)
    parser.add_argument("--name", default="ga")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    # Configs.
    neural = NeuralConfig(
        hidden_layers=tuple(int(w) for w in args.hidden.split(",") if w),
        activation=args.activation,
        initializer=args.initializer,
    )
    config = SimulationConfig(width=args.size, height=args.size, max_days=args.days, seed=args.seed, neural=neural)
    training = TrainingConfig(
        brain_name=args.brain,
        population_size=args.population,
        generations=args.generations,
        rounds_per_generation=args.rounds,
        workers=args.workers,
        seed=args.seed,
    )
    # Train.
    trainer = GeneticTrainer(config, training)
    trainer.run(
        on_generation=lambda s: print(
            f"gen {s.generation:3d}  best {s.best_fitness:.3f}  mean {s.mean_fitness:.3f}  val {s.val_fitness:.3f}  {s.seconds:.1f}s"
        )
    )
    # Save.
    folder = save_run(trainer, "genetic", args.name, args.results)
    print(f"saved to {folder}")


# Needed for multiprocessing on macOS.
if __name__ == "__main__":
    main()
