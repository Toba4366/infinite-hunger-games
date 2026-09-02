"""Train the neural brain by policy gradient (REINFORCE with baseline) and write a results folder.

python experiments/run_rl.py --epochs 30 --episodes 4 --learners 6 --workers 4
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
from hunger_games.training import ReinforceTrainer, RLConfig, save_run  # noqa: E402


def main() -> None:
    """Parse arguments, train, save."""
    # Arguments.
    parser = argparse.ArgumentParser(description="Policy-gradient training run")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=4, help="games per epoch")
    parser.add_argument("--learners", type=int, default=6, help="learner tributes per game")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--entropy", type=float, default=0.01, help="entropy bonus")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", default="16", help="hidden layer widths, e.g. 32,16")
    parser.add_argument("--activation", default="tanh")
    parser.add_argument("--initializer", default="xavier_uniform")
    parser.add_argument("--opponents", default="voting", help="brain the non-learner tributes use")
    parser.add_argument("--size", type=int, default=120)
    parser.add_argument("--days", type=int, default=SimulationConfig.max_days)
    parser.add_argument("--name", default="rl")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    # Configs.
    neural = NeuralConfig(
        hidden_layers=tuple(int(w) for w in args.hidden.split(",") if w),
        activation=args.activation,
        initializer=args.initializer,
    )
    config = SimulationConfig(
        width=args.size, height=args.size, max_days=args.days, seed=args.seed, neural=neural, brain_name=args.opponents
    )
    rl = RLConfig(
        epochs=args.epochs,
        episodes_per_epoch=args.episodes,
        learners_per_game=args.learners,
        learning_rate=args.lr,
        entropy_bonus=args.entropy,
        workers=args.workers,
        seed=args.seed,
    )
    # Train.
    trainer = ReinforceTrainer(config, rl)
    trainer.run(
        on_epoch=lambda e: print(
            f"epoch {e.epoch:3d}  ploss {e.policy_loss:.3f}  vloss {e.value_loss:.3f}  H {e.entropy:.2f}  train {e.train_return:.2f}  val {e.val_return:.2f}  surv {e.train_survival:.0f}  win {e.win_rate:.2f}  {e.seconds:.1f}s"
        )
    )
    # Save.
    folder = save_run(trainer, "reinforce", args.name, args.results)
    print(f"saved to {folder}")


# Needed for multiprocessing on macOS.
if __name__ == "__main__":
    main()
