"""__main__.py - the command line: `python -m hunger_games <command>`.

python -m hunger_games watch                 # animate one game
python -m hunger_games watch --shape round   # the 75th-games arena
python -m hunger_games simulate --games 500  # play many, write CSVs
python -m hunger_games analyze               # chapter 3 charts
python -m hunger_games ui                    # the game makers' dashboard
"""

# Command-line parsing.
import argparse

# Everything the commands need.
from hunger_games.analysis import make_report
from hunger_games.config import ArenaShape, LayoutName, SimulationConfig
from hunger_games.game import Game
from hunger_games.renderer import Renderer
from hunger_games.runner import Runner


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """The settings every command shares. Defaults come from SimulationConfig so they never drift."""
    # The defaults, read from the config class itself.
    defaults = SimulationConfig()
    # Arena outline.
    parser.add_argument(
        "--shape", choices=[shape.value for shape in ArenaShape], default=defaults.shape.value, help="arena outline"
    )
    # Supply layout.
    parser.add_argument(
        "--layout", choices=[layout.value for layout in LayoutName], default=defaults.layout.value, help="supply layout"
    )
    # Randomness dial.
    parser.add_argument("--chaos", type=float, default=defaults.chaos, help="randomness from 0.0 to 1.0")
    # Seed.
    parser.add_argument("--seed", type=int, default=defaults.seed, help="random seed for repeatable games")
    # Grid size (square).
    parser.add_argument("--size", type=int, default=defaults.width, help="arena width and height in cells")
    # Number of tributes.
    parser.add_argument("--players", type=int, default=defaults.num_players, help="number of tributes")
    # Brain type.
    parser.add_argument("--brain", default=defaults.brain_name, help="brain type: voting, random or neural")
    # Game makers on/off as a pair of flags, defaulting to the config (on: the slow safe circle).
    parser.add_argument(
        "--gamemaker",
        action=argparse.BooleanOptionalAction,
        default=defaults.gamemaker_enabled,
        help="game maker interventions (the slowly shrinking circle); --no-gamemaker turns them off",
    )
    # Sponsors on/off as a pair of flags, defaulting to the config.
    parser.add_argument(
        "--sponsors",
        action=argparse.BooleanOptionalAction,
        default=defaults.sponsors_enabled,
        help="sponsor gifts; --no-sponsors turns them off",
    )
    # Maximum length.
    parser.add_argument("--days", type=int, default=defaults.max_days, help="maximum game length in days")


def config_from_arguments(args: argparse.Namespace) -> SimulationConfig:
    """Turn parsed command-line options into a `SimulationConfig`."""
    # Build the config field by field.
    return SimulationConfig(
        width=args.size,
        height=args.size,
        shape=ArenaShape(args.shape),
        layout=LayoutName(args.layout),
        num_players=args.players,
        chaos=args.chaos,
        seed=args.seed,
        brain_name=args.brain,
        gamemaker_enabled=args.gamemaker,
        sponsors_enabled=args.sponsors,
        max_days=args.days,
    )


def main() -> None:
    """Parse the command line and run the chosen command."""
    # The top-level parser.
    parser = argparse.ArgumentParser(prog="hunger_games", description="Infinite Hunger Games simulator")
    # One sub-command per verb.
    commands = parser.add_subparsers(dest="command", required=True)
    # `watch`: animate one game.
    watch = commands.add_parser("watch", help="animate a single game on screen")
    # Shared settings.
    add_config_arguments(watch)
    # Speed.
    watch.add_argument("--speed", type=int, default=1, help="simulation ticks per animation frame")
    # Optional file output instead of a window.
    watch.add_argument("--save", default=None, help="write a GIF/MP4 instead of opening a window")
    # `simulate`: play many games.
    simulate = commands.add_parser("simulate", help="play many games and write CSV results")
    # Shared settings.
    add_config_arguments(simulate)
    # How many games.
    simulate.add_argument("--games", type=int, default=100, help="number of games to play")
    # How many cores.
    simulate.add_argument("--workers", type=int, default=1, help="CPU cores to use")
    # Where to write.
    simulate.add_argument("--output", default="output", help="folder for the CSV files")
    # `analyze`: draw the charts.
    analyze = commands.add_parser("analyze", help="draw the chapter 3 charts from saved results")
    # Where to read.
    analyze.add_argument("--output", default="output", help="folder holding the CSV files")
    # Pop a window too.
    analyze.add_argument("--show", action="store_true", help="open the chart window")
    # `ui`: the dashboard.
    commands.add_parser("ui", help="open the game makers' dashboard")
    # Parse.
    args = parser.parse_args()
    # Dispatch.
    if args.command == "watch":
        # Build one game.
        game = Game(config_from_arguments(args))
        # Wrap it in a renderer.
        renderer = Renderer(game, ticks_per_frame=args.speed)
        # Either save or show.
        if args.save:
            renderer.save(args.save)
        else:
            renderer.show()
    elif args.command == "simulate":
        # Build the batch runner.
        runner = Runner(config_from_arguments(args), num_games=args.games, workers=args.workers, output_dir=args.output)
        # Play.
        eliminations, players, games = runner.run()
        # Report where things went.
        print(
            f"wrote {len(eliminations)} eliminations, {len(players)} player rows, {len(games)} games to {args.output}/"
        )
    elif args.command == "analyze":
        # Draw the charts.
        report = make_report(args.output, show=args.show)
        # Report where the picture went.
        print(f"report saved to {report}")
    elif args.command == "ui":
        # Import here so the other commands never need Dear PyGui.
        from hunger_games.ui import launch

        # Open the window.
        launch()


# Run `main()` when executed as `python -m hunger_games`.
if __name__ == "__main__":
    main()
