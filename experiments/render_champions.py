"""Render one tournament game per champion of a finished comparison run as a GIF.

A comparison run folder holds every variant's champion under `runs/<variant>_<stamp>/champion.json`.
This script rebuilds each champion, plays the first tournament game (seed 50000, the same seed
every champion faced in the tournament) with the champion's copies in the learner slots against
voting tributes, records every tick, and writes `gifs/<variant>.gif` plus `gifs/index.md` with a
caption per GIF: whether a learner copy won, how long the copies survived, how many kills.

Usage:
    python experiments/render_champions.py results/full_methods_20260903_025758
    python experiments/render_champions.py <run folder> --seed 50003 --step 3 --fps 12

Every GIF is a real game, not a montage: the gold-starred tributes are the champion's copies.
"""

# Command-line flags.
import argparse

# The run's config and champion files are JSON.
import json

# Paths.
import sys
from pathlib import Path

# Arrays.
import numpy as np

# The project root, so the script runs from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hunger_games.config import NeuralConfig, SimulationConfig  # noqa: E402
from hunger_games.renderer import export_recording_gif  # noqa: E402
from hunger_games.training.common import LearnerSpec, learner_ids  # noqa: E402
from hunger_games.training.reinforce import play_rl_episode  # noqa: E402


def load_spec(champion_file: Path) -> LearnerSpec:
    """Turn a champion.json into the LearnerSpec the episode player expects."""
    # Parse.
    data = json.loads(champion_file.read_text())
    # A NEAT genome is a dictionary; a neural genome is a flat list of weights.
    if isinstance(data["genome"], dict):
        return LearnerSpec("neat", data["genome"])
    # The neural architecture the weights belong to.
    neural = data.get("neural")
    config = NeuralConfig(**{**neural, "hidden_layers": tuple(neural["hidden_layers"])}) if neural else NeuralConfig()
    return LearnerSpec("neural", np.asarray(data["genome"], dtype=float), config)


def champions_of(run_folder: Path) -> list[tuple[str, Path]]:
    """Every (variant name, champion file) in the run, in the order the run listed its variants."""
    # The run's configuration names the variants in order.
    config = json.loads((run_folder / "config.json").read_text())
    found = []
    for variant in config["variants"]:
        # Keep the name to a single path component, as the run folders themselves do.
        name = Path(str(variant["name"])).name
        # The final run folder of that variant (the extension phase writes under `runs/` too).
        candidates = sorted((run_folder / "runs").glob(f"{name}_*/champion.json"))
        if candidates:
            found.append((name, candidates[-1]))
    return found


def render(run_folder: str | Path, seed: int = 50000, step: int = 2, fps: int = 15, max_frames: int = 600) -> Path:
    """Play one tournament game per champion, write the GIFs and the index, return the gifs folder."""
    folder = Path(run_folder)
    # The base config and the tournament settings the run used.
    config_data = json.loads((folder / "config.json").read_text())
    base = SimulationConfig.from_dict(config_data["base_config"])
    comparison = config_data["comparison"]
    # The tournament plays on the base config with the comparison's seed.
    config = SimulationConfig(**{**base.to_dict_raw(), "seed": int(comparison["seed"])})
    learners = learner_ids(config.num_players, int(comparison["tournament_learners"]))
    # Where the GIFs go.
    target = folder / "gifs"
    target.mkdir(parents=True, exist_ok=True)
    lines = [f"# Champions in action: one tournament game each (seed {seed})", ""]
    lines.append("| variant | GIF | learner copy won | mean survival (ticks) | kills |")
    lines.append("| --- | --- | --- | --- | --- |")
    for name, champion_file in champions_of(folder):
        # Rebuild the champion and play the game greedily with a full recording.
        spec = load_spec(champion_file)
        result = play_rl_episode(config, None, spec, learners, seed, True, True)
        # Render.
        path = export_recording_gif(
            result["recording"], target / f"{name}.gif", fps=fps, step=step, max_frames=max_frames
        )
        # Caption numbers from the learner copies' outcomes.
        outcomes = list(result["outcomes"].values())
        survival = float(np.mean([o["survival"] for o in outcomes]))
        kills = float(np.sum([o["kills"] for o in outcomes]))
        won = "yes" if result.get("learner_won") else "no"
        lines.append(f"| {name} | [{path.name}]({path.name}) | {won} | {survival:.0f} | {kills:.0f} |")
        print(f"{name}: won={won} survival={survival:.0f} kills={kills:.0f} -> {path}")
    (target / "index.md").write_text("\n".join(lines) + "\n")
    return target


def main() -> None:
    """Parse the flags and render."""
    parser = argparse.ArgumentParser(description="One tournament-game GIF per champion of a comparison run.")
    parser.add_argument("run_folder", help="a results/<name>_<timestamp> folder written by run_comparison.py")
    parser.add_argument("--seed", type=int, default=50000, help="game seed (the tournament used 50000 + i)")
    parser.add_argument("--step", type=int, default=2, help="draw every Nth tick")
    parser.add_argument("--fps", type=int, default=15, help="frames per second")
    parser.add_argument("--max-frames", type=int, default=600, help="cap on frames per GIF")
    args = parser.parse_args()
    target = render(args.run_folder, args.seed, args.step, args.fps, args.max_frames)
    print(f"written to {target}")


if __name__ == "__main__":
    main()
