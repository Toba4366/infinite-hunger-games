"""training/runs.py - write a training run to a results folder.

Both trainers produce a history and behaviour telemetry. `save_run` turns
that into the same folder layout a research sweep uses:

    results/<name>_<timestamp>/
        config.json    simulation config, trainer config, method
        history.json   one row per generation or epoch
        champion.json  the best genome (or policy) in the champion file format
        plots/         one PNG per chart, plus the growing-curve GIF
"""

# JSON.
import json

# Paths.
from pathlib import Path

# Run folders and plots.
from hunger_games.research.experiments import make_run_dir
from hunger_games.research.plots import training_run_plots


def save_run(trainer, method: str, name: str, results_dir: str | Path = "results") -> Path:
    """Write config, history, champion and plots for a GeneticTrainer, ReinforceTrainer or ImitationTrainer."""
    # Folder.
    folder = make_run_dir(results_dir, name)
    # The trainer's own settings dataclass.
    from dataclasses import asdict

    trainer_config = asdict(trainer.settings)
    # Config.
    (folder / "config.json").write_text(
        json.dumps(
            {"method": method, "simulation": trainer.config.to_dict(), "trainer": trainer_config}, indent=2, default=str
        )
    )
    # History.
    rows = trainer.history_rows()
    (folder / "history.json").write_text(json.dumps(rows, indent=2, default=str))
    # Champion (every trainer writes the same file shape).
    if trainer.champion is not None:
        trainer.save_champion(folder / "champion.json")
    # Telemetry per step.
    summaries = [stats.telemetry for stats in trainer.history if stats.telemetry]
    # Plots.
    training_run_plots(rows, summaries, folder / "plots", method)
    # Done.
    return folder
