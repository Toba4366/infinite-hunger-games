"""research/comparison.py - train every method under the same budget, then let the champions fight.

The research question the project exists to answer: which way of training
a brain makes the most sense for the Hunger Games? This module trains a
list of variants (a method plus its settings, and optionally a network
size or initializer to compare) for the same number of iterations or the
same wall-clock budget, records the shared learning curves, and then runs
a tournament in which every champion plays the same seeded games against
voting opponents. It writes one folder with a results table, one PNG per
chart, a LaTeX table for a paper, and a generated report that ranks the
methods and explains the trade-offs.
"""

# Deep copies of settings.
import copy

# JSON.
import json

# Timing.
import time

# Silencing a pandas deprecation note.
import warnings
from collections.abc import Callable

# Settings.
from dataclasses import asdict, dataclass, field

# Paths.
from pathlib import Path

# Type hints.
from typing import Any

# numpy.
import numpy as np

# pandas for the tables.
import pandas as pd

# Settings.
from hunger_games.config import SimulationConfig

# Plots and run folders.
from hunger_games.research import plots
from hunger_games.research.experiments import make_run_dir

# Trainers and the shared pieces.
from hunger_games.training.common import Curriculum, CurriculumConfig, LearnerSpec, learner_ids
from hunger_games.training.genetic import GeneticTrainer, TrainingConfig
from hunger_games.training.imitation import ImitationConfig, ImitationTrainer
from hunger_games.training.neat import NeatTrainer, NeatTrainerConfig
from hunger_games.training.ppo import PPOConfig, PPOTrainer
from hunger_games.training.reinforce import ReinforceTrainer, RLConfig, _run_episode_job
from hunger_games.training.runs import save_run

# The trainer class and default settings for every method name.
METHODS: dict[str, tuple[type, Callable[[], Any]]] = {
    "imitation": (ImitationTrainer, ImitationConfig),
    "genetic": (GeneticTrainer, TrainingConfig),
    "neat": (NeatTrainer, NeatTrainerConfig),
    "reinforce": (ReinforceTrainer, RLConfig),
    "ppo": (PPOTrainer, PPOConfig),
}

# One-line implementation notes per method, used in the generated report.
METHOD_NOTES = {
    "imitation": "Supervised learning: copies the voting brain. Needs a teacher; cannot exceed it.",
    "genetic": "Evolves weights only. No gradients; simple; scales poorly with weight count.",
    "neat": "Evolves weights and structure with species. More machinery; small networks; slow per generation.",
    "reinforce": "Policy gradient with a value baseline. One pass per batch; high variance.",
    "ppo": "Clipped policy gradient with GAE and several passes per batch. Most stable of the reward methods.",
}


@dataclass
class Variant:
    """One thing to compare: a method, its settings, and the simulation it trains in."""

    # A label for tables and plots.
    name: str
    # Method name from METHODS.
    method: str
    # The method's settings dataclass (defaults if None).
    settings: Any = None
    # Overrides on the simulation config (dotted keys allowed), e.g. {"neural.hidden_layers": (16,)}.
    config_overrides: dict = field(default_factory=dict)
    # Whether to train with the opponent curriculum.
    curriculum: bool = False
    # The name of another variant whose champion seeds this one (e.g. imitation before ppo).
    warm_from: str | None = None


@dataclass
class ComparisonConfig:
    """How much to train and how to judge."""

    # Run folder label.
    name: str = "comparison"
    # Iterations per variant (epochs, generations).
    iterations: int = 20
    # Optional wall-clock budget per variant in seconds (stops early if reached).
    time_budget: float | None = None
    # Games each champion plays in the final tournament.
    tournament_games: int = 75
    # Learner copies per tournament game.
    tournament_learners: int = 6
    # CPU workers for trainers and the tournament.
    workers: int = 1
    # Seed.
    seed: int = 0
    # Where run folders go.
    results_dir: str = "results"


def set_overrides(config: SimulationConfig, overrides: dict) -> SimulationConfig:
    """Copy a config with dotted overrides applied."""
    # Copy.
    updated = copy.deepcopy(config)
    for dotted, value in overrides.items():
        target = updated
        parts = dotted.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)
    return updated


def count_lines(method: str) -> int:
    """Lines of code in the method's module (a rough measure of how much there is to implement)."""
    # Module per method (reinforce is shared by ppo, which adds its own file).
    files = {
        "imitation": ["imitation.py"],
        "genetic": ["genetic.py"],
        "neat": ["neat.py", "../brain/neat.py"],
        "reinforce": ["reinforce.py"],
        "ppo": ["reinforce.py", "ppo.py"],
    }[method]
    base = Path(__file__).resolve().parent.parent / "training"
    return sum(len((base / f).read_text().splitlines()) for f in files)


class MethodComparison:
    """Trains every variant and runs the tournament."""

    def __init__(self, base_config: SimulationConfig, comparison: ComparisonConfig, variants: list[Variant]) -> None:
        """Remember what to compare."""
        # Base settings.
        self.base_config = base_config
        self.comparison = comparison
        self.variants = variants
        # Results per variant.
        self.learning: dict[str, list[dict]] = {}
        self.champions: dict[str, LearnerSpec] = {}
        self.train_seconds: dict[str, float] = {}
        self.tournament: dict[str, dict] = {}
        self.run_dir: Path | None = None
        self._stop = False

    # ------------------------------------------------------------ training

    def _build(self, variant: Variant) -> Any:
        """Construct the trainer for a variant."""
        trainer_class, default_settings = METHODS[variant.method]
        settings = variant.settings if variant.settings is not None else default_settings()
        # Shared knobs.
        for name, value in (("workers", self.comparison.workers), ("seed", self.comparison.seed)):
            if hasattr(settings, name):
                setattr(settings, name, value)
        config = set_overrides(self.base_config, variant.config_overrides)
        config.seed = self.comparison.seed
        curriculum = Curriculum(CurriculumConfig()) if variant.curriculum else None
        initial = None
        if variant.warm_from is not None and variant.warm_from in self.champions:
            spec = self.champions[variant.warm_from]
            initial = spec.genome if (spec.kind == "neat") == (variant.method == "neat") else None
        return trainer_class(config, settings, initial_genome=initial, curriculum=curriculum)

    def train_all(self, on_progress: Callable[[str, int], None] | None = None) -> None:
        """Train every variant in order (so warm starts can use earlier champions)."""
        for variant in self.variants:
            if self._stop:
                break
            trainer = self._build(variant)
            started = time.time()
            for iteration in range(self.comparison.iterations):
                if self._stop:
                    break
                trainer.step()
                if on_progress is not None:
                    on_progress(variant.name, iteration + 1)
                if self.comparison.time_budget is not None and time.time() - started >= self.comparison.time_budget:
                    break
            self.train_seconds[variant.name] = time.time() - started
            self.learning[variant.name] = [s.to_row() for s in trainer.learning_history]
            self.champions[variant.name] = trainer.champion_spec()
            if self.run_dir is not None:
                save_run(trainer, variant.method, variant.name, self.run_dir / "runs")

    # ---------------------------------------------------------- tournament

    def run_tournament(self, on_progress: Callable[[str, int, int], None] | None = None) -> None:
        """Every champion plays the same seeded games as the learner against voting opponents."""
        config = SimulationConfig(**{**self.base_config.to_dict_raw(), "seed": self.comparison.seed})
        learners = learner_ids(config.num_players, self.comparison.tournament_learners)
        for name, spec in self.champions.items():
            if self._stop:
                break
            jobs = [
                (config, None, spec, learners, 50000 + i, True, False) for i in range(self.comparison.tournament_games)
            ]
            if self.comparison.workers > 1:
                from concurrent.futures import ProcessPoolExecutor

                with ProcessPoolExecutor(max_workers=self.comparison.workers) as pool:
                    results = list(pool.map(_run_episode_job, jobs))
            else:
                results = [_run_episode_job(job) for job in jobs]
            outcomes = [o for r in results for o in r["outcomes"].values()]
            self.tournament[name] = {
                "mean_score": float(np.mean([o["return"] for o in outcomes])),
                "win_rate": float(np.mean([o["won"] for o in outcomes])),
                "mean_survival": float(np.mean([o["survival"] for o in outcomes])),
                "mean_kills": float(np.mean([o["kills"] for o in outcomes])),
                "games": len(results),
            }
            if on_progress is not None:
                on_progress(name, len(self.tournament), len(self.champions))

    # ------------------------------------------------------------- report

    def run(self, on_progress: Callable[[str, str], None] | None = None) -> Path:
        """Train, fight, write."""
        self._stop = False
        self.run_dir = make_run_dir(self.comparison.results_dir, self.comparison.name)
        (self.run_dir / "config.json").write_text(
            json.dumps(
                {
                    "base_config": self.base_config.to_dict(),
                    "comparison": asdict(self.comparison),
                    "variants": [
                        {**asdict(v), "settings": asdict(v.settings) if v.settings is not None else None}
                        for v in self.variants
                    ],
                },
                indent=2,
                default=str,
            )
        )
        self.train_all(lambda name, i: on_progress(name, f"iteration {i}") if on_progress else None)
        self.run_tournament(
            lambda name, done, total: on_progress(name, f"tournament {done}/{total}") if on_progress else None
        )
        self.write()
        return self.run_dir

    def stop(self) -> None:
        """Stop after the current step."""
        self._stop = True

    def table(self) -> pd.DataFrame:
        """One row per variant with training and tournament numbers."""
        rows = []
        for variant in self.variants:
            name = variant.name
            learning = self.learning.get(name, [])
            tournament = self.tournament.get(name, {})
            rows.append(
                {
                    "variant": name,
                    "method": variant.method,
                    "iterations": len(learning),
                    "train_seconds": round(self.train_seconds.get(name, 0.0), 1),
                    "final_mean_score": learning[-1]["mean_score"] if learning else float("nan"),
                    "best_val_score": max((r["val_score"] for r in learning), default=float("nan")),
                    "tournament_score": tournament.get("mean_score", float("nan")),
                    "tournament_win_rate": tournament.get("win_rate", float("nan")),
                    "tournament_survival": tournament.get("mean_survival", float("nan")),
                    "tournament_kills": tournament.get("mean_kills", float("nan")),
                    "lines_of_code": count_lines(variant.method),
                }
            )
        return pd.DataFrame(rows)

    def write(self) -> None:
        """results.csv, summary.json, plots, a LaTeX table and report.md."""
        if self.run_dir is None:
            return
        table = self.table()
        table.to_csv(self.run_dir / "results.csv", index=False)
        (self.run_dir / "summary.json").write_text(
            json.dumps(
                {"table": table.to_dict(orient="records"), "tournament": self.tournament, "learning": self.learning},
                indent=2,
                default=str,
            )
        )
        # A LaTeX table for a paper (pandas warns about a future signature change; the current call is fine).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            (self.run_dir / "results_table.tex").write_text(table.to_latex(index=False, float_format="%.2f"))
        folder = self.run_dir / "plots"
        # Learning curves overlaid.
        plots.overlay_curves(
            {
                name: ([r["iteration"] for r in rows], [r["mean_score"] for r in rows])
                for name, rows in self.learning.items()
            },
            "Mean score per iteration",
            "iteration",
            "score",
            folder / "score_by_method.png",
        )
        plots.overlay_curves(
            {
                name: ([r["cumulative_seconds"] for r in rows], [r["mean_score"] for r in rows])
                for name, rows in self.learning.items()
            },
            "Mean score against training time",
            "seconds",
            "score",
            folder / "score_by_time.png",
        )
        plots.overlay_curves(
            {
                name: ([r["iteration"] for r in rows], [r["val_score"] for r in rows])
                for name, rows in self.learning.items()
            },
            "Validation score per iteration",
            "iteration",
            "score",
            folder / "validation_by_method.png",
        )
        plots.overlay_curves(
            {
                name: ([r["iteration"] for r in rows], [r["entropy"] for r in rows])
                for name, rows in self.learning.items()
            },
            "Policy entropy per iteration",
            "iteration",
            "nats",
            folder / "entropy_by_method.png",
        )
        plots.overlay_curves(
            {
                name: ([r["iteration"] for r in rows], [r["mean_length"] for r in rows])
                for name, rows in self.learning.items()
            },
            "Average game length per iteration",
            "iteration",
            "ticks",
            folder / "length_by_method.png",
        )
        # Tournament bars.
        if self.tournament:
            names = list(self.tournament)
            for key, title in (
                ("mean_score", "Tournament: mean score"),
                ("win_rate", "Tournament: win rate"),
                ("mean_survival", "Tournament: survival (ticks)"),
                ("mean_kills", "Tournament: kills per game"),
            ):
                plots.bars(
                    names, [self.tournament[n][key] for n in names], title, key, folder / f"tournament_{key}.png"
                )
        plots.bars(
            list(table["variant"]),
            list(table["lines_of_code"]),
            "Implementation size (lines of code)",
            "lines",
            folder / "lines_of_code.png",
        )
        plots.bars(
            list(table["variant"]),
            list(table["train_seconds"]),
            "Training time (seconds)",
            "seconds",
            folder / "train_seconds.png",
        )
        (self.run_dir / "report.md").write_text(self.report(table))

    def report(self, table: pd.DataFrame) -> str:
        """A generated write-up ranking the methods."""
        lines = [f"# Method comparison: {self.comparison.name}", ""]
        lines.append(
            f"Every variant trained for up to {self.comparison.iterations} iterations"
            + (f" or {self.comparison.time_budget:.0f} seconds" if self.comparison.time_budget else "")
            + f", then each champion played {self.comparison.tournament_games} seeded games as the learner against voting opponents."
        )
        lines.append("")
        if self.tournament:
            ranked = table.sort_values("tournament_score", ascending=False)
            lines.append("## Ranking by tournament score")
            lines.append("")
            lines.append(
                "| rank | variant | method | tournament score | win rate | survival | train seconds | lines of code |"
            )
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
            for rank, row in enumerate(ranked.itertuples(), 1):
                lines.append(
                    f"| {rank} | {row.variant} | {row.method} | {row.tournament_score:.2f} | {row.tournament_win_rate:.2f} | {row.tournament_survival:.0f} | {row.train_seconds:.0f} | {row.lines_of_code} |"
                )
            lines.append("")
            best = ranked.iloc[0]
            lines.append(
                f"**Best in the tournament:** {best.variant} ({best.method}) with a mean score of {best.tournament_score:.2f} and a win rate of {best.tournament_win_rate:.2f}."
            )
            cheapest = table.sort_values("lines_of_code").iloc[0]
            lines.append(f"**Simplest to implement:** {cheapest.method} ({cheapest.lines_of_code} lines).")
            fastest = table.sort_values("train_seconds").iloc[0]
            lines.append(
                f"**Fastest to train under this budget:** {fastest.variant} ({fastest.train_seconds:.0f} seconds)."
            )
            lines.append("")
        lines.append("## Why the methods differ")
        lines.append("")
        for method in dict.fromkeys(v.method for v in self.variants):
            lines.append(f"- **{method}**: {METHOD_NOTES[method]}")
        lines.append("")
        lines.append("## Charts")
        lines.append("")
        lines.append(
            "`plots/score_by_method.png`, `plots/score_by_time.png`, `plots/validation_by_method.png`, `plots/entropy_by_method.png`, `plots/length_by_method.png`, `plots/tournament_*.png`, `plots/lines_of_code.png`, `plots/train_seconds.png`. Each variant's own run folder is under `runs/`."
        )
        return "\n".join(lines) + "\n"
