"""research/plots.py - one PNG per chart, ready to drop into a paper.

Every function here draws exactly one figure, saves it to the path it is
given, closes it, and returns the path. Nothing is combined into grids, so
each chart can be cited on its own. The functions read either the CSV
tables the runner writes, a telemetry summary from research/telemetry.py,
or the history rows a trainer keeps.
"""

# Filesystem paths.
from pathlib import Path

# matplotlib, told not to open windows.
import matplotlib

# Use the file-only backend so these work on servers and in worker threads.
matplotlib.use("Agg")
# The plotting interface.
import matplotlib.pyplot as plt

# numpy for arrays.
import numpy as np

# pandas for the CSV tables.
import pandas as pd

# Animation for the growing-curve GIF.
from matplotlib.animation import FuncAnimation

# Need bin labels and alive bin labels for axes.
from hunger_games.research.telemetry import ALIVE_BIN_LABELS, NEED_BIN_LABELS

# A consistent colour per action, in ACTION_NAMES order.
ACTION_COLORS = ["#9e9e9e", "#4c78a8", "#1e90ff", "#f2d648", "#59a14f", "#e377c2", "#ff7f0e", "#c41e3a", "#7b3fa0"]


def _save(fig, path: str | Path) -> Path:
    """Tidy, save and close a figure."""
    # Layout.
    fig.tight_layout()
    # Make sure the folder exists.
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Save.
    fig.savefig(str(path), dpi=140)
    # Free the memory.
    plt.close(fig)
    # Done.
    return Path(path)


# =============================================================== chapter 3


def eliminations_per_day(eliminations: pd.DataFrame, num_games: int, path: str | Path) -> Path:
    """Average eliminations per day (chapter 2's exponential decay)."""
    # Figure.
    fig, ax = plt.subplots(figsize=(7, 4))
    # Bars.
    (eliminations.groupby("day").size() / max(1, num_games)).plot.bar(ax=ax, color="firebrick")
    # Labels.
    ax.set_title("Eliminations per day (average per game)")
    ax.set_xlabel("day")
    ax.set_ylabel("eliminations")
    # Save.
    return _save(fig, path)


def eliminations_by_method(eliminations: pd.DataFrame, path: str | Path) -> Path:
    """Share of the three chapter 3 categories."""
    # Figure.
    fig, ax = plt.subplots(figsize=(6, 4))
    # Bars as percentages.
    (eliminations["method"].value_counts(normalize=True) * 100).plot.bar(ax=ax, color="slateblue")
    # Labels.
    ax.set_title("Eliminations by method")
    ax.set_ylabel("% of eliminations")
    ax.set_xlabel("")
    # Save.
    return _save(fig, path)


def weapons_used(eliminations: pd.DataFrame, path: str | Path) -> Path:
    """Weapons in player-versus-player eliminations."""
    # Figure.
    fig, ax = plt.subplots(figsize=(7, 4))
    # Bars.
    eliminations[eliminations["method"] == "player_vs_player"]["weapon"].value_counts().plot.bar(
        ax=ax, color="darkorange"
    )
    # Labels.
    ax.set_title("Weapons used (player vs player)")
    ax.set_ylabel("eliminations")
    ax.set_xlabel("")
    # Save.
    return _save(fig, path)


def placement_by_score(players: pd.DataFrame, path: str | Path) -> Path:
    """Average placing per training score (lower is better; drawn with better at the top)."""
    # Figure.
    fig, ax = plt.subplots(figsize=(7, 4))
    # Line.
    players.groupby("training_score")["placement"].mean().plot(ax=ax, marker="o", color="seagreen")
    # Better placings at the top.
    ax.invert_yaxis()
    # Labels.
    ax.set_title("Average placing by training score")
    ax.set_xlabel("training score")
    ax.set_ylabel("placing (1 = victor)")
    # Save.
    return _save(fig, path)


def kills_by_score(players: pd.DataFrame, path: str | Path) -> Path:
    """Average kills per training score."""
    # Figure.
    fig, ax = plt.subplots(figsize=(7, 4))
    # Line.
    players.groupby("training_score")["kills"].mean().plot(ax=ax, marker="o", color="steelblue")
    # Labels.
    ax.set_title("Average kills by training score")
    ax.set_xlabel("training score")
    ax.set_ylabel("kills")
    # Save.
    return _save(fig, path)


def game_lengths(games: pd.DataFrame, path: str | Path) -> Path:
    """How many games lasted each number of days."""
    # Figure.
    fig, ax = plt.subplots(figsize=(7, 4))
    # Bars.
    games["days"].value_counts().sort_index().plot.bar(ax=ax, color="grey")
    # Labels.
    ax.set_title("Game length in days")
    ax.set_xlabel("days")
    ax.set_ylabel("games")
    # Save.
    return _save(fig, path)


def death_heatmap(eliminations: pd.DataFrame, width: int, height: int, path: str | Path, cells: int = 30) -> Path:
    """Where tributes die, as a 2-D density over the arena."""
    # Bin the death positions.
    grid, _, _ = np.histogram2d(eliminations["y"], eliminations["x"], bins=cells, range=[[0, height], [0, width]])
    # Draw.
    return heatmap(grid, "Where tributes die", path)


def deaths_by_district(eliminations: pd.DataFrame, path: str | Path) -> Path:
    """Eliminations suffered per district."""
    # Figure.
    fig, ax = plt.subplots(figsize=(7, 4))
    # Bars.
    eliminations["victim_district"].value_counts().sort_index().plot.bar(ax=ax, color="peru")
    # Labels.
    ax.set_title("Deaths by district")
    ax.set_xlabel("district")
    ax.set_ylabel("deaths")
    # Save.
    return _save(fig, path)


# ================================================================ heatmaps


def heatmap(grid, title: str, path: str | Path, cmap: str = "magma") -> Path:
    """A 2-D density plot of the arena."""
    # As an array.
    grid = np.asarray(grid, dtype=float)
    # Figure.
    fig, ax = plt.subplots(figsize=(5.5, 5))
    # Draw, normalised so colour means share of time.
    image = ax.imshow(grid / max(1.0, grid.sum()), cmap=cmap, origin="upper")
    # Colour bar.
    fig.colorbar(image, ax=ax, label="share of time")
    # Labels.
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    # Save.
    return _save(fig, path)


def armed_vs_unarmed(summary: dict, path: str | Path) -> Path:
    """Side-by-side heatmaps: where unarmed tributes go versus where armed ones go."""
    # Figure with two panels.
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # Each panel.
    for ax, key, title in zip(
        axes, ("unarmed_heat", "armed_heat"), ("Unarmed (weapon < 0.4)", "Armed (weapon >= 0.4)"), strict=False
    ):
        # The grid.
        grid = np.asarray(summary[key], dtype=float)
        # Draw.
        ax.imshow(grid / max(1.0, grid.sum()), cmap="magma", origin="upper")
        # Labels.
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    # Title.
    fig.suptitle("Where tributes spend time, by armament")
    # Save.
    return _save(fig, path)


# =============================================================== behaviour


def action_distribution(summary: dict, path: str | Path) -> Path:
    """Share of each action over all decisions."""
    # Counts.
    counts = np.asarray(summary["action_counts"], dtype=float)
    # Figure.
    fig, ax = plt.subplots(figsize=(8, 4))
    # Bars.
    ax.bar(summary["action_names"], counts / max(1.0, counts.sum()) * 100, color=ACTION_COLORS[: len(counts)])
    # Labels.
    ax.set_title(f"Action distribution (entropy {summary['entropy']:.2f} nats)")
    ax.set_ylabel("% of decisions")
    # Save.
    return _save(fig, path)


def action_by_need(summary: dict, need: str, path: str | Path) -> Path:
    """Stacked bars: which actions are chosen at each level of a need (thirst, hunger or health)."""
    # The matrix, rows = need bins, columns = actions.
    matrix = np.asarray(summary[f"action_by_{need}"], dtype=float)
    # Row shares.
    shares = matrix / np.maximum(1.0, matrix.sum(axis=1, keepdims=True)) * 100
    # Figure.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # Running bottom for stacking.
    bottom = np.zeros(len(NEED_BIN_LABELS))
    # Each action.
    for column, name in enumerate(summary["action_names"]):
        ax.bar(
            NEED_BIN_LABELS,
            shares[:, column],
            bottom=bottom,
            label=name,
            color=ACTION_COLORS[column % len(ACTION_COLORS)],
        )
        bottom += shares[:, column]
    # Labels.
    ax.set_title(f"What tributes do at each {need} level")
    ax.set_xlabel(f"{need} bar")
    ax.set_ylabel("% of decisions")
    ax.legend(fontsize=7, ncol=3)
    # Save.
    return _save(fig, path)


def need_action_curves(summary: dict, path: str | Path) -> Path:
    """P(drink | thirst) and P(eat | hunger): the instinct curves, which should rise steeply as bars empty."""
    # Names.
    names = summary["action_names"]
    # Column indices.
    drink, eat, heal = names.index("drink"), names.index("eat"), names.index("heal")
    # Probabilities per bin.
    thirst = np.asarray(summary["action_by_thirst"], dtype=float)
    hunger = np.asarray(summary["action_by_hunger"], dtype=float)
    health = np.asarray(summary["action_by_health"], dtype=float)
    # Shares.
    p_drink = thirst[:, drink] / np.maximum(1.0, thirst.sum(axis=1))
    p_eat = hunger[:, eat] / np.maximum(1.0, hunger.sum(axis=1))
    p_heal = health[:, heal] / np.maximum(1.0, health.sum(axis=1))
    # Figure.
    fig, ax = plt.subplots(figsize=(7, 4))
    # Lines.
    ax.plot(NEED_BIN_LABELS, p_drink * 100, marker="o", label="P(drink | thirst)", color="#1e90ff")
    ax.plot(NEED_BIN_LABELS, p_eat * 100, marker="s", label="P(eat | hunger)", color="#f2d648")
    ax.plot(NEED_BIN_LABELS, p_heal * 100, marker="^", label="P(heal | health)", color="#e377c2")
    # Labels.
    ax.set_title("Instinct curves: action rate against the matching need")
    ax.set_xlabel("need bar level")
    ax.set_ylabel("% of decisions")
    ax.legend()
    # Save.
    return _save(fig, path)


def consumption_timing(summary: dict, path: str | Path) -> Path:
    """Histograms of the bar level at the moment of drinking, eating and healing."""
    # Figure with three panels.
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    # Bin centres.
    centres = np.arange(10) * 10 + 5
    # Each panel.
    for ax, key, label, color in zip(
        axes,
        ("thirst_at_drink", "hunger_at_eat", "health_at_heal"),
        ("thirst when drinking", "hunger when eating", "health when healing"),
        ("#1e90ff", "#f2d648", "#e377c2"),
        strict=False,
    ):
        ax.bar(centres, summary[key], width=9, color=color)
        ax.set_title(label)
        ax.set_xlabel("% of bar")
    # Title.
    fig.suptitle("Item usage timing (learning tributes cluster at low levels)")
    # Save.
    return _save(fig, path)


def fight_or_flight(summary: dict, path: str | Path) -> Path:
    """Share of attack versus flee decisions, by health, when someone is in sight."""
    # The matrix.
    combat = np.asarray(summary["combat_by_health"], dtype=float)
    # Shares.
    total = np.maximum(1.0, combat.sum(axis=1))
    # Figure.
    fig, ax = plt.subplots(figsize=(7, 4))
    # Stacked bars.
    ax.bar(NEED_BIN_LABELS, combat[:, 0] / total * 100, label="attack", color="#c41e3a")
    ax.bar(
        NEED_BIN_LABELS, combat[:, 1] / total * 100, bottom=combat[:, 0] / total * 100, label="flee", color="#7b3fa0"
    )
    # Labels.
    ax.set_title("Fight or flight by health (someone in sight)")
    ax.set_xlabel("health bar")
    ax.set_ylabel("% of combat decisions")
    ax.legend()
    # Save.
    return _save(fig, path)


def proximity_vs_alive(summary: dict, path: str | Path) -> Path:
    """Average distance kept from the nearest visible tribute, by how many remain."""
    # Means.
    means = np.asarray(summary["proximity_sum"]) / np.maximum(1.0, np.asarray(summary["proximity_count"]))
    # Figure.
    fig, ax = plt.subplots(figsize=(7, 4))
    # Line, from most alive to the final few.
    ax.plot(ALIVE_BIN_LABELS[::-1], means[::-1], marker="o", color="#ff7f0e")
    # Labels.
    ax.set_title("Distance kept from the nearest tribute as the field shrinks")
    ax.set_xlabel("tributes remaining")
    ax.set_ylabel("mean distance (cells)")
    # Save.
    return _save(fig, path)


def action_by_alive(summary: dict, path: str | Path) -> Path:
    """Stacked bars of actions by how many tributes remain (aggression should rise at the end)."""
    # Matrix.
    matrix = np.asarray(summary["action_by_alive"], dtype=float)[::-1]
    # Shares.
    shares = matrix / np.maximum(1.0, matrix.sum(axis=1, keepdims=True)) * 100
    # Figure.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # Stack.
    bottom = np.zeros(len(ALIVE_BIN_LABELS))
    # Each action.
    for column, name in enumerate(summary["action_names"]):
        ax.bar(
            ALIVE_BIN_LABELS[::-1],
            shares[:, column],
            bottom=bottom,
            label=name,
            color=ACTION_COLORS[column % len(ACTION_COLORS)],
        )
        bottom += shares[:, column]
    # Labels.
    ax.set_title("What tributes do as the field shrinks")
    ax.set_xlabel("tributes remaining")
    ax.set_ylabel("% of decisions")
    ax.legend(fontsize=7, ncol=3)
    # Save.
    return _save(fig, path)


def deaths_by_cause(summary: dict, path: str | Path) -> Path:
    """Deaths by cause from a telemetry summary."""
    # Figure.
    fig, ax = plt.subplots(figsize=(7, 4))
    # Data.
    causes = summary["deaths_by_cause"]
    # Bars.
    ax.bar(list(causes.keys()), list(causes.values()), color="slategrey")
    # Labels.
    ax.set_title("Deaths by cause")
    ax.set_ylabel("deaths")
    ax.tick_params(axis="x", rotation=45)
    # Save.
    return _save(fig, path)


# ========================================================= over training


def curves(xs, series: dict[str, list[float]], title: str, xlabel: str, ylabel: str, path: str | Path) -> Path:
    """A generic multi-line chart: one line per entry of `series`."""
    # Figure.
    fig, ax = plt.subplots(figsize=(7, 4))
    # Each line.
    for label, ys in series.items():
        ax.plot(xs, ys, marker=".", label=label)
    # Labels.
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Legend when more than one line.
    if len(series) > 1:
        ax.legend()
    # Save.
    return _save(fig, path)


def stacked_area_over_training(summaries: list[dict], path: str | Path, xlabel: str = "generation") -> Path:
    """Action share per training step, as a stacked area (random at first, structured later)."""
    # Nothing.
    if not summaries:
        return curves([], {}, "Action distribution over training", xlabel, "%", path)
    # Matrix: rows = steps, columns = actions.
    matrix = np.asarray([s["action_counts"] for s in summaries], dtype=float)
    # Shares.
    shares = matrix / np.maximum(1.0, matrix.sum(axis=1, keepdims=True)) * 100
    # Figure.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # Area.
    ax.stackplot(
        range(len(summaries)), shares.T, labels=summaries[0]["action_names"], colors=ACTION_COLORS[: shares.shape[1]]
    )
    # Labels.
    ax.set_title("Action distribution over training")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("% of decisions")
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    # Save.
    return _save(fig, path)


def death_needs_over_training(summaries: list[dict], path: str | Path, xlabel: str = "generation") -> Path:
    """Mean thirst, hunger and health at death per training step (should fall if starvation stops)."""
    # Series.
    xs = list(range(len(summaries)))
    # Each bar.
    series = {
        "thirst at death": [s["mean_death_needs"][0] for s in summaries],
        "hunger at death": [s["mean_death_needs"][1] for s in summaries],
        "health at death": [s["mean_death_needs"][2] for s in summaries],
    }
    # Draw.
    return curves(xs, series, "Resource levels at death over training", xlabel, "bar level (0 to 1)", path)


def behaviour_metrics_over_training(summaries: list[dict], path: str | Path, xlabel: str = "generation") -> Path:
    """Survival ticks, win rate, kill rate and action entropy per training step, on separate axes."""
    # Figure with four panels.
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    # X.
    xs = list(range(len(summaries)))
    # Panels.
    for ax, key, title in zip(
        axes.ravel(),
        ("mean_survival_ticks", "win_rate", "kill_rate", "entropy"),
        ("Survival (ticks)", "Win rate", "Kills per game", "Policy entropy (nats)"),
        strict=False,
    ):
        ax.plot(xs, [s[key] for s in summaries], marker=".")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
    # Title.
    fig.suptitle("Behaviour over training")
    # Save.
    return _save(fig, path)


def timing(history_rows: list[dict], path: str | Path, xlabel: str = "step") -> Path:
    """Seconds per step and cumulative training time."""
    # X.
    xs = [row.get("generation", row.get("epoch", i)) for i, row in enumerate(history_rows)]
    # Cumulative if present, else running sum.
    cumulative = (
        [row["cumulative_seconds"] for row in history_rows]
        if history_rows and "cumulative_seconds" in history_rows[0]
        else list(np.cumsum([row["seconds"] for row in history_rows]))
    )
    # Figure.
    fig, ax = plt.subplots(figsize=(7, 4))
    # Per step.
    ax.bar(xs, [row["seconds"] for row in history_rows], color="lightgrey", label="seconds per step")
    # Cumulative on a second axis.
    ax2 = ax.twinx()
    ax2.plot(xs, cumulative, color="firebrick", marker=".", label="cumulative")
    # Labels.
    ax.set_title("Training time")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("seconds per step")
    ax2.set_ylabel("cumulative seconds")
    # Save.
    return _save(fig, path)


def curve_gif(
    xs, series: dict[str, list[float]], title: str, xlabel: str, ylabel: str, path: str | Path, fps: int = 6
) -> Path:
    """Animate a chart growing one point at a time (a loss curve that draws itself)."""
    # Figure.
    fig, ax = plt.subplots(figsize=(7, 4))
    # One line per series, empty to start.
    lines = {label: ax.plot([], [], marker=".", label=label)[0] for label in series}
    # Fixed axes so the view does not jump.
    all_values = [v for ys in series.values() for v in ys] or [0.0, 1.0]
    # A single point would make the limits identical, so widen them a little.
    low_x, high_x = (min(xs), max(xs)) if len(xs) else (0, 1)
    ax.set_xlim(low_x - 0.5 if low_x == high_x else low_x, high_x + 0.5 if low_x == high_x else high_x)
    ax.set_ylim(
        min(all_values) - 0.05 * (max(all_values) - min(all_values) + 1e-9),
        max(all_values) + 0.05 * (max(all_values) - min(all_values) + 1e-9),
    )
    # Labels.
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    # Per-frame update.
    def draw(frame: int):
        # Show the first `frame + 1` points of each line.
        for label, line in lines.items():
            line.set_data(xs[: frame + 1], series[label][: frame + 1])
        return list(lines.values())

    # Animate.
    animation = FuncAnimation(fig, draw, frames=max(1, len(xs)), interval=1000 // fps, repeat=False)
    # Make sure the folder exists.
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Save as a GIF.
    animation.save(str(path), writer="pillow", fps=fps)
    # Close.
    plt.close(fig)
    # Done.
    return Path(path)


# ================================================== comparison helpers


def overlay_curves(
    series: dict[str, tuple[list, list]], title: str, xlabel: str, ylabel: str, path: str | Path
) -> Path:
    """Several (xs, ys) lines on one chart, one per method or variant."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, (xs, ys) in series.items():
        if len(xs):
            ax.plot(xs, ys, marker=".", label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if series:
        ax.legend(fontsize=8)
    return _save(fig, path)


def bars(labels: list[str], values: list[float], title: str, ylabel: str, path: str | Path) -> Path:
    """A simple bar chart, one bar per label."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(labels, values, color="slateblue")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    return _save(fig, path)


# ==================================================== shared learning curves


def learning_curve_plots(learning_rows: list[dict], folder: str | Path) -> list[Path]:
    """The curves every method shares: score, entropy, game length, win rate, time, curriculum."""
    # Nothing.
    if not learning_rows:
        return []
    folder = Path(folder)
    xs = [r["iteration"] for r in learning_rows]
    return [
        curves(
            xs,
            {
                "mean score": [r["mean_score"] for r in learning_rows],
                "best score": [r["best_score"] for r in learning_rows],
                "validation score": [r["val_score"] for r in learning_rows],
            },
            "Score per iteration",
            "iteration",
            "score (episode return)",
            folder / "score.png",
        ),
        curves(
            xs,
            {"entropy": [r["entropy"] for r in learning_rows]},
            "Policy entropy",
            "iteration",
            "nats",
            folder / "entropy_shared.png",
        ),
        curves(
            xs,
            {"mean game length": [r["mean_length"] for r in learning_rows]},
            "Average game length (learner survival)",
            "iteration",
            "ticks",
            folder / "game_length.png",
        ),
        curves(
            xs,
            {"win rate": [r["win_rate"] for r in learning_rows]},
            "Win rate",
            "iteration",
            "rate",
            folder / "win_rate_shared.png",
        ),
        curves(
            [r["cumulative_seconds"] for r in learning_rows],
            {"mean score": [r["mean_score"] for r in learning_rows]},
            "Score against wall-clock time",
            "seconds",
            "score",
            folder / "score_vs_time.png",
        ),
        curves(
            xs,
            {"opponents": [r["opponents"] for r in learning_rows]},
            "Curriculum: opponents per iteration",
            "iteration",
            "opponents",
            folder / "curriculum.png",
        ),
        curve_gif(
            xs,
            {
                "mean score": [r["mean_score"] for r in learning_rows],
                "validation": [r["val_score"] for r in learning_rows],
            },
            "Score per iteration",
            "iteration",
            "score",
            folder / "score.gif",
        ),
    ]


# ============================================================== bundles


def training_run_plots(history_rows: list[dict], summaries: list[dict], folder: str | Path, method: str) -> list[Path]:
    """Write every chart a training run should have into `folder` and return the paths."""
    # Folder.
    folder = Path(folder)
    # Paths written.
    written = []
    # Nothing to draw.
    if not history_rows:
        return written
    # X axis.
    step = "generation" if method == "genetic" else ("iteration" if method == "neat" else "epoch")
    xs = [row[step] for row in history_rows]
    # Performance curves.
    if method == "neat":
        written.append(
            curves(
                xs,
                {
                    "species": [r.get("extra_species", 0) for r in history_rows],
                    "hidden nodes": [r.get("extra_hidden_nodes", 0) for r in history_rows],
                },
                "NEAT structure",
                step,
                "count",
                folder / "neat_structure.png",
            )
        )
        written.append(
            curves(
                xs,
                {
                    "best": [r["best_score"] for r in history_rows],
                    "mean": [r["mean_score"] for r in history_rows],
                    "validation": [r["val_score"] for r in history_rows],
                },
                "Fitness by generation",
                step,
                "score",
                folder / "fitness.png",
            )
        )
    elif method == "imitation":
        written.append(
            curves(
                xs,
                {
                    "training loss": [r["train_loss"] for r in history_rows],
                    "validation loss": [r["val_loss"] for r in history_rows],
                },
                "Imitation loss (cross-entropy)",
                step,
                "loss",
                folder / "losses.png",
            )
        )
        written.append(
            curves(
                xs,
                {
                    "training accuracy": [r["train_accuracy"] for r in history_rows],
                    "validation accuracy": [r["val_accuracy"] for r in history_rows],
                },
                "How often the student picks the teacher's action",
                step,
                "accuracy",
                folder / "accuracy.png",
            )
        )
        written.append(
            curves(
                xs,
                {"validation survival (ticks)": [r["val_survival"] for r in history_rows]},
                "Student survival in validation games",
                step,
                "ticks",
                folder / "survival.png",
            )
        )
        written.append(
            curves(
                xs,
                {"validation win rate": [r["val_win_rate"] for r in history_rows]},
                "Student win rate in validation games",
                step,
                "rate",
                folder / "win_rate.png",
            )
        )
        written.append(
            curve_gif(
                xs,
                {
                    "training": [r["train_loss"] for r in history_rows],
                    "validation": [r["val_loss"] for r in history_rows],
                },
                "Imitation loss (cross-entropy)",
                step,
                "loss",
                folder / "losses.gif",
            )
        )
    elif method == "genetic":
        written.append(
            curves(
                xs,
                {
                    "best fitness": [r["best_fitness"] for r in history_rows],
                    "mean fitness": [r["mean_fitness"] for r in history_rows],
                    "validation fitness": [r["val_fitness"] for r in history_rows],
                },
                "Fitness by generation",
                step,
                "fitness",
                folder / "fitness.png",
            )
        )
        written.append(
            curve_gif(
                xs,
                {
                    "best": [r["best_fitness"] for r in history_rows],
                    "validation": [r["val_fitness"] for r in history_rows],
                },
                "Fitness by generation",
                step,
                "fitness",
                folder / "fitness.gif",
            )
        )
    else:
        written.append(
            curves(
                xs,
                {
                    "training return": [r["train_return"] for r in history_rows],
                    "validation return": [r["val_return"] for r in history_rows],
                },
                "Cumulative reward per episode",
                step,
                "reward",
                folder / "reward.png",
            )
        )
        written.append(
            curves(
                xs,
                {
                    "policy loss": [r["policy_loss"] for r in history_rows],
                    "value loss": [r["value_loss"] for r in history_rows],
                },
                "Losses",
                step,
                "loss",
                folder / "losses.png",
            )
        )
        written.append(
            curves(
                xs,
                {"policy entropy": [r["entropy"] for r in history_rows]},
                "Policy entropy",
                step,
                "nats",
                folder / "entropy.png",
            )
        )
        written.append(
            curves(
                xs,
                {
                    "training": [r["train_survival"] for r in history_rows],
                    "validation": [r["val_survival"] for r in history_rows],
                },
                "Survival time (ticks)",
                step,
                "ticks",
                folder / "survival.png",
            )
        )
        written.append(
            curves(
                xs,
                {
                    "training win rate": [r["win_rate"] for r in history_rows],
                    "validation win rate": [r["val_win_rate"] for r in history_rows],
                    "kills per game": [r["kill_rate"] for r in history_rows],
                },
                "Win and kill rate",
                step,
                "rate",
                folder / "win_kill_rate.png",
            )
        )
        written.append(
            curve_gif(
                xs,
                {
                    "training": [r["train_return"] for r in history_rows],
                    "validation": [r["val_return"] for r in history_rows],
                },
                "Cumulative reward per episode",
                step,
                "reward",
                folder / "reward.gif",
            )
        )
    # Timing.
    written.append(timing(history_rows, folder / "timing.png", step))
    # Behaviour over training.
    if summaries:
        written.append(stacked_area_over_training(summaries, folder / "action_distribution_over_training.png", step))
        written.append(death_needs_over_training(summaries, folder / "death_needs_over_training.png", step))
        written.append(behaviour_metrics_over_training(summaries, folder / "behaviour_over_training.png", step))
        # The latest behaviour in detail.
        written.extend(behaviour_plots(summaries[-1], folder))
    # Done.
    return written


def behaviour_plots(summary: dict, folder: str | Path) -> list[Path]:
    """Every behaviour chart for one telemetry summary."""
    # Folder.
    folder = Path(folder)
    # All of them.
    return [
        action_distribution(summary, folder / "action_distribution.png"),
        action_by_need(summary, "thirst", folder / "actions_by_thirst.png"),
        action_by_need(summary, "hunger", folder / "actions_by_hunger.png"),
        action_by_need(summary, "health", folder / "actions_by_health.png"),
        need_action_curves(summary, folder / "instinct_curves.png"),
        consumption_timing(summary, folder / "consumption_timing.png"),
        fight_or_flight(summary, folder / "fight_or_flight.png"),
        proximity_vs_alive(summary, folder / "proximity_vs_remaining.png"),
        action_by_alive(summary, folder / "actions_by_remaining.png"),
        heatmap(summary["position_heat"], "Where tributes spend time", folder / "position_heatmap.png"),
        armed_vs_unarmed(summary, folder / "armed_vs_unarmed_heatmaps.png"),
        deaths_by_cause(summary, folder / "deaths_by_cause.png"),
    ]


def batch_plots(
    eliminations: pd.DataFrame,
    players: pd.DataFrame,
    games: pd.DataFrame,
    folder: str | Path,
    width: int = 120,
    height: int = 120,
) -> list[Path]:
    """Every chapter 3 chart for a batch of games, one PNG each."""
    # Folder.
    folder = Path(folder)
    # All of them.
    written = [
        eliminations_per_day(eliminations, len(games), folder / "eliminations_per_day.png"),
        eliminations_by_method(eliminations, folder / "eliminations_by_method.png"),
        weapons_used(eliminations, folder / "weapons_used.png"),
        placement_by_score(players, folder / "placement_by_score.png"),
        kills_by_score(players, folder / "kills_by_score.png"),
        game_lengths(games, folder / "game_lengths.png"),
        deaths_by_district(eliminations, folder / "deaths_by_district.png"),
    ]
    # The death heatmap needs positions.
    if len(eliminations):
        written.append(death_heatmap(eliminations, width, height, folder / "death_heatmap.png"))
    # Done.
    return written
