"""Tests for the method comparison and its tournament."""

import time

from hunger_games.config import SimulationConfig
from hunger_games.research.comparison import ComparisonConfig, MethodComparison, Variant, count_lines
from hunger_games.training import ImitationConfig, RLConfig
from hunger_games.training.common import IterationStats


def test_comparison_trains_fights_and_reports(tmp_path):
    """Two variants train, the tournament scores both champions, and the folder has the table, plots and report."""
    config = SimulationConfig(seed=3, width=40, height=40, max_days=3)
    variants = [
        Variant(
            "imitation",
            "imitation",
            ImitationConfig(demonstration_games=1, epochs=1, validation_games=1, learners_per_game=4),
        ),
        Variant(
            "reinforce_warm",
            "reinforce",
            RLConfig(epochs=1, episodes_per_epoch=1, learners_per_game=4, validation_games=1),
            warm_from="imitation",
        ),
    ]
    comparison = MethodComparison(
        config,
        ComparisonConfig(name="t", iterations=1, tournament_games=2, tournament_learners=4, results_dir=tmp_path),
        variants,
    )
    folder = comparison.run()
    table = comparison.table()
    assert list(table["variant"]) == ["imitation", "reinforce_warm"]
    assert all(comparison.tournament[name]["games"] == 2 for name in comparison.tournament)
    assert (
        (folder / "results.csv").exists()
        and (folder / "report.md").exists()
        and (folder / "results_table.tex").exists()
    )
    assert (folder / "plots" / "tournament_win_rate.png").exists() and (
        folder / "plots" / "score_by_method.png"
    ).exists()
    assert (folder / "runs").exists()
    assert "Ranking by tournament score" in (folder / "report.md").read_text()
    assert count_lines("ppo") > count_lines("reinforce")


def test_comparison_extends_variants_that_miss_the_criterion(tmp_path):
    """A variant short of the criterion after the first budget keeps training in the extension phase."""
    config = SimulationConfig(seed=3, width=40, height=40, max_days=3)
    variants = [
        Variant(
            "reinforce_cold",
            "reinforce",
            RLConfig(epochs=1, episodes_per_epoch=1, learners_per_game=4, validation_games=1),
        ),
    ]
    # A window of two cannot be filled in the one-iteration first budget, so the extension must supply the second.
    # A threshold of zero is met by any window, so the variant stops as soon as the window is full.
    comparison = MethodComparison(
        config,
        ComparisonConfig(
            name="t",
            iterations=1,
            until_win_rate=0.0,
            win_window=2,
            extended_iterations=5,
            tournament_games=1,
            tournament_learners=4,
            results_dir=tmp_path,
        ),
        variants,
    )
    messages = []
    folder = comparison.run(on_progress=lambda name, what: messages.append(what))
    # One first-budget iteration plus one extension iteration met the criterion.
    assert comparison.criterion["reinforce_cold"][0] == 2
    assert comparison.extended["reinforce_cold"] == 1
    assert comparison.table()["extended_iterations"].tolist() == [1]
    assert any(what.startswith("extending") for what in messages)
    # The first-budget snapshot and the final run folder are both saved.
    assert (folder / "runs_first_budget").exists() and (folder / "runs").exists()
    assert "extended" in (folder / "report.md").read_text()


class _StubCurriculum:
    """A curriculum whose stage the test advances by hand."""

    def __init__(self, final_stage: int) -> None:
        """Start at the final stage minus one, so one promotion finishes it."""
        self.stage = final_stage - 1
        self.final_stage = final_stage

    @property
    def finished(self) -> bool:
        """True once the stage has reached the final index."""
        return self.stage >= self.final_stage


class _StubTrainer:
    """A trainer that wins every validation game and is promoted to the final stage on its fifth step."""

    def __init__(self) -> None:
        """Empty history, a curriculum one promotion short of finished."""
        self.learning_history: list[IterationStats] = []
        self.curriculum = _StubCurriculum(final_stage=4)

    def step(self) -> IterationStats:
        """Record a winning iteration at the current stage, then promote on the fifth."""
        stats = IterationStats(
            iteration=len(self.learning_history),
            scores=[1.0],
            mean_score=1.0,
            best_score=1.0,
            entropy=0.0,
            mean_length=10.0,
            win_rate=1.0,
            val_score=1.0,
            seconds=0.0,
            cumulative_seconds=0.0,
            val_win_rate=1.0,
            stage=self.curriculum.stage,
            opponents=11 if self.curriculum.stage == 3 else 23,
        )
        self.learning_history.append(stats)
        # Promotion happens after the record, as in the real trainers.
        if len(self.learning_history) == 5:
            self.curriculum.stage += 1
        return stats


def test_win_criterion_needs_a_full_window_at_the_final_stage():
    """The wins that earned the final promotion do not count; the window must be played at the final stage."""
    comparison = MethodComparison(
        SimulationConfig(seed=0), ComparisonConfig(name="t", until_win_rate=0.5, win_window=5), []
    )
    trainer = _StubTrainer()
    reached = comparison._train_steps(Variant("x", "reinforce"), trainer, 20, time.time(), None, None)
    # Five winning iterations at stage 3 promote the trainer at step 5; five more at stage 4 meet the criterion.
    assert reached[0] == 10
    assert all(s.stage == 4 for s in trainer.learning_history[-5:])
