"""Tests for the method comparison and its tournament."""

from hunger_games.config import SimulationConfig
from hunger_games.research.comparison import ComparisonConfig, MethodComparison, Variant, count_lines
from hunger_games.training import ImitationConfig, RLConfig


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
