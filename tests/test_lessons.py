"""Tests for the lesson curriculum: stages with rules, promotion on survival, and per-episode variants."""

from hunger_games.config import SimulationConfig
from hunger_games.training.common import (
    Curriculum,
    CurriculumConfig,
    Stage,
    apply_overrides,
    episode_config,
    stage_config,
)


def test_opponent_ladder_is_unchanged():
    """Without explicit stages the curriculum is the classic ladder, judged on wins."""
    curriculum = Curriculum(CurriculumConfig())
    assert [s.opponents for s in curriculum.stages] == [1, 3, 7, 11, 23]
    assert curriculum.stage_spec.metric == "win_rate" and curriculum.opponents == 1
    # Five majority-win iterations promote, as before.
    assert [curriculum.observe(0.0, 1.0) for _ in range(5)] == [False, False, False, False, True]
    assert curriculum.opponents == 3


def test_lessons_start_with_survival_and_end_with_generalisation():
    """The lesson curriculum has eight stages: two survival lessons, five win lessons, one generalisation lesson."""
    curriculum = Curriculum(CurriculumConfig.lessons())
    names = [s.name for s in curriculum.stages]
    assert names[:2] == ["survive", "survive the rules"] and names[-1] == "generalise"
    assert [s.opponents for s in curriculum.stages] == [0, 0, 1, 3, 7, 11, 23, 23]
    assert curriculum.stage_spec.metric == "survival"
    # Winning does not promote a survival lesson; five half-game survivals sit below the 0.6 bar.
    assert not any(curriculum.observe(0.0, win_rate=1.0, survival=0.5) for _ in range(5))
    # Surviving most of the game does: within five more iterations the window mean passes the bar.
    promoted = [curriculum.observe(0.0, win_rate=0.0, survival=0.95) for _ in range(5)]
    assert promoted.count(True) == 1 and curriculum.stage == 1
    assert curriculum.describe() == "survive the rules (0 opponents)"
    assert not curriculum.finished


def test_stage_config_applies_opponents_and_rules():
    """The stage config sizes the roster and applies the lesson's rules without touching the base config."""
    base = SimulationConfig(seed=0, width=40, height=40, max_days=2)
    survive = Curriculum(CurriculumConfig.lessons()).stages[0]
    config = stage_config(base, survive, learners=6)
    assert config.num_players == 6 and config.gamemaker_enabled is False and config.sponsors_enabled is False
    assert base.gamemaker_enabled is True
    beat_seven = Curriculum(CurriculumConfig.lessons()).stages[4]
    assert stage_config(base, beat_seven, learners=6).num_players == 13
    # Dotted overrides reach nested settings and enums are given as strings.
    changed = apply_overrides(base, {"neural.hidden_layers": [16], "layout": "cornucopia"})
    assert tuple(changed.neural.hidden_layers) == (16,) and changed.layout.value == "cornucopia"


def test_episode_config_picks_a_variant_by_seed():
    """The generalisation lesson varies the rules per episode, deterministically by seed, covering every variant."""
    base = SimulationConfig(seed=0, width=40, height=40, max_days=2)
    generalise = Curriculum(CurriculumConfig.lessons()).stages[-1]
    config = stage_config(base, generalise, learners=6)
    seen = set()
    for seed in range(60):
        episode = episode_config(config, generalise, seed)
        seen.add((episode.layout.value, episode.shape.value, episode.gamemaker_enabled, episode.sponsors_enabled))
        assert episode_config(config, generalise, seed).layout == episode.layout
    assert len(seen) >= 4
    # A stage without variants returns the stage config itself.
    assert episode_config(config, Stage("beat 1", 1), 7) is config


def test_comparison_rejects_an_unknown_curriculum_name():
    """A mistyped curriculum name fails loudly instead of silently becoming the opponent ladder."""
    import pytest

    from hunger_games.research.comparison import ComparisonConfig, MethodComparison, Variant

    comparison = MethodComparison(SimulationConfig(seed=0), ComparisonConfig(name="t"), [])
    with pytest.raises(ValueError):
        comparison._build(Variant("x", "reinforce", curriculum="lesson"))
