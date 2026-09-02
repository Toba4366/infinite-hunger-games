"""Tests for the research layer: the MLP's gradients, telemetry, the RL trainer, plots and sweeps."""

import json

import numpy as np

from hunger_games.brain.mlp import MLP, Adam
from hunger_games.config import SimulationConfig
from hunger_games.game import Game
from hunger_games.research import plots
from hunger_games.research.experiments import Sweep, SweepConfig, set_field
from hunger_games.research.telemetry import ACTION_NAMES, BehaviorTelemetry
from hunger_games.training import GeneticTrainer, ReinforceTrainer, RLConfig, TrainingConfig, save_run


def small() -> SimulationConfig:
    return SimulationConfig(seed=7, width=50, height=50, max_days=4)


def test_mlp_backward_matches_finite_differences():
    """Analytic gradients must agree with numeric ones for every activation."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 5))
    target = rng.normal(size=(6, 3))
    for activation in ("tanh", "relu", "leaky_relu", "sigmoid", "selu"):
        net = MLP([5, 4, 3], activation, rng=np.random.default_rng(1))
        out, cache = net.forward_cached(x)
        grads = net.backward(cache, 2.0 * (out - target) / out.size)
        eps = 1e-6
        for layer in range(2):
            i, j = 1, 2
            w = net.weights[layer]
            w[i, j] += eps
            plus = np.mean((net.forward(x) - target) ** 2)
            w[i, j] -= 2 * eps
            minus = np.mean((net.forward(x) - target) ** 2)
            w[i, j] += eps
            assert abs((plus - minus) / (2 * eps) - grads[layer][0][i, j]) < 1e-5, activation


def test_adam_reduces_a_simple_loss():
    """A few Adam steps on a regression problem must lower the loss."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 4))
    target = x @ rng.normal(size=(4, 2))
    net = MLP([4, 8, 2], "tanh", rng=rng)
    optimizer = Adam(net, 0.01)
    losses = []
    for _ in range(200):
        out, cache = net.forward_cached(x)
        losses.append(float(np.mean((out - target) ** 2)))
        optimizer.step(net.backward(cache, 2.0 * (out - target) / out.size))
    assert losses[-1] < losses[0] * 0.5
    genome = net.genome()
    net.set_genome(genome * 0 + 1.0)
    assert (net.genome() == 1.0).all()


def test_telemetry_counts_every_decision_and_merges():
    """Every decision of every tracked tribute is tallied, and merging adds summaries up."""
    game = Game(small())
    telemetry = BehaviorTelemetry(50, 50).attach(game)
    decisions = []
    game.decision_hooks.append(lambda p, per, a: decisions.append(1))
    game.run()
    summary = telemetry.summary()
    assert sum(summary["action_counts"]) == len(decisions)
    assert summary["games"] == 1
    assert len(summary["survival_ticks"]) == 24
    assert summary["death_count"] == len(game.eliminations)
    assert set(summary["action_names"]) == set(ACTION_NAMES)
    merged = BehaviorTelemetry.merge([summary, summary])
    assert merged["games"] == 2
    assert sum(merged["action_counts"]) == 2 * len(decisions)
    assert len(merged["survival_ticks"]) == 48
    json.dumps(merged)


def test_reinforce_trainer_runs_and_saves(tmp_path):
    """Two epochs of REINFORCE run, log every metric, and the run folder gets its charts."""
    trainer = ReinforceTrainer(
        small(), RLConfig(epochs=2, episodes_per_epoch=1, learners_per_game=4, validation_games=1, seed=0)
    )
    history = trainer.run()
    assert len(history) == 2
    row = history[-1].to_row()
    for key in (
        "policy_loss",
        "value_loss",
        "entropy",
        "train_return",
        "val_return",
        "train_survival",
        "win_rate",
        "seconds",
        "cumulative_seconds",
    ):
        assert key in row and np.isfinite(row[key])
    assert 0.0 < history[-1].entropy <= np.log(16) + 1e-6
    assert trainer.champion.size == trainer.policy.parameter_count
    folder = save_run(trainer, "reinforce", "test_rl", tmp_path)
    assert (folder / "history.json").exists() and (folder / "champion.json").exists()
    assert len(list((folder / "plots").glob("*.png"))) >= 10
    assert (folder / "plots" / "reward.gif").exists()


def test_genetic_trainer_validation_telemetry_and_run_folder(tmp_path):
    """The GA logs validation fitness and telemetry per generation and writes a run folder."""
    trainer = GeneticTrainer(
        small(),
        TrainingConfig(
            brain_name="voting", population_size=24, generations=2, rounds_per_generation=1, validation_games=1, seed=0
        ),
    )
    trainer.run()
    rows = trainer.history_rows()
    assert all("val_fitness" in row and "cumulative_seconds" in row for row in rows)
    assert trainer.history[-1].telemetry["games"] >= 1
    assert trainer.previous_champion() is not None
    folder = save_run(trainer, "genetic", "test_ga", tmp_path)
    assert (folder / "plots" / "fitness.png").exists() and (folder / "plots" / "fitness.gif").exists()


def test_sweep_sets_nested_fields_and_writes_results(tmp_path):
    """A sweep over a nested field writes results.csv, summary.json and one plot per metric."""
    config = small()
    assert set_field(config, "terrain.water_threshold", 0.4).terrain.water_threshold == 0.4
    assert config.terrain.water_threshold == 0.25
    sweep = Sweep(
        config, SweepConfig(name="t", parameter="chaos", values=[0.0, 0.5], games_per_value=2, results_dir=tmp_path)
    )
    folder = sweep.run()
    assert (folder / "results.csv").exists() and (folder / "summary.json").exists()
    assert len(sweep.rows) == 2 and all("victor_rate" in row for row in sweep.rows)
    assert len(list((folder / "plots").glob("*.png"))) >= 5


def test_plots_write_one_png_each(tmp_path):
    """Every behaviour plot function writes its file."""
    game = Game(small())
    telemetry = BehaviorTelemetry(50, 50).attach(game)
    game.run()
    written = plots.behaviour_plots(telemetry.summary(), tmp_path)
    assert len(written) == 12 and all(path.exists() for path in written)
