"""Tests for the comparison runner's setting flags: --set parsing and the per-side --cold-set / --warm-set."""

# The experiments folder must be put on the import path by hand.
import sys

# Paths.
from pathlib import Path

# pytest.raises checks that the script stops with SystemExit on bad input.
import pytest

# A Variant names a method, its settings and (for warm starts) the variant it starts from.
from hunger_games.research.comparison import Variant

# `experiments/` is not a package, so import the runner script by folder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

# The runner under test (nothing runs at import time; main() is behind a guard).
import run_comparison as runner  # noqa: E402


def test_parse_settings_reads_literals_and_strings():
    """Numbers and booleans parse as Python literals; anything else stays a string; commas separate values."""
    # Three specs: two floats from one flag, a boolean, and a plain string.
    pairs = runner.parse_settings(["learning_rate=1e-3,3e-3", "record_showcase=False", "name=abc"])
    # Each value keeps the order it was given and its parsed type.
    assert pairs == [("learning_rate", 0.001), ("learning_rate", 0.003), ("record_showcase", False), ("name", "abc")]
    # A spec without "=" is a usage error.
    with pytest.raises(SystemExit):
        runner.parse_settings(["nonsense"])


def test_side_settings_reach_only_their_side():
    """--cold-set changes cold variants, --warm-set changes warm ones, imitation is left alone."""
    # One imitation variant, two cold variants and one warm variant.
    variants = [
        Variant("imitation", "imitation"),
        Variant("reinforce_cold", "reinforce"),
        Variant("reinforce_warm", "reinforce", warm_from="imitation"),
        Variant("ppo_cold", "ppo"),
    ]
    # Cold: a batch size for every method and a learning rate for REINFORCE only; warm: an entropy bonus.
    runner.apply_side_settings(
        variants,
        cold_specs=["episodes_per_epoch=16", "reinforce.learning_rate=3e-3"],
        warm_specs=["entropy_bonus=0.02"],
    )
    # Look variants up by name.
    by_name = {v.name: v for v in variants}
    # Imitation is neither cold nor warm, so it never gets a settings object.
    assert by_name["imitation"].settings is None
    # Both cold variants get the batch size; only REINFORCE gets the prefixed learning rate.
    assert by_name["reinforce_cold"].settings.episodes_per_epoch == 16
    assert by_name["reinforce_cold"].settings.learning_rate == 0.003
    assert by_name["ppo_cold"].settings.episodes_per_epoch == 16
    assert by_name["ppo_cold"].settings.learning_rate == 0.001
    # The warm variant keeps the default batch size and gets its own entropy bonus.
    assert by_name["reinforce_warm"].settings.episodes_per_epoch == 4
    assert by_name["reinforce_warm"].settings.entropy_bonus == 0.02


def test_check_known_rejects_unknown_fields_and_methods():
    """A setting no chosen method has, or a prefix naming a method not being run, stops the script."""
    # Both of these name real fields of methods that are being run.
    runner.check_known(["episodes_per_epoch=16", "reinforce.learning_rate=3e-3"], ["reinforce", "ppo"])
    # No method has a field called nonsense.
    with pytest.raises(SystemExit):
        runner.check_known(["nonsense=1"], ["reinforce", "ppo"])
    # REINFORCE is not among the methods being run.
    with pytest.raises(SystemExit):
        runner.check_known(["reinforce.learning_rate=3e-3"], ["ppo"])
