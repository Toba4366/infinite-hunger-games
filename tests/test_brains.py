"""Tests for perception vectors and the brain interface."""

import numpy as np

from hunger_games.actions import Action
from hunger_games.brain import BRAIN_REGISTRY, create_brain
from hunger_games.brain.voting import DEFAULT_GENES, VotingBrain
from hunger_games.config import SimulationConfig
from hunger_games.game import Game
from hunger_games.perception import VECTOR_SIZE
from hunger_games.resources import ResourceKind


def first_perception():
    game = Game(SimulationConfig(width=60, height=60, seed=5))
    player = game.players[0]
    return player.perceive(game.arena, game.players, False, 0.0, 1.0, game.config.vision_radius)


def test_vector_size_matches_constant():
    """A neural network relies on the perception vector having a fixed length."""
    assert first_perception().to_vector().shape == (VECTOR_SIZE,)


def test_every_registered_brain_returns_an_action():
    """Any brain in the registry must satisfy the interface."""
    perception = first_perception()
    rng = np.random.default_rng(0)
    for name in BRAIN_REGISTRY:
        for chaos in (0.0, 1.0):
            brain = create_brain(name, chaos, rng)
            assert isinstance(brain.decide(perception, rng), Action)


def test_genome_round_trip():
    """Genomes must survive a save/load cycle so a genetic algorithm can evolve them."""
    for name in BRAIN_REGISTRY:
        brain = create_brain(name, 0.0, np.random.default_rng(1))
        genome = brain.genome()
        mutated = genome + 0.1
        brain.set_genome(mutated)
        assert np.allclose(brain.genome(), mutated)


def test_voting_brain_drinks_when_thirsty_in_water():
    """The thirst instinct should win when standing in water with an empty bar."""
    perception = first_perception()
    perception.thirst = 0.05
    perception.in_water = True
    perception.nearby_players = []
    perception.in_danger_zone = False
    perception.resource_here_kind = ResourceKind.NONE
    perception.nearby_resource_distance = float("inf")
    action = VotingBrain(chaos=0.0).decide(perception, np.random.default_rng(0))
    assert action.kind.value == "drink"
    assert len(DEFAULT_GENES) == 8
