"""Tests for the neural initializers, activations and the multi-layer network."""

import numpy as np
import pytest

from hunger_games.brain.initializers import ACTIVATIONS, INITIALIZER_NOTES, INITIALIZERS, initialize
from hunger_games.brain.neural import MENU_SIZE, NeuralBrain
from hunger_games.config import NeuralConfig
from hunger_games.perception import VECTOR_SIZE


def test_every_initializer_has_a_note_and_the_right_shape():
    """Each named initializer must build a (fan_in, fan_out) matrix and have a tooltip."""
    rng = np.random.default_rng(0)
    for name in INITIALIZERS:
        assert name in INITIALIZER_NOTES
        assert initialize(name, rng, 7, 5).shape == (7, 5)


def test_constant_style_initializers():
    """zeros, ones and constant fill the whole matrix with one value."""
    rng = np.random.default_rng(0)
    assert not initialize("zeros", rng, 4, 3).any()
    assert (initialize("ones", rng, 4, 3) == 1.0).all()
    assert (initialize("constant", rng, 4, 3, scale=0.7) == 0.7).all()


def test_scaled_variance_rules():
    """Xavier, He and LeCun limits follow their formulas."""
    rng = np.random.default_rng(1)
    fan_in, fan_out = 100, 50
    assert np.abs(initialize("xavier_uniform", rng, fan_in, fan_out)).max() <= np.sqrt(6.0 / (fan_in + fan_out))
    assert np.abs(initialize("he_uniform", rng, fan_in, fan_out)).max() <= np.sqrt(6.0 / fan_in)
    assert np.abs(initialize("lecun_uniform", rng, fan_in, fan_out)).max() <= np.sqrt(3.0 / fan_in)
    assert abs(initialize("he_normal", rng, 10000, 5).std() - np.sqrt(2.0 / 10000)) < 0.002


def test_orthogonal_identity_and_sparse():
    """Orthogonal columns are perpendicular, identity is the identity, sparse is mostly zero."""
    rng = np.random.default_rng(2)
    q = initialize("orthogonal", rng, 8, 8)
    assert np.allclose(q.T @ q, np.eye(8), atol=1e-8)
    assert np.array_equal(initialize("identity", rng, 4, 4), np.eye(4))
    sparse = initialize("sparse", rng, 100, 100, scale=0.1, sparsity=0.1)
    assert 0.05 < (sparse != 0).mean() < 0.15


def test_activations_behave():
    """Each activation squashes or clips as advertised."""
    x = np.array([-2.0, 0.0, 2.0])
    assert (ACTIVATIONS["relu"](x) == [0.0, 0.0, 2.0]).all()
    assert ACTIVATIONS["leaky_relu"](x)[0] == pytest.approx(-0.02)
    assert 0.0 < ACTIVATIONS["sigmoid"](x)[0] < 0.5 < ACTIVATIONS["sigmoid"](x)[2] < 1.0
    assert ACTIVATIONS["selu"](x)[2] == pytest.approx(2.0 * 1.0507009873554805)


def test_multilayer_network_shapes_and_genome():
    """Layer sizes follow the config and the genome round-trips through every layer."""
    config = NeuralConfig(hidden_layers=(32, 16), activation="relu", initializer="he_uniform")
    brain = NeuralBrain(config=config, rng=np.random.default_rng(0))
    assert brain.layer_sizes == [VECTOR_SIZE, 32, 16, MENU_SIZE]
    expected = VECTOR_SIZE * 32 + 32 + 32 * 16 + 16 + 16 * MENU_SIZE + MENU_SIZE
    assert brain.parameter_count == expected == brain.genome().size
    genome = brain.genome() * 2.0
    brain.set_genome(genome)
    assert np.allclose(brain.genome(), genome)
    with pytest.raises(ValueError):
        brain.set_genome(np.zeros(3))
    assert "relu" in brain.describe() and "he_uniform" in brain.describe()
