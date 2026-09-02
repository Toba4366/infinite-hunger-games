# `test_initializers.py`

**Source:** [tests/test_initializers.py](../../tests/test_initializers.py)
**Tests:** [../brain/neural.md](../brain/neural.md) (`NeuralBrain`, `MENU_SIZE`) and through it `hunger_games/brain/mlp.py` (`MLP`), [../brain/initializers.md](../brain/initializers.md) (`INITIALIZERS`, `INITIALIZER_NOTES`, `ACTIVATIONS`, `initialize`), [../config.md](../config.md) (`NeuralConfig`), [../perception.md](../perception.md) (`VECTOR_SIZE`)

## Purpose

Before a neural network is trained, every weight needs a starting value. `brain/initializers.py` offers fourteen ways to pick those values, from the deliberately useless (`zeros`) to the textbook rules (Xavier, He, LeCun, orthogonal). It also holds the five activation functions a `NeuralBrain` can put between its layers, and their derivatives for backpropagation. The dashboard lets a game maker pick any initializer and activation by name, so every name in the registry must work and must have a tooltip.

This file checks four things. Every registered initializer builds a matrix of the requested shape and has a note. The constant initializers fill with exactly the value they promise. The scaled-variance rules stay within their theoretical limits. The structural ones (orthogonal, identity, sparse) have the algebraic property they are named after. Then it checks the activations on three hand-picked inputs, and finally it builds a two-hidden-layer `NeuralBrain` and proves the layer sizes, the parameter count, and the genome round trip all agree.

Without these tests a typo in one formula, say `sqrt(6 / fan_in)` where `sqrt(2 / fan_in)` was meant, would silently make every trained network worse, and a missing tooltip would crash the dashboard. The derivatives are not tested here; `tests/test_research.py` checks them against finite differences.

## Concepts you need

**Test discovery.** pytest collects every `test_*` function in this file. There are six.

**`pytest.approx`.** Floating point maths rarely gives exact answers. `x == pytest.approx(-0.02)` passes if `x` is within a tiny relative tolerance of `-0.02`. Use it whenever the expected value comes from a formula.

**`pytest.raises`.** A context manager that passes only if the block inside raises the named exception. `with pytest.raises(ValueError): brain.set_genome(np.zeros(3))` proves a wrong-sized genome is rejected instead of silently reshaped.

**numpy boolean reductions.** `(matrix == 1.0).all()` is True only if every element is 1.0. `not matrix.any()` is True only if every element is zero. `(sparse != 0).mean()` is the fraction of non-zero entries.

**`np.allclose` and `np.array_equal`.** `allclose` compares with tolerance, `array_equal` compares exactly. The orthogonal test uses `allclose` with `atol=1e-8` because QR decomposition has rounding error; the identity test uses `array_equal` because `np.eye` is exact.

**fan_in and fan_out.** For a weight matrix of shape `(fan_in, fan_out)`, `fan_in` is how many inputs feed a neuron and `fan_out` is how many neurons the layer has. The scaled rules use these to keep signals from growing or shrinking as they pass through.

**Delegation.** `NeuralBrain` no longer holds weights itself. It builds an `MLP` (kept as `brain.network`) and forwards `genome`, `set_genome`, `parameter_count` and `describe` to it. The last test calls the brain's methods, so it covers the delegation and the `MLP` in one go.

**Running a subset.** `python -m pytest tests/test_initializers.py -k scaled` runs one test.

## Walkthrough

### `test_every_initializer_has_a_note_and_the_right_shape()`

**Setup.** One generator, `default_rng(0)`, shared by every call.

**`for name in INITIALIZERS: assert name in INITIALIZER_NOTES`.** The dashboard shows `INITIALIZER_NOTES[name]` as a tooltip. A registry entry without a note would raise `KeyError` in the UI. This is the test that fires when someone adds an initializer and forgets the note.

**`assert initialize(name, rng, 7, 5).shape == (7, 5)`.** Every initializer must return a `(fan_in, fan_out)` matrix. A rectangular shape catches transposed results, which a square shape would miss. `orthogonal` in particular pads its QR result when the layer is wider than it is tall, and this line proves the padding produces the right shape.

**Why 7 and 5.** Small, unequal, and not a multiple of each other, so a swapped axis or an off-by-one in the padding shows up.

### `test_constant_style_initializers()`

**`assert not initialize("zeros", rng, 4, 3).any()`.** Every entry is zero.

**`assert (initialize("ones", rng, 4, 3) == 1.0).all()`.** Every entry is one.

**`assert (initialize("constant", rng, 4, 3, scale=0.7) == 0.7).all()`.** The `constant` initializer uses `scale` as its fill value. A failure would mean `scale` was ignored, which would make the dashboard's `init_scale` slider do nothing for this option.

### `test_scaled_variance_rules()`

**Setup.** `default_rng(1)`, `fan_in, fan_out = 100, 50`. Large enough that the sampled maximum gets close to the limit, so an off-by-a-factor bug is caught.

**`assert np.abs(initialize("xavier_uniform", ...)).max() <= np.sqrt(6.0 / (fan_in + fan_out))`.** Glorot's uniform rule draws from `[-limit, limit]` with `limit = sqrt(6 / (fan_in + fan_out))`, about 0.2 here. No sample may exceed it.

**`assert np.abs(initialize("he_uniform", ...)).max() <= np.sqrt(6.0 / fan_in)`.** He's rule ignores `fan_out` and uses `sqrt(6 / fan_in)`, about 0.245.

**`assert np.abs(initialize("lecun_uniform", ...)).max() <= np.sqrt(3.0 / fan_in)`.** LeCun's rule uses `sqrt(3 / fan_in)`, about 0.173.

These three lines each check an upper bound only. If an implementation drew from a much smaller range it would still pass, so the test is a guard against "too big", not "too small".

**`assert abs(initialize("he_normal", rng, 10000, 5).std() - np.sqrt(2.0 / 10000)) < 0.002`.** He's normal rule has standard deviation `sqrt(2 / fan_in)`, which is 0.01414 for `fan_in = 10000`. Fifty thousand samples give a sample standard deviation within 0.002 of that. A failure would mean the formula used a variance where a standard deviation was meant, or the wrong fan.

**Why 10000 by 5.** The bigger the sample, the tighter the standard deviation estimate. 50000 numbers is plenty and still instant.

### `test_orthogonal_identity_and_sparse()`

**`q = initialize("orthogonal", rng, 8, 8)` then `assert np.allclose(q.T @ q, np.eye(8), atol=1e-8)`.** A matrix is orthogonal when its transpose times itself is the identity, meaning every column is a unit vector perpendicular to every other. This is exactly what lets the layer pass a signal through without stretching or shrinking it. The tolerance absorbs QR rounding error.

**`assert np.array_equal(initialize("identity", rng, 4, 4), np.eye(4))`.** Exact comparison. `np.eye` puts ones on the diagonal.

**`sparse = initialize("sparse", rng, 100, 100, scale=0.1, sparsity=0.1)` then `assert 0.05 < (sparse != 0).mean() < 0.15`.** With 10000 entries and a 10 percent keep mask, the non-zero fraction lands very close to 0.10. The wide window, 0.05 to 0.15, means the test never flakes, while still failing if `sparsity` were ignored (fraction 1.0) or inverted (fraction 0.9).

### `test_activations_behave()`

**Setup.** `x = np.array([-2.0, 0.0, 2.0])`: one negative, zero, one positive.

**`assert (ACTIVATIONS["relu"](x) == [0.0, 0.0, 2.0]).all()`.** ReLU zeroes negatives and passes positives through.

**`assert ACTIVATIONS["leaky_relu"](x)[0] == pytest.approx(-0.02)`.** The negative side is scaled by 0.01, so -2.0 becomes -0.02.

**`assert 0.0 < sigmoid(x)[0] < 0.5 < sigmoid(x)[2] < 1.0`.** Sigmoid maps negatives below 0.5 and positives above, always strictly inside (0, 1). This chained comparison checks four inequalities at once.

**`assert ACTIVATIONS["selu"](x)[2] == pytest.approx(2.0 * 1.0507009873554805)`.** On the positive side SELU is linear with slope lambda, so 2.0 becomes about 2.1014. The constant is the one from the SELU paper, copied into the source.

`tanh` is not tested here because it is `np.tanh` straight from numpy.

### `test_multilayer_network_shapes_and_genome()`

**Setup.** `NeuralConfig(hidden_layers=(32, 16), activation="relu", initializer="he_uniform")` and `NeuralBrain(config=config, rng=default_rng(0))`. Two hidden layers, so the `MLP` inside has three weight matrices and three bias vectors.

**`assert brain.layer_sizes == [VECTOR_SIZE, 32, 16, MENU_SIZE]`.** `VECTOR_SIZE` is 50, the length of the perception vector. `MENU_SIZE` is 16: six simple actions, attack, flee and eight moves. The list must start with the input width and end with the output width, with the hidden widths in between. `NeuralBrain` builds this list and hands it to `MLP`, which uses neighbouring pairs as `(fan_in, fan_out)`.

**`expected = VECTOR_SIZE * 32 + 32 + 32 * 16 + 16 + 16 * MENU_SIZE + MENU_SIZE` then `assert brain.parameter_count == expected == brain.genome().size`.** Each layer has `fan_in * fan_out` weights plus `fan_out` biases. That is 1632 + 528 + 272 = 2432. The chained equality proves `MLP.parameter_count` and the flattened genome agree with the hand calculation.

**`genome = brain.genome() * 2.0; brain.set_genome(genome); assert np.allclose(brain.genome(), genome)`.** Doubling every value and loading it back must give the same doubled vector. `MLP.set_genome` walks the layers with a cursor, slicing out each weight matrix and then its bias. If it sliced in the wrong order or forgot a bias, the round trip would scramble it.

**`with pytest.raises(ValueError): brain.set_genome(np.zeros(3))`.** A three-value genome cannot fit a 2432-parameter network. `MLP.set_genome` raises `ValueError` with both numbers in the message. The clear error matters because a saved champion from a different architecture is the most common way to hit this.

**`assert "relu" in brain.describe() and "he_uniform" in brain.describe()`.** `describe()` is the one-line summary the dashboard shows, produced by `MLP.describe` as `50 -> 32 -> 16 -> 16, relu, he_uniform, 2432 params`. It must name the activation and initializer.

## How to run and extend

```bash
python -m pytest tests/test_initializers.py
python -m pytest tests/test_initializers.py -v
python -m pytest tests/test_initializers.py -k "orthogonal or multilayer"
python -m pytest tests/test_initializers.py::test_activations_behave
```

**1. Unknown names fail loudly.** `initialize` raises `KeyError` with the list of valid names.

```python
def test_unknown_initializer_raises():
    with pytest.raises(KeyError):
        initialize("glorot", np.random.default_rng(0), 4, 4)
```

**2. Seeds make initializers repeatable.** Same seed, same matrix.

```python
def test_initializers_are_seeded():
    a = initialize("normal", np.random.default_rng(5), 6, 6)
    b = initialize("normal", np.random.default_rng(5), 6, 6)
    assert np.array_equal(a, b)
```

**3. A zero-initialised network gives every action the same score.** This is why `zeros` is "comparison only".

```python
def test_zero_network_is_indifferent():
    brain = NeuralBrain(config=NeuralConfig(initializer="zeros"), rng=np.random.default_rng(0))
    logits = brain.forward(np.ones(VECTOR_SIZE))
    assert np.allclose(logits, logits[0])
```

**4. A single hidden layer still has three sizes.** The default config has `hidden_layers=(16,)`, giving 1088 parameters.

```python
def test_default_network_shape():
    brain = NeuralBrain(rng=np.random.default_rng(0))
    assert brain.layer_sizes == [VECTOR_SIZE, 16, MENU_SIZE]
    assert brain.parameter_count == VECTOR_SIZE * 16 + 16 + 16 * MENU_SIZE + MENU_SIZE
```

**5. Every activation has a derivative.** `ACTIVATION_DERIVATIVES` must have the same keys as `ACTIVATIONS`, or `MLP.__init__` raises `KeyError` for the missing one.

```python
from hunger_games.brain.initializers import ACTIVATION_DERIVATIVES

def test_every_activation_has_a_derivative():
    assert set(ACTIVATION_DERIVATIVES) == set(ACTIVATIONS)
```

## Gotchas

**Every initializer takes the same five arguments.** `zeros(rng, fan_in, fan_out, scale, sparsity)` ignores four of them. That is on purpose so `initialize` can call any of them the same way. Do not "simplify" a signature or the registry breaks.

**Upper bounds only.** The three uniform checks would pass for a matrix of zeros. Pair them with a spread check if you change the formulas.

**The `sparse` initializer clamps `scale`.** It uses `max(scale, 1e-3)` so `scale=0.0` does not produce an all-zero matrix by accident.

**`VECTOR_SIZE` and `MENU_SIZE` are imported, not hard-coded.** If the perception vector or the action menu grows, this test follows automatically. The hand-written `expected` formula is the only thing that would need updating if the biases changed.

**`brain.weights` and `brain.biases` are views into the `MLP`.** They are properties that return `brain.network.weights` and `brain.network.biases`. Editing an element through them edits the network; reassigning `brain.weights = ...` does nothing useful.

**Randomness in `sparse` and `he_normal`.** Both tests use tolerances wide enough that they never flake with the fixed seeds. If you change a seed, re-check the margins.

**Order of `INITIALIZERS` matters for `--help` and the dashboard.** The dict order is the display order. Adding to the end keeps existing screenshots valid.
