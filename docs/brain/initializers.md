# `initializers.py`

**Source:** [hunger_games/brain/initializers.py](../../hunger_games/brain/initializers.py)
**Depends on:** `numpy` only
**Used by:** [brain/mlp.py](mlp.md) (`initialize`, `ACTIVATIONS`, `ACTIVATION_DERIVATIVES`); [hunger_games/ui/app.py](../ui/app.md) (`ACTIVATIONS` and `INITIALIZERS` fill the drop-downs, `INITIALIZER_NOTES` are the tooltips); `tests/test_initializers.py`

## Purpose

A neural network is a pile of weight matrices. Before training, every weight needs a starting value, and the choice matters more than it looks. All zeros makes every neuron identical forever. Too large and the activations saturate. Too small and the signal fades to nothing by the last layer. This file collects fourteen ways to pick those starting values, each as a function that builds one matrix of shape `(fan_in, fan_out)`.

The file also holds the five *activation functions* (the squashing curves applied between layers) and, new in this version, their *derivatives*. The derivatives are what make backpropagation possible: [mlp.py](mlp.md) multiplies by them on the way back through each hidden layer. The genetic algorithm never needs them, the REINFORCE trainer cannot work without them.

Everything is looked up by name through three dictionaries (`INITIALIZERS`, `ACTIVATIONS`, `ACTIVATION_DERIVATIVES`), so the dashboard and the config can pick any of them with a string.

## Concepts you need

**fan_in and fan_out.** For one layer, `fan_in` is how many numbers come in and `fan_out` how many go out. The weight matrix has one row per input and one column per output, so its shape is `(fan_in, fan_out)`. The first layer of the default brain has `fan_in = 50` (the perception vector) and `fan_out = 16`.

**Variance scaling.** Each output of a layer is a sum of `fan_in` products. If every weight has variance `v`, the sum has variance about `fan_in * v`. To keep the signal the same size from layer to layer you want `fan_in * v` near 1, so the weights must shrink as the layer gets wider. Xavier, He and LeCun are three formulas for exactly how much.

**Uniform versus normal.** A uniform draw between `-L` and `+L` has variance `L^2 / 3`. A normal draw with standard deviation `s` has variance `s^2`. That is why each rule comes in two flavours with different-looking constants that give the same variance: `sqrt(6 / n)` for uniform equals `sqrt(2 / n)` for normal, since `6 / 3 = 2`.

**Symmetry.** If two neurons start with identical weights they receive identical gradients and stay identical forever. `zeros`, `ones` and `constant` all have this problem. They exist so you can watch it happen.

**Activation.** After the weighted sum `z = x @ W + b`, a hidden layer applies a curve to get `a = f(z)`. Without it, stacking layers would be no better than one layer, because a chain of linear maps is still linear.

**Derivative.** The slope of the activation at the point it was evaluated. Backpropagation needs `f'(z)` for every hidden unit. Some slopes are easiest to write in terms of the output `a`, others in terms of the input `z`, so every derivative function here receives both.

## Walkthrough

Every initializer has the same signature, `(rng, fan_in, fan_out, scale, sparsity)`, even when it ignores most of the arguments. That is what lets `initialize()` call any of them without knowing which one it has.

### `Generator = np.random.Generator`

A short alias for the type hint.

### `zeros(rng, fan_in, fan_out, scale, sparsity) -> np.ndarray`

`np.zeros((fan_in, fan_out))`. Every weight is 0, so every neuron computes the same thing. Comparison only.

### `ones(rng, fan_in, fan_out, scale, sparsity) -> np.ndarray`

`np.ones((fan_in, fan_out))`. Also symmetric, and with `fan_in = 50` the sums are huge, so tanh and sigmoid saturate at once. Comparison only.

### `constant(rng, fan_in, fan_out, scale, sparsity) -> np.ndarray`

`np.full((fan_in, fan_out), scale)`. Every weight equals the user's `init_scale`. Still symmetric.

### `uniform(rng, fan_in, fan_out, scale, sparsity) -> np.ndarray`

`rng.uniform(-scale, scale, (fan_in, fan_out))`. A flat spread. The width is whatever you set, with no regard for the layer's size.

### `normal(rng, fan_in, fan_out, scale, sparsity) -> np.ndarray`

`rng.normal(0.0, scale, (fan_in, fan_out))`. A bell curve with standard deviation `scale`.

### `xavier_uniform(rng, fan_in, fan_out, scale, sparsity) -> np.ndarray`

Glorot's rule. `limit = sqrt(6 / (fan_in + fan_out))`, then uniform in `[-limit, limit]`. Balances the forward signal and the backward gradient by using both fan sizes. Best with `tanh` and `sigmoid`. This is the default initializer.

### `xavier_normal(rng, fan_in, fan_out, scale, sparsity) -> np.ndarray`

The same variance from a bell curve: `std = sqrt(2 / (fan_in + fan_out))`.

### `he_uniform(rng, fan_in, fan_out, scale, sparsity) -> np.ndarray`

Kaiming He's rule. `limit = sqrt(6 / fan_in)`. ReLU zeros half of its inputs, which halves the variance, so He doubles it back and ignores `fan_out`. Best with `relu` and `leaky_relu`.

### `he_normal(rng, fan_in, fan_out, scale, sparsity) -> np.ndarray`

`std = sqrt(2 / fan_in)`.

### `lecun_uniform(rng, fan_in, fan_out, scale, sparsity) -> np.ndarray`

LeCun's rule. `limit = sqrt(3 / fan_in)`, which gives variance exactly `1 / fan_in`. That is the assumption behind SELU's self-normalising maths.

### `lecun_normal(rng, fan_in, fan_out, scale, sparsity) -> np.ndarray`

`std = sqrt(1 / fan_in)`.

The three scaled rules for the default brain's first layer (`fan_in = 50`, `fan_out = 16`):

| Rule | Uniform limit | Normal std |
| --- | --- | --- |
| Xavier | `sqrt(6 / 66) = 0.302` | `sqrt(2 / 66) = 0.174` |
| He | `sqrt(6 / 50) = 0.346` | `sqrt(2 / 50) = 0.200` |
| LeCun | `sqrt(3 / 50) = 0.245` | `sqrt(1 / 50) = 0.141` |

### `orthogonal(rng, fan_in, fan_out, scale, sparsity) -> np.ndarray`

1. Draw a standard normal matrix of shape `(fan_in, fan_out)`.
2. `q, r = np.linalg.qr(...)` splits it into an orthogonal part `q` and an upper-triangular `r`.
3. Multiply each column of `q` by the sign of the matching diagonal entry of `r`, which makes the draw uniform over all orthogonal matrices instead of biased.
4. QR only produces `min(fan_in, fan_out)` orthogonal columns. If the layer is wider than it is tall (`fan_in < fan_out`), `q` is padded with zero columns to reach the right shape.

Orthogonal columns are perpendicular unit vectors, so the layer neither stretches nor shrinks the signal. The test checks `q.T @ q` is the identity for an 8 by 8 layer.

### `identity(rng, fan_in, fan_out, scale, sparsity) -> np.ndarray`

`np.eye(fan_in, fan_out)`: 1 on the leading diagonal, 0 elsewhere. A square layer passes its input straight through. For the 50 -> 16 layer only the first 16 inputs reach the hidden layer at all, so it is rarely useful here.

### `sparse(rng, fan_in, fan_out, scale, sparsity) -> np.ndarray`

Draw a normal matrix with `std = max(scale, 1e-3)`, then multiply by a random mask that keeps roughly a `sparsity` fraction of the entries. With the default `sparsity = 0.1`, about 90 percent of the weights start at exactly zero. The `max` guard stops a zero `init_scale` from making the whole matrix zero.

### `INITIALIZERS`

Name to function for all fourteen, in the order above. The dashboard's drop-down lists them in this order.

### `INITIALIZER_NOTES`

Name to one-line description, shown as the tooltip under the drop-down. The test checks every initializer has a note.

### `initialize(name: str, rng: Generator, fan_in: int, fan_out: int, scale: float = 0.05, sparsity: float = 0.1) -> np.ndarray`

The one entry point. Looks the name up (an unknown name raises `KeyError("Unknown initializer 'x'. Choose from: zeros, ones, ...")`) and calls it. `MLP.__init__` calls this once per layer.

```python
import numpy as np
from hunger_games.brain.initializers import initialize

rng = np.random.default_rng(0)
w = initialize("xavier_uniform", rng, 50, 16)
print(w.shape, abs(w).max() <= (6 / 66) ** 0.5)   # (50, 16) True
print(initialize("constant", rng, 2, 3, scale=0.7))  # a 2x3 matrix of 0.7
```

### `relu(x) -> np.ndarray`

`np.maximum(0.0, x)`. Negatives become 0, positives pass through.

### `leaky_relu(x) -> np.ndarray`

`np.where(x > 0.0, x, 0.01 * x)`. Like relu, but negatives are scaled by 0.01 instead of zeroed, so a neuron can never get stuck at zero output with zero gradient.

### `sigmoid(x) -> np.ndarray`

`1 / (1 + exp(-x))`. Squashes to 0..1. Its slope is at most 0.25, which is why deep sigmoid networks learn slowly.

### `selu(x) -> np.ndarray`

`lam * (x if x > 0 else alpha * (exp(x) - 1))` with `alpha = 1.6732632423543772` and `lam = 1.0507009873554805`. The `np.minimum(x, 0.0)` inside the exponent stops `exp` from overflowing on large positive inputs that the `where` would discard anyway. With LeCun initialisation the activations stay near mean 0 and variance 1 layer after layer.

### `ACTIVATIONS`

| Name | Function | Output range | Pair with |
| --- | --- | --- | --- |
| `tanh` | `np.tanh` | -1..1 | Xavier |
| `relu` | `relu` | 0..inf | He |
| `leaky_relu` | `leaky_relu` | -inf..inf | He |
| `sigmoid` | `sigmoid` | 0..1 | Xavier |
| `selu` | `selu` | about -1.76..inf | LeCun |

`tanh` maps straight to numpy's own function. It is the default.

### `_tanh_derivative(z, a) -> np.ndarray`

`1 - a^2`. Written in terms of the output, because `tanh'(z) = 1 - tanh(z)^2` and `a` is already `tanh(z)`. At `z = 0.5`, `a = 0.462` and the slope is `0.786`.

### `_relu_derivative(z, a) -> np.ndarray`

`(z > 0).astype(float)`: 1 where the input was positive, 0 otherwise. Note the strict `>`: at exactly `z = 0` the slope is taken as 0.

### `_leaky_relu_derivative(z, a) -> np.ndarray`

`np.where(z > 0, 1.0, 0.01)`. Never zero, so no neuron can die.

### `_sigmoid_derivative(z, a) -> np.ndarray`

`a * (1 - a)`. Largest at `a = 0.5` (slope 0.25), tiny near 0 and 1.

### `_selu_derivative(z, a) -> np.ndarray`

`lam` where `z > 0`, else `lam * alpha * exp(z)` (again with `exp(min(z, 0))` for safety). At `z = -1` the slope is `1.0507 * 1.6733 * 0.3679 = 0.647`.

### `ACTIVATION_DERIVATIVES`

Name to derivative, one for each entry of `ACTIVATIONS`. Every function takes `(z, a)`: the pre-activation sum and the activation, both as saved in the forward cache by `MLP.forward_cached`. `MLP.backward` looks up the one matching its activation name.

| Name | Slope formula | Uses |
| --- | --- | --- |
| `tanh` | `1 - a^2` | `a` |
| `relu` | `1 if z > 0 else 0` | `z` |
| `leaky_relu` | `1 if z > 0 else 0.01` | `z` |
| `sigmoid` | `a (1 - a)` | `a` |
| `selu` | `lam if z > 0 else lam alpha exp(z)` | `z` |

The functions are named with a leading underscore because nobody should call them directly. Reach them through the dictionary.

## How to use it / experiment

**Pick a pairing in the config.**

```python
from hunger_games.config import NeuralConfig
NeuralConfig(activation="relu", initializer="he_normal")
NeuralConfig(activation="selu", initializer="lecun_normal")
NeuralConfig(activation="tanh", initializer="orthogonal")
```

**Watch the signal size change through a layer.** Feed random inputs through one matrix from each rule and compare the output spread.

```python
import numpy as np
from hunger_games.brain.initializers import INITIALIZERS, initialize

rng = np.random.default_rng(0)
x = rng.normal(size=(1000, 50))
for name in INITIALIZERS:
    w = initialize(name, rng, 50, 16, scale=0.05, sparsity=0.1)
    print(f"{name:16s} output std {np.std(x @ w):.3f}")
```

With standard normal inputs, LeCun and `orthogonal` give an output spread of about 1.0, Xavier about 1.2 and He about 1.4. `uniform` and `normal` with `scale = 0.05` give about 0.21 and 0.36, and `sparse` about 0.11: the signal shrinks with every layer. `ones` gives about 7.4, which pushes tanh flat at once.

**Check a derivative numerically.** The slope should match a tiny finite difference.

```python
import numpy as np
from hunger_games.brain.initializers import ACTIVATIONS, ACTIVATION_DERIVATIVES

z = np.array([-1.5, -0.2, 0.3, 2.0])
for name in ACTIVATIONS:
    f, df = ACTIVATIONS[name], ACTIVATION_DERIVATIVES[name]
    numeric = (f(z + 1e-6) - f(z - 1e-6)) / 2e-6
    print(name, np.allclose(numeric, df(z, f(z)), atol=1e-5))
```

**Add an activation.** Write the function, write its derivative in `(z, a)` form, and add both to the two dictionaries. `MLP.__init__` looks both up by name, so a name present in `ACTIVATIONS` but missing from `ACTIVATION_DERIVATIVES` raises `KeyError` the moment a network is built.

## Gotchas

- `scale` is only read by `constant`, `uniform`, `normal` and `sparse`. Changing `init_scale` in the dashboard does nothing for the Xavier, He, LeCun, orthogonal and identity rules.
- `sparsity` is only read by `sparse`.
- `identity` on a non-square layer is `np.eye(fan_in, fan_out)`: inputs beyond `fan_out`, or outputs beyond `fan_in`, get all-zero weights.
- `orthogonal` pads with zero columns when `fan_in < fan_out`, so those extra outputs start at exactly zero (plus bias).
- `zeros`, `ones` and `constant` are symmetric. A network built from them can be evolved by the genetic algorithm (mutation breaks the symmetry), but gradient descent keeps every neuron in a layer identical.
- The `relu` derivative is 0 at `z == 0` exactly. With float inputs this almost never happens, but it does for a `zeros` network on the first step, where every `z` is 0 and no gradient reaches the weights below.
- The derivative functions assume `a` really is `f(z)`. Pass in a mismatched pair and `tanh` and `sigmoid` return nonsense with no error.
- `INITIALIZERS` and `ACTIVATIONS` are plain module-level dicts. Adding an entry at runtime works, but the dashboard reads them once when it builds its drop-downs.
