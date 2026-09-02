# `mlp.py`

**Source:** [hunger_games/brain/mlp.py](../../hunger_games/brain/mlp.py)
**Depends on:** `numpy`; [brain/initializers.py](initializers.md) (`ACTIVATION_DERIVATIVES`, `ACTIVATIONS`, `initialize`)
**Used by:** [brain/neural.py](neural.md) (`NeuralBrain.network` is an `MLP`); [training/reinforce.py](../../hunger_games/training/reinforce.py) (`MLP` for the value network, `Adam` for both networks, `forward_cached` and `backward` in the update); [hunger_games/ui/session.py](../ui/session.md) (`network_snapshot` calls `hidden_activations` on the selected tribute's network); `tests/test_research.py` (finite-difference gradient check, Adam convergence)

## Purpose

This is the maths engine behind the neural brain. It is a *multi-layer perceptron*: a list of weight matrices with one squashing function between them and a raw linear output. It is written in plain numpy with no framework, so every step is visible.

It does three things:

1. **Forward.** Push an input vector (or a batch of them) through the layers to get outputs.
2. **Backward.** Given how the loss changes with the outputs, work out how it changes with every weight and bias. This is backpropagation: the chain rule applied one layer at a time, from the output back to the input.
3. **Genome.** Expose every weight and bias as one flat vector, and load one back. The genetic algorithm only ever sees this vector.

The `Adam` class at the bottom is an optimiser: given the gradients, it decides how far to move each parameter. Plain gradient descent is also provided as `apply_gradients`.

Nothing in this file knows about the Hunger Games. It is a generic network that [neural.py](neural.md) points at the perception vector and the action menu.

## Concepts you need

**Layer sizes.** A list like `[50, 16, 16]`: 50 inputs, one hidden layer of 16, 16 outputs. Between each pair of neighbours there is one weight matrix of shape `(size_in, size_out)` and one bias vector of length `size_out`. Two matrices for three sizes.

**Row-vector convention.** Inputs are rows. A single input is a 1-D array of length `size_in`; a batch is a 2-D array with one row per example. The layer computes `z = x @ W + b`, which works for both because numpy broadcasts `b` across rows.

**Pre-activation and activation.** `z` is the weighted sum. `a = f(z)` is what leaves the layer. The last layer skips `f` and returns `z` raw, so the outputs can be any size and sign (the neural brain feeds them to a softmax; the value network reads the single number directly).

**Gradient.** For each parameter, how much the loss would change if that parameter grew by a tiny amount. Move every parameter a little *against* its gradient and the loss falls.

**Chain rule.** If the loss depends on `a`, and `a` depends on `z`, and `z` depends on `W`, then `dL/dW = dL/da * da/dz * dz/dW`. Backpropagation is that product, computed for every layer, reusing the piece already computed for the layer above.

**Batch sums.** When a batch of rows goes through, every row contributes to every weight. The weight gradient is the sum of the per-row contributions. If your loss is a *mean* over rows, put the `1 / batch_size` into the output gradient before calling `backward`.

**Finite differences.** A slow but trustworthy way to estimate a gradient: nudge one weight up by `eps`, measure the loss, nudge it down, measure again, and divide the difference by `2 * eps`. If backpropagation is correct the two numbers agree to many decimal places.

## Walkthrough

### `class MLP`

Fully connected layers with one activation between them and a raw output.

#### `__init__(self, layer_sizes: list[int], activation: str = "tanh", initializer: str = "xavier_uniform", init_scale: float = 0.05, sparsity: float = 0.1, rng: np.random.Generator | None = None) -> None`

Builds the layers.

- `layer_sizes`: every layer's width, input first, output last. Copied into `self.layer_sizes`.
- `activation`: a key of `ACTIVATIONS`. Stored as `self.activation_name`; the function goes in `self.activation` and its slope in `self.activation_derivative`.
- `initializer`: a key of `INITIALIZERS`, stored as `self.initializer_name`.
- `init_scale`, `sparsity`: passed through to the initializer (only some of them read these, see [initializers.md](initializers.md)).
- `rng`: the generator for the starting weights. `None` means a fresh unseeded one.

`self.weights` is a list of matrices, one per neighbouring pair of sizes, each built by `initialize(...)`. `self.biases` is a list of zero vectors, one per layer after the input.

```python
import numpy as np
from hunger_games.brain.mlp import MLP

net = MLP([50, 16, 16], "tanh", "xavier_uniform", rng=np.random.default_rng(0))
print([w.shape for w in net.weights])   # [(50, 16), (16, 16)]
print([b.shape for b in net.biases])    # [(16,), (16,)]
```

#### `parameter_count` (property) `-> int`

The total number of weights plus biases. For sizes `n0, n1, ..., nk` it is the sum over layers of `n_in * n_out + n_out`. For `[50, 16, 16]`: `50 * 16 + 16 + 16 * 16 + 16 = 1088`.

#### `describe(self) -> str`

One line for the dashboard: `"50 -> 16 -> 16, tanh, xavier_uniform, 1088 params"`.

#### `genome(self) -> np.ndarray`

Flattens every layer into one vector, in the order `W0.ravel(), b0, W1.ravel(), b1, ...`. `ravel` is row-major, so the first `size_out` entries are input 0's outgoing weights, the next `size_out` are input 1's, and so on.

For the default brain: entries 0..799 are the 50 by 16 first matrix, 800..815 its bias, 816..1071 the 16 by 16 second matrix, 1072..1087 its bias.

#### `set_genome(self, genome: np.ndarray) -> None`

The reverse. Converts to a float array, checks the size matches `parameter_count` (else `ValueError("Genome has 3 values but this network needs 1088")`), then walks a cursor along the vector slicing out each matrix and bias in the same order.

The slices are *views* into the genome array. If you pass a float64 numpy array, the network's weights share memory with it afterwards. See the gotchas.

```python
g = net.genome()
net.set_genome(g * 0 + 1.0)
print((net.genome() == 1.0).all())   # True
```

#### `forward(self, inputs: np.ndarray) -> np.ndarray`

Runs `forward_cached` and returns only the outputs. Accepts a single vector or a batch of rows; returns the same shape family (a vector of `layer_sizes[-1]`, or a matrix with that many columns).

#### `forward_cached(self, inputs: np.ndarray) -> tuple[np.ndarray, list]`

The forward pass that also remembers what backprop needs. For each layer `i`:

1. `z = signal @ W_i + b_i`
2. `a = activation(z)` for hidden layers, `a = z` for the last layer
3. append `(signal, z, a)` to the cache
4. `signal = a`

Returns `(outputs, cache)`.

**Cache layout.** `cache[i]` is a tuple of three arrays for layer `i`: the layer's *input*, its *pre-activation* `z`, and its *activation* `a`. For `MLP([5, 4, 3])` on a batch of 6 rows:

| Layer | Input shape | `z` shape | `a` shape |
| --- | --- | --- | --- |
| 0 | `(6, 5)` | `(6, 4)` | `(6, 4)` (tanh applied) |
| 1 | `(6, 4)` | `(6, 3)` | `(6, 3)` (raw, equals `z`) |

For a single vector the arrays are 1-D and `backward` promotes them to 2-D itself.

#### `hidden_activations(self, inputs: np.ndarray) -> list[np.ndarray]`

Runs `forward_cached` and returns just the `a` of every layer, hidden ones first and the raw output last. Despite the name, the output layer is included: the dashboard's network view draws every column after the inputs from this list and feeds `activations[-1]` (the logits) to `NeuralBrain.probabilities`.

#### `backward(self, cache: list, grad_outputs: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]`

Backpropagation. `grad_outputs` is `dL/d(outputs)`, the same shape as the outputs. Returns one `(grad_w, grad_b)` pair per layer, in layer order, each the same shape as the matching weight and bias. The gradients are *summed over the batch*.

The loop walks the layers from last to first. For layer `i` with cached `(layer_input, z, a)` and the incoming gradient `grad = dL/da_i`:

1. **Hidden layers only:** `grad = grad * activation_derivative(z, a)`. This turns `dL/da` into `dL/dz`. The output layer is linear, so `dL/dz = dL/da` and the step is skipped.
2. `grad_w = layer_input.T @ grad`. Shape `(size_in, size_out)`. The matrix product sums over the batch rows automatically.
3. `grad_b = grad.sum(axis=0)`. The bias enters every row with coefficient 1, so its gradient is the column sum.
4. `grad = grad @ W_i.T`. This is `dL/d(layer_input)`, which is `dL/da` for the layer below.

**Worked example.** A network `MLP([1, 1, 1], "tanh")` with `W0 = 0.5`, `b0 = 0`, `W1 = 2.0`, `b1 = 0`, input `x = 1.0`, and a loss whose gradient at the output is `1.0` (for example, loss equals the output).

Forward:

| Step | Value |
| --- | --- |
| `z0 = 1.0 * 0.5 + 0` | `0.5` |
| `a0 = tanh(0.5)` | `0.46212` |
| `z1 = 0.46212 * 2.0 + 0` | `0.92423` |
| output `= z1` | `0.92423` |

Backward, starting with `grad = 1.0` at the output:

| Layer 1 (output, linear) | Value |
| --- | --- |
| `grad_w1 = a0 * grad` | `0.46212` |
| `grad_b1 = grad` | `1.0` |
| `grad = grad * W1` (to the layer below) | `2.0` |

| Layer 0 (hidden, tanh) | Value |
| --- | --- |
| slope `1 - a0^2 = 1 - 0.21355` | `0.78645` |
| `grad = 2.0 * 0.78645` | `1.57290` |
| `grad_w0 = x * grad` | `1.57290` |
| `grad_b0 = grad` | `1.57290` |

Reading it as the chain rule: `dL/dW0 = dL/dz1 * dz1/da0 * da0/dz0 * dz0/dW0 = 1.0 * 2.0 * 0.78645 * 1.0`. Each factor is one line of the loop. Run it yourself:

```python
net = MLP([1, 1, 1], "tanh")
net.set_genome(np.array([0.5, 0.0, 2.0, 0.0]))
out, cache = net.forward_cached(np.array([1.0]))
print(out)                                   # [0.92423431]
print(net.backward(cache, np.array([1.0])))  # [(1.5729, 1.5729), (0.4621, 1.0)]
```

**A note on batch sums.** With a batch of `n` rows, `grad_w` and `grad_b` are sums of the `n` per-row gradients. Both callers in this package want a mean, so they divide the *output* gradient by the batch size before calling `backward`: the test passes `2 * (out - target) / out.size` for a mean-squared-error loss, and the REINFORCE trainer does `grad_logits /= count`. Dividing at the top is cheaper than dividing every layer's result, and the chain rule carries the factor down unchanged.

#### `apply_gradients(self, grads: list[tuple[np.ndarray, np.ndarray]], learning_rate: float) -> None`

Plain gradient descent: for every layer, `W = W - learning_rate * grad_w` and `b = b - learning_rate * grad_b`. It builds new arrays rather than editing in place.

### `class Adam`

Gradient descent with a per-parameter memory. Two running averages per parameter: `m` (the recent average gradient, a momentum) and `v` (the recent average squared gradient, a scale). Dividing `m` by `sqrt(v)` means every parameter moves at roughly the learning rate no matter how big or small its raw gradient is, which makes one learning rate work across layers of very different size.

#### `__init__(self, network: MLP, learning_rate: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> None`

Stores the network and the four constants. `self.m` and `self.v` are lists, one `(for_weights, for_biases)` tuple of zero arrays per layer. `self.t = 0` counts steps for bias correction.

#### `step(self, grads: list[tuple[np.ndarray, np.ndarray]]) -> None`

Increments `t`, then for every weight array and every bias array, with gradient `g`:

```
m = beta1 * m + (1 - beta1) * g
v = beta2 * v + (1 - beta2) * g * g
m_hat = m / (1 - beta1 ** t)
v_hat = v / (1 - beta2 ** t)
parameter = parameter - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
```

The `m_hat` and `v_hat` lines are *bias correction*. Because `m` and `v` start at zero they are too small for the first few steps; dividing by `1 - beta ** t` (which is `0.1` and `0.001` at `t = 1`, and tends to 1) scales them back up.

One consequence worth knowing: on the very first step `m_hat = g` and `v_hat = g^2`, so the update is `learning_rate * g / (|g| + epsilon)`. Every parameter moves by almost exactly `learning_rate` in the direction of its gradient, regardless of the gradient's size.

The code updates the weight array (`slot 0`) and the bias array (`slot 1`) of each layer with the same five lines, storing the new `m` and `v` back into the right half of the layer's tuple.

## How to use it / experiment

**A finite-difference gradient check.** This is the same check as `test_mlp_backward_matches_finite_differences` in [tests/test_research.py](../../tests/test_research.py). Fit a `5 -> 4 -> 3` network to random targets with a mean-squared-error loss, get the analytic gradient from `backward`, then nudge one weight by hand.

```python
import numpy as np
from hunger_games.brain.mlp import MLP

rng = np.random.default_rng(0)
x = rng.normal(size=(6, 5))          # 6 examples, 5 inputs
target = rng.normal(size=(6, 3))     # 6 targets, 3 outputs
net = MLP([5, 4, 3], "tanh", rng=np.random.default_rng(1))

out, cache = net.forward_cached(x)
# Loss = mean((out - target)^2). Its gradient at the outputs is 2 (out - target) / out.size.
grads = net.backward(cache, 2.0 * (out - target) / out.size)

eps = 1e-6
i, j = 1, 2
for layer in range(2):
    w = net.weights[layer]
    w[i, j] += eps
    plus = np.mean((net.forward(x) - target) ** 2)
    w[i, j] -= 2 * eps
    minus = np.mean((net.forward(x) - target) ** 2)
    w[i, j] += eps                    # restore
    numeric = (plus - minus) / (2 * eps)
    print(layer, numeric, grads[layer][0][i, j])
```

Output:

```
0 -0.05904760647 -0.05904760636
1  0.02796974119  0.02796974130
```

The two columns agree to nine decimal places. The test loops over all five activations and demands agreement within `1e-5`.

**Train with Adam.** The second research test: a `4 -> 8 -> 2` network learns a random linear map.

```python
rng = np.random.default_rng(0)
x = rng.normal(size=(32, 4))
target = x @ rng.normal(size=(4, 2))
net = MLP([4, 8, 2], "tanh", rng=rng)
optimizer = Adam(net, 0.01)
for step in range(200):
    out, cache = net.forward_cached(x)
    loss = np.mean((out - target) ** 2)
    optimizer.step(net.backward(cache, 2.0 * (out - target) / out.size))
    if step % 50 == 0:
        print(step, round(loss, 4))
```

The loss goes `4.397` at step 0, `0.236` at step 50, and `0.034` after 200 steps.

**Swap in plain gradient descent.** Replace `optimizer.step(grads)` with `net.apply_gradients(grads, 0.05)`. It converges too, but you will need to tune the learning rate by hand for each problem, which is the problem Adam solves.

**Look inside a brain's network.** `NeuralBrain.network` is an `MLP`, so everything here works on a tribute's brain: `brain.network.hidden_activations(perception.to_vector())` is what the dashboard draws.

**How the trainers use it.** The genetic algorithm ([../training/genetic.md](../training/genetic.md)) only calls `genome()` and `set_genome()`; it never touches `backward`. The REINFORCE trainer builds a value network `MLP([VECTOR_SIZE, 32, 1], activation, "xavier_uniform")`, calls `forward_cached` on a batch of states, computes the loss gradient at the logits and at the value, calls `backward` on both networks, clips the gradients, and hands them to two `Adam` instances.

## Gotchas

- `backward` needs the cache from the *same* `forward_cached` call, on the *same* weights. Change a weight and then call `backward` with an old cache and the numbers are silently wrong.
- Gradients are summed over the batch, not averaged. Scale the output gradient by `1 / n` yourself for a mean loss.
- `set_genome` stores views. `np.asarray(genome, dtype=float)` returns the caller's array unchanged when it is already float64, so afterwards `net.weights[0]` and the caller's array share memory. Changing one changes the other. Pass `genome.copy()` if that matters. `genome()` builds a fresh vector, so reading is always safe.
- `apply_gradients` and `Adam.step` replace the weight arrays with new ones, which breaks any view relationship from a previous `set_genome`. In-place edits (`net.weights[0][i, j] += eps`, as in the test) do not.
- `hidden_activations` includes the output layer. If you want only the hidden ones, drop the last entry.
- A wrong-length genome raises `ValueError` with both sizes in the message. The usual cause is a `NeuralConfig` that differs from the one the genome was trained with.
- `Adam` keeps its `m`, `v` and `t` state tied to one network. If you `set_genome` a completely different policy into that network mid-training, the momentum still points where the old policy was going.
- There is no gradient clipping here. The REINFORCE trainer clips before calling `Adam.step`; if you use `Adam` on your own and the loss explodes, that is the first thing to add.
- `forward` on an integer array works (it is converted to float), but an input of the wrong length raises a numpy shape error from the first `@`.
