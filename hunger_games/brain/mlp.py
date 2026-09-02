"""brain/mlp.py - a multi-layer perceptron in plain numpy, with backpropagation.

This is the maths engine behind the neural brain (brain/neural.py) and the
value network used by reinforcement learning (training/reinforce.py). It
does three things: run inputs forward through the layers, run gradients
backward through them (the chain rule, layer by layer), and expose all its
numbers as one flat genome vector for the genetic algorithm.
"""

# numpy for the matrices.
import numpy as np

# Initializers, activations and their derivatives by name.
from hunger_games.brain.initializers import ACTIVATION_DERIVATIVES, ACTIVATIONS, initialize


class MLP:
    """Fully connected layers with one activation between them and a raw (linear) output."""

    def __init__(
        self,
        layer_sizes: list[int],
        activation: str = "tanh",
        initializer: str = "xavier_uniform",
        init_scale: float = 0.05,
        sparsity: float = 0.1,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Build the layers with freshly initialised weights and zero biases."""
        # Sizes of every layer, input first, output last.
        self.layer_sizes = list(layer_sizes)
        # The activation's name (kept for describe()).
        self.activation_name = activation
        # The activation function.
        self.activation = ACTIVATIONS[activation]
        # Its derivative, for backpropagation.
        self.activation_derivative = ACTIVATION_DERIVATIVES[activation]
        # The initializer's name (kept for describe()).
        self.initializer_name = initializer
        # A generator for the starting weights.
        rng = rng if rng is not None else np.random.default_rng()
        # One weight matrix per pair of neighbouring layers.
        self.weights = [
            initialize(initializer, rng, fan_in, fan_out, init_scale, sparsity)
            for fan_in, fan_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:], strict=False)
        ]
        # One bias vector per layer after the input, all starting at zero.
        self.biases = [np.zeros(size) for size in self.layer_sizes[1:]]

    # ------------------------------------------------------------ shapes

    @property
    def parameter_count(self) -> int:
        """How many numbers the network holds."""
        # Weights plus biases.
        return sum(w.size for w in self.weights) + sum(b.size for b in self.biases)

    def describe(self) -> str:
        """A one-line summary, e.g. '50 -> 16 -> 16, tanh, xavier_uniform, 1088 params'."""
        # Arrows between the sizes.
        shape = " -> ".join(str(size) for size in self.layer_sizes)
        # The rest.
        return f"{shape}, {self.activation_name}, {self.initializer_name}, {self.parameter_count} params"

    # ------------------------------------------------------------ genome

    def genome(self) -> np.ndarray:
        """All weights and biases flattened into one vector, layer by layer."""
        # Collect the pieces.
        parts = []
        # Weights then bias for each layer.
        for weight, bias in zip(self.weights, self.biases, strict=False):
            parts.append(weight.ravel())
            parts.append(bias)
        # Join.
        return np.concatenate(parts)

    def set_genome(self, genome: np.ndarray) -> None:
        """Split a flat vector back into the weight matrices and bias vectors."""
        # As a float array.
        genome = np.asarray(genome, dtype=float)
        # A wrong-sized genome usually means a different architecture; say so clearly.
        if genome.size != self.parameter_count:
            raise ValueError(f"Genome has {genome.size} values but this network needs {self.parameter_count}")
        # Where we are in the flat vector.
        cursor = 0
        # Refill each layer.
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases, strict=False)):
            # Weights.
            self.weights[index] = genome[cursor : cursor + weight.size].reshape(weight.shape)
            # Advance.
            cursor += weight.size
            # Bias.
            self.biases[index] = genome[cursor : cursor + bias.size]
            # Advance.
            cursor += bias.size

    # ----------------------------------------------------------- forward

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Inputs (one vector, or a batch as rows) to raw outputs."""
        # Only the outputs are needed.
        return self.forward_cached(inputs)[0]

    def forward_cached(self, inputs: np.ndarray) -> tuple[np.ndarray, list]:
        """Forward pass that also returns what backprop needs.

        The cache holds, for every layer, the layer's input, its pre-activation
        sum `z`, and its activation `a`.
        """
        # Start with the inputs.
        signal = np.asarray(inputs, dtype=float)
        # The cache.
        cache = []
        # The last layer is the output layer, which gets no activation.
        last = len(self.weights) - 1
        # Push through every layer.
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases, strict=False)):
            # Weighted sum plus bias.
            z = signal @ weight + bias
            # Squash between hidden layers, leave the output raw.
            a = self.activation(z) if index != last else z
            # Remember for backprop.
            cache.append((signal, z, a))
            # Feed forward.
            signal = a
        # Outputs and cache.
        return signal, cache

    def hidden_activations(self, inputs: np.ndarray) -> list[np.ndarray]:
        """The activation of every layer (hidden and output) for one input, for visualisation."""
        # Run forward.
        _, cache = self.forward_cached(inputs)
        # Pull the activations.
        return [a for _, _, a in cache]

    # ---------------------------------------------------------- backward

    def backward(self, cache: list, grad_outputs: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Backpropagate a gradient at the outputs down to every weight and bias.

        Returns one (grad_w, grad_b) pair per layer, summed over the batch. Divide by
        the batch size yourself if you want a mean.
        """
        # As a 2-D batch (rows) even for a single vector.
        grad = np.atleast_2d(np.asarray(grad_outputs, dtype=float))
        # Gradients per layer, filled from the back.
        grads: list[tuple[np.ndarray, np.ndarray]] = [None] * len(self.weights)  # type: ignore[list-item]
        # The last layer index.
        last = len(self.weights) - 1
        # Walk the layers backward.
        for index in range(last, -1, -1):
            # This layer's input, pre-activation and activation.
            layer_input, z, a = cache[index]
            # As 2-D batches.
            layer_input = np.atleast_2d(layer_input)
            # Hidden layers: multiply by the activation's slope (the output layer is linear).
            if index != last:
                grad = grad * self.activation_derivative(np.atleast_2d(z), np.atleast_2d(a))
            # Weight gradient: input transposed times the gradient.
            grad_w = layer_input.T @ grad
            # Bias gradient: sum over the batch.
            grad_b = grad.sum(axis=0)
            # Store.
            grads[index] = (grad_w, grad_b)
            # Gradient with respect to this layer's input, for the layer below.
            grad = grad @ self.weights[index].T
        # Done.
        return grads

    def apply_gradients(self, grads: list[tuple[np.ndarray, np.ndarray]], learning_rate: float) -> None:
        """Plain gradient descent: subtract learning_rate times each gradient."""
        # Each layer.
        for index, (grad_w, grad_b) in enumerate(grads):
            self.weights[index] = self.weights[index] - learning_rate * grad_w
            self.biases[index] = self.biases[index] - learning_rate * grad_b


class Adam:
    """The Adam optimiser: gradient descent with per-parameter momentum and scaling."""

    def __init__(
        self, network: MLP, learning_rate: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8
    ) -> None:
        """Create the running averages for every parameter of the network."""
        # The network being trained.
        self.network = network
        # Step size.
        self.learning_rate = learning_rate
        # Momentum decay.
        self.beta1 = beta1
        # Scale decay.
        self.beta2 = beta2
        # Numerical safety.
        self.epsilon = epsilon
        # First moment (momentum) per parameter array.
        self.m = [(np.zeros_like(w), np.zeros_like(b)) for w, b in zip(network.weights, network.biases, strict=False)]
        # Second moment (scale) per parameter array.
        self.v = [(np.zeros_like(w), np.zeros_like(b)) for w, b in zip(network.weights, network.biases, strict=False)]
        # How many steps have been taken (for bias correction).
        self.t = 0

    def step(self, grads: list[tuple[np.ndarray, np.ndarray]]) -> None:
        """Update every parameter from its gradient."""
        # Count the step.
        self.t += 1
        # Each layer.
        for index, (grad_w, grad_b) in enumerate(grads):
            # Update each of the two arrays (weight, bias) the same way.
            for slot, grad in ((0, grad_w), (1, grad_b)):
                # The running averages.
                m = self.beta1 * self.m[index][slot] + (1.0 - self.beta1) * grad
                v = self.beta2 * self.v[index][slot] + (1.0 - self.beta2) * grad * grad
                # Store them back.
                self.m[index] = (m, self.m[index][1]) if slot == 0 else (self.m[index][0], m)
                self.v[index] = (v, self.v[index][1]) if slot == 0 else (self.v[index][0], v)
                # Bias-corrected estimates.
                m_hat = m / (1.0 - self.beta1**self.t)
                v_hat = v / (1.0 - self.beta2**self.t)
                # The update.
                update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
                # Apply to the right array.
                if slot == 0:
                    self.network.weights[index] = self.network.weights[index] - update
                else:
                    self.network.biases[index] = self.network.biases[index] - update
