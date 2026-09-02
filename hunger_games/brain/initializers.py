"""brain/initializers.py - every way to choose a neural network's starting weights.

Before a network is trained, every weight needs a starting value. The choice
matters more than it seems: all zeros makes every neuron identical forever,
too large makes the activations saturate, too small makes signals fade.
Each function below builds one weight matrix of shape (fan_in, fan_out),
where fan_in is the number of inputs to the layer and fan_out the number of
outputs. The dashboard lets you pick any of them by name.
"""

# numpy for the matrices and the random draws.
import numpy as np

# A random generator type hint.
Generator = np.random.Generator


# ---------------------------------------------------------------- constants


def zeros(rng: Generator, fan_in: int, fan_out: int, scale: float, sparsity: float) -> np.ndarray:
    """Every weight is 0. Symmetric: every neuron learns the same thing. For comparison only."""
    # A matrix of zeros.
    return np.zeros((fan_in, fan_out))


def ones(rng: Generator, fan_in: int, fan_out: int, scale: float, sparsity: float) -> np.ndarray:
    """Every weight is 1. Also symmetric, and the sums quickly saturate. For comparison only."""
    # A matrix of ones.
    return np.ones((fan_in, fan_out))


def constant(rng: Generator, fan_in: int, fan_out: int, scale: float, sparsity: float) -> np.ndarray:
    """Every weight is the same user-chosen value (`scale`)."""
    # A matrix filled with the constant.
    return np.full((fan_in, fan_out), scale)


# ------------------------------------------------------------ simple random


def uniform(rng: Generator, fan_in: int, fan_out: int, scale: float, sparsity: float) -> np.ndarray:
    """Weights drawn evenly between -scale and +scale."""
    # Flat distribution over the interval.
    return rng.uniform(-scale, scale, (fan_in, fan_out))


def normal(rng: Generator, fan_in: int, fan_out: int, scale: float, sparsity: float) -> np.ndarray:
    """Weights drawn from a bell curve centred on 0 with standard deviation `scale`."""
    # Gaussian distribution.
    return rng.normal(0.0, scale, (fan_in, fan_out))


# ----------------------------------------------------------- scaled variance


def xavier_uniform(rng: Generator, fan_in: int, fan_out: int, scale: float, sparsity: float) -> np.ndarray:
    """Glorot's rule: uniform with limit sqrt(6 / (fan_in + fan_out)). Good for tanh and sigmoid."""
    # The limit keeps the variance of the signal the same going in and out of the layer.
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    # Flat distribution within that limit.
    return rng.uniform(-limit, limit, (fan_in, fan_out))


def xavier_normal(rng: Generator, fan_in: int, fan_out: int, scale: float, sparsity: float) -> np.ndarray:
    """Glorot's rule with a bell curve: standard deviation sqrt(2 / (fan_in + fan_out))."""
    # The matching standard deviation for a normal distribution.
    std = np.sqrt(2.0 / (fan_in + fan_out))
    # Gaussian distribution with that spread.
    return rng.normal(0.0, std, (fan_in, fan_out))


def he_uniform(rng: Generator, fan_in: int, fan_out: int, scale: float, sparsity: float) -> np.ndarray:
    """Kaiming He's rule: uniform with limit sqrt(6 / fan_in). Designed for ReLU."""
    # ReLU throws away half the signal, so the limit is larger than Xavier's and ignores fan_out.
    limit = np.sqrt(6.0 / fan_in)
    # Flat distribution within that limit.
    return rng.uniform(-limit, limit, (fan_in, fan_out))


def he_normal(rng: Generator, fan_in: int, fan_out: int, scale: float, sparsity: float) -> np.ndarray:
    """Kaiming He's rule with a bell curve: standard deviation sqrt(2 / fan_in)."""
    # The matching standard deviation.
    std = np.sqrt(2.0 / fan_in)
    # Gaussian distribution with that spread.
    return rng.normal(0.0, std, (fan_in, fan_out))


def lecun_uniform(rng: Generator, fan_in: int, fan_out: int, scale: float, sparsity: float) -> np.ndarray:
    """LeCun's rule: uniform with limit sqrt(3 / fan_in). Pairs with SELU."""
    # Variance 1 / fan_in, which is what SELU's self-normalising maths assumes.
    limit = np.sqrt(3.0 / fan_in)
    # Flat distribution within that limit.
    return rng.uniform(-limit, limit, (fan_in, fan_out))


def lecun_normal(rng: Generator, fan_in: int, fan_out: int, scale: float, sparsity: float) -> np.ndarray:
    """LeCun's rule with a bell curve: standard deviation sqrt(1 / fan_in)."""
    # The matching standard deviation.
    std = np.sqrt(1.0 / fan_in)
    # Gaussian distribution with that spread.
    return rng.normal(0.0, std, (fan_in, fan_out))


# ------------------------------------------------------ structural / matrix


def orthogonal(rng: Generator, fan_in: int, fan_out: int, scale: float, sparsity: float) -> np.ndarray:
    """Columns (or rows) are perpendicular unit vectors, so the layer neither
    stretches nor shrinks the signal. Helps deep and recurrent networks.
    """
    # Start with a random Gaussian matrix.
    random_matrix = rng.normal(0.0, 1.0, (fan_in, fan_out))
    # QR decomposition splits it into an orthogonal part (Q) and the rest (R).
    q, r = np.linalg.qr(random_matrix)
    # Fix the signs so the result is uniformly distributed over orthogonal matrices.
    q = q * np.sign(np.diag(r))
    # QR only gives min(fan_in, fan_out) orthogonal columns, so pad if the layer is wider.
    if q.shape != (fan_in, fan_out):
        # Build a fresh matrix of the right shape.
        padded = np.zeros((fan_in, fan_out))
        # Copy in as much of Q as fits.
        padded[: q.shape[0], : q.shape[1]] = q
        # Use the padded version.
        q = padded
    # Return the orthogonal matrix.
    return q


def identity(rng: Generator, fan_in: int, fan_out: int, scale: float, sparsity: float) -> np.ndarray:
    """A 1 on the diagonal and 0 elsewhere, so the layer passes its input straight through."""
    # np.eye handles non-square shapes by putting 1s on the leading diagonal.
    return np.eye(fan_in, fan_out)


def sparse(rng: Generator, fan_in: int, fan_out: int, scale: float, sparsity: float) -> np.ndarray:
    """Only a fraction (`sparsity`) of the weights are non-zero, like sparsely wired neurons."""
    # Start from a normal draw.
    weights = rng.normal(0.0, max(scale, 1e-3), (fan_in, fan_out))
    # A mask that keeps roughly `sparsity` of the entries.
    mask = rng.random((fan_in, fan_out)) < sparsity
    # Zero out everything the mask rejects.
    return weights * mask


# ------------------------------------------------------------------ registry

# Name -> function, so the config and the dashboard can pick one by string.
INITIALIZERS = {
    "zeros": zeros,
    "ones": ones,
    "constant": constant,
    "uniform": uniform,
    "normal": normal,
    "xavier_uniform": xavier_uniform,
    "xavier_normal": xavier_normal,
    "he_uniform": he_uniform,
    "he_normal": he_normal,
    "lecun_uniform": lecun_uniform,
    "lecun_normal": lecun_normal,
    "orthogonal": orthogonal,
    "identity": identity,
    "sparse": sparse,
}

# One-line descriptions shown as tooltips in the dashboard.
INITIALIZER_NOTES = {
    "zeros": "All weights 0. Every neuron identical; the network cannot learn. Comparison only.",
    "ones": "All weights 1. Also symmetric and saturating. Comparison only.",
    "constant": "All weights equal to init_scale.",
    "uniform": "Even spread between -init_scale and +init_scale.",
    "normal": "Bell curve, mean 0, standard deviation init_scale.",
    "xavier_uniform": "Glorot uniform, scaled by fan_in + fan_out. Best with tanh or sigmoid.",
    "xavier_normal": "Glorot normal, scaled by fan_in + fan_out. Best with tanh or sigmoid.",
    "he_uniform": "Kaiming uniform, scaled by fan_in. Best with relu / leaky_relu.",
    "he_normal": "Kaiming normal, scaled by fan_in. Best with relu / leaky_relu.",
    "lecun_uniform": "LeCun uniform, variance 1/fan_in. Best with selu.",
    "lecun_normal": "LeCun normal, variance 1/fan_in. Best with selu.",
    "orthogonal": "Perpendicular unit columns; preserves signal size. Good for deep nets.",
    "identity": "Passes input straight through. Only meaningful for square layers.",
    "sparse": "Only a `sparsity` fraction of weights are non-zero.",
}


def initialize(
    name: str, rng: Generator, fan_in: int, fan_out: int, scale: float = 0.05, sparsity: float = 0.1
) -> np.ndarray:
    """Build one weight matrix using the named initializer."""
    # Unknown names are a configuration mistake worth a clear message.
    if name not in INITIALIZERS:
        raise KeyError(f"Unknown initializer '{name}'. Choose from: {', '.join(INITIALIZERS)}")
    # Call the chosen function.
    return INITIALIZERS[name](rng, fan_in, fan_out, scale, sparsity)


# ------------------------------------------------------------- activations


def relu(x: np.ndarray) -> np.ndarray:
    """Rectified linear unit: negatives become 0, positives pass through."""
    # Element-wise maximum with zero.
    return np.maximum(0.0, x)


def leaky_relu(x: np.ndarray) -> np.ndarray:
    """Like relu, but negatives are scaled by 0.01 instead of zeroed, so they never fully die."""
    # Keep a small slope on the negative side.
    return np.where(x > 0.0, x, 0.01 * x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Squash to the range 0..1."""
    # The logistic function.
    return 1.0 / (1.0 + np.exp(-x))


def selu(x: np.ndarray) -> np.ndarray:
    """Scaled exponential linear unit, which keeps activations self-normalised with LeCun init."""
    # The two constants from the SELU paper.
    alpha, lam = 1.6732632423543772, 1.0507009873554805
    # Positive side is scaled linear; negative side is a scaled exponential curve.
    return lam * np.where(x > 0.0, x, alpha * (np.exp(np.minimum(x, 0.0)) - 1.0))


# Name -> function for the activation between layers.
ACTIVATIONS = {
    "tanh": np.tanh,
    "relu": relu,
    "leaky_relu": leaky_relu,
    "sigmoid": sigmoid,
    "selu": selu,
}


# ---------------------------------------------------- activation derivatives


def _tanh_derivative(z: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Slope of tanh, written in terms of its output: 1 - a^2."""
    # Uses the activation value, which is cheaper than recomputing tanh.
    return 1.0 - a * a


def _relu_derivative(z: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Slope of relu: 1 where the input was positive, else 0."""
    # A step function.
    return (z > 0.0).astype(float)


def _leaky_relu_derivative(z: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Slope of leaky relu: 1 where positive, 0.01 where negative."""
    # Never fully zero, so no neuron can die.
    return np.where(z > 0.0, 1.0, 0.01)


def _sigmoid_derivative(z: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Slope of the sigmoid in terms of its output: a (1 - a)."""
    # Largest at a = 0.5, tiny near 0 and 1.
    return a * (1.0 - a)


def _selu_derivative(z: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Slope of selu: lambda where positive, lambda * alpha * exp(z) where negative."""
    # The two constants from the SELU paper.
    alpha, lam = 1.6732632423543772, 1.0507009873554805
    # Piecewise slope.
    return np.where(z > 0.0, lam, lam * alpha * np.exp(np.minimum(z, 0.0)))


# Name -> derivative function taking (pre-activation z, activation a). Needed for backpropagation.
ACTIVATION_DERIVATIVES = {
    "tanh": _tanh_derivative,
    "relu": _relu_derivative,
    "leaky_relu": _leaky_relu_derivative,
    "sigmoid": _sigmoid_derivative,
    "selu": _selu_derivative,
}
