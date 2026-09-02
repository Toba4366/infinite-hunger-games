"""brain/neural.py - the neural-network brain.

The network (an `MLP` from brain/mlp.py) reads the perception vector and
produces one score per item on a fixed action menu. It starts untrained.
Two things can train it: the genetic algorithm (training/genetic.py), which
evolves the flat `genome()`, and the policy-gradient learner
(training/reinforce.py), which backpropagates rewards through the same
network. The layer widths, activation and initializer come from
`NeuralConfig` in config.py, which the dashboard edits.
"""

# numpy for the maths.
import numpy as np

# The menu of actions and the compass directions.
from hunger_games.actions import DIRECTIONS, SIMPLE_ACTIONS, Action, ActionType

# The base class.
from hunger_games.brain.base import Brain

# The maths engine.
from hunger_games.brain.mlp import MLP

# The neural settings.
from hunger_games.config import NeuralConfig

# The perception type and the size of its flattened vector.
from hunger_games.perception import VECTOR_SIZE, Perception

# The network chooses from this menu: 6 simple actions, attack, flee, 8 moves.
MENU_SIZE = len(SIMPLE_ACTIONS) + 2 + len(DIRECTIONS)
# Index of the "attack nearest" menu item.
ATTACK_INDEX = len(SIMPLE_ACTIONS)
# Index of the "flee nearest" menu item.
FLEE_INDEX = ATTACK_INDEX + 1
# Index of the first "move in direction" menu item.
FIRST_MOVE_INDEX = FLEE_INDEX + 1
# Human-readable names of the menu items, in order (used by the dashboard and the research plots).
MENU_NAMES = (
    [action.kind.value for action in SIMPLE_ACTIONS]
    + ["attack", "flee"]
    + [f"move {name}" for name in ("up-left", "up", "up-right", "left", "right", "down-left", "down", "down-right")]
)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Turn raw scores into probabilities that sum to one."""
    # Divide by temperature; subtract the max for numerical safety.
    scaled = np.asarray(logits, dtype=float) / max(temperature, 1e-6)
    # Exponentiate relative to the maximum (per row if batched).
    weights = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
    # Normalise.
    return weights / weights.sum(axis=-1, keepdims=True)


class NeuralBrain(Brain):
    """Perception vector -> hidden layers -> one score per action."""

    # Label for the results CSV.
    name = "neural"

    def __init__(
        self,
        chaos: float = 0.0,
        config: NeuralConfig | None = None,
        genome: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Build a network from the config (or load a genome)."""
        # Store the chaos dial.
        super().__init__(chaos)
        # Use the given settings or the defaults.
        self.config = config if config is not None else NeuralConfig()
        # Layer sizes: input, each hidden layer, then the output menu.
        self.layer_sizes = [VECTOR_SIZE, *self.config.hidden_layers, MENU_SIZE]
        # The network itself.
        self.network = MLP(
            self.layer_sizes,
            self.config.activation,
            self.config.initializer,
            self.config.init_scale,
            self.config.sparsity,
            rng,
        )
        # The last decision's probabilities, kept so the dashboard can show what the network "thought".
        self.last_probabilities: np.ndarray | None = None
        # The last chosen menu index (the reinforcement learner reads this).
        self.last_index: int = 0
        # Load a genome if one was supplied.
        if genome is not None:
            self.set_genome(genome)

    # ---------------------------------------------------------- genome API

    @property
    def parameter_count(self) -> int:
        """How many numbers the genome holds."""
        # Delegate.
        return self.network.parameter_count

    @property
    def weights(self) -> list[np.ndarray]:
        """The weight matrices (a view into the network)."""
        # Delegate.
        return self.network.weights

    @property
    def biases(self) -> list[np.ndarray]:
        """The bias vectors (a view into the network)."""
        # Delegate.
        return self.network.biases

    def genome(self) -> np.ndarray:
        """All weights and biases flattened into one vector."""
        # Delegate.
        return self.network.genome()

    def set_genome(self, genome: np.ndarray) -> None:
        """Load a flat vector of weights and biases."""
        # Delegate (raises a clear error on a size mismatch).
        self.network.set_genome(genome)

    # ---------------------------------------------------------- deciding

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Run the network: return one raw score ("logit") per menu item."""
        # Delegate.
        return self.network.forward(inputs)

    def probabilities(self, logits: np.ndarray) -> np.ndarray:
        """The chance of each menu item given the chaos dial (a one-hot at chaos 0)."""
        # No chaos: the best item is certain.
        if self.chaos <= 0.0:
            # A one-hot vector.
            certain = np.zeros(MENU_SIZE)
            # On the best item.
            certain[int(np.argmax(logits))] = 1.0
            return certain
        # Otherwise a softmax with chaos as the temperature.
        return softmax(logits, self.chaos)

    def decide_index(self, perception: Perception, rng: np.random.Generator) -> int:
        """Pick a menu index (the reinforcement learner needs the index, not just the action)."""
        # Scores.
        logits = self.forward(perception.to_vector())
        # Probabilities.
        probabilities = self.probabilities(logits)
        # Remember for the dashboard.
        self.last_probabilities = probabilities
        # Certain: take it; otherwise sample.
        index = int(np.argmax(probabilities)) if self.chaos <= 0.0 else int(rng.choice(MENU_SIZE, p=probabilities))
        # Remember it.
        self.last_index = index
        # Done.
        return index

    def decide(self, perception: Perception, rng: np.random.Generator) -> Action:
        """Run the network forward and turn the chosen output into an action."""
        # Choose an index.
        index = self.decide_index(perception, rng)
        # Convert it into a concrete action.
        return self.menu_to_action(index, perception)

    @staticmethod
    def menu_to_action(index: int, perception: Perception) -> Action:
        """Map a menu index to an `Action`, filling in targets from the perception."""
        # Simple actions map straight across.
        if index < len(SIMPLE_ACTIONS):
            return SIMPLE_ACTIONS[index]
        # The nearest other player, if any.
        threat = perception.nearest_threat
        # "Attack nearest": attack if in reach, close in if not, rest if alone.
        if index == ATTACK_INDEX:
            # Alone: nothing to attack.
            if threat is None:
                return Action(ActionType.REST)
            # Within reach: attack.
            if threat.distance <= perception.reach:
                return Action.attack(threat.player_id)
            # Otherwise step toward them.
            return Action.move(*threat.direction_toward())
        # "Flee nearest": step away if anyone is in sight, rest otherwise.
        if index == FLEE_INDEX:
            # Alone: nothing to flee from.
            if threat is None:
                return Action(ActionType.REST)
            # Step away.
            return Action.flee(*threat.direction_away())
        # Everything else is a move in one of the eight directions.
        dx, dy = DIRECTIONS[index - FIRST_MOVE_INDEX]
        # Build the move.
        return Action.move(dx, dy)

    @staticmethod
    def action_to_menu_index(action: Action) -> int:
        """The reverse of menu_to_action: which menu item an action corresponds to (for imitation labels)."""
        # Simple actions map straight across.
        for index, simple in enumerate(SIMPLE_ACTIONS):
            if action.kind is simple.kind:
                return index
        # Attack and flee.
        if action.kind is ActionType.ATTACK:
            return ATTACK_INDEX
        if action.kind is ActionType.FLEE:
            return FLEE_INDEX
        # A move: find its direction; a zero move is a rest.
        if (action.dx, action.dy) in DIRECTIONS:
            return FIRST_MOVE_INDEX + DIRECTIONS.index((action.dx, action.dy))
        # Anything else counts as resting.
        return 0

    def describe(self) -> str:
        """A one-line summary for the dashboard."""
        # Delegate.
        return self.network.describe()
