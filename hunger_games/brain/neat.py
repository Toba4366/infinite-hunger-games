"""brain/neat.py - NEAT genomes: networks whose shape evolves, not just their weights.

NEAT (NeuroEvolution of Augmenting Topologies, Stanley and Miikkulainen
2002) is the algorithm the Monopoly video uses. A genome is a list of
node genes and connection genes. Networks start minimal (inputs wired to
outputs) and grow by mutation: a new connection between two nodes, or a
new node spliced into an existing connection. Every new connection gets
an innovation number, a global counter, so two genomes can be lined up
gene by gene for crossover and for measuring how different they are,
which is what keeps species apart.

The network is kept feed-forward: a connection is only allowed from a
node of lower depth to one of higher depth, so evaluation is a single
pass in depth order.
"""

# Genes as small records.
from dataclasses import dataclass, field

# numpy for the maths.
import numpy as np

# The action menu, the base class and the softmax.
from hunger_games.actions import Action
from hunger_games.brain.base import Brain
from hunger_games.brain.initializers import ACTIVATIONS
from hunger_games.brain.neural import MENU_SIZE, NeuralBrain, softmax

# The perception.
from hunger_games.perception import VECTOR_SIZE, Perception


@dataclass
class NodeGene:
    """One neuron."""

    # Unique id.
    id: int
    # "input", "bias", "hidden" or "output".
    kind: str
    # Activation name for hidden and output nodes.
    activation: str = "tanh"


@dataclass
class ConnectionGene:
    """One weighted link between two nodes."""

    # The global innovation number.
    innovation: int
    # Source node id.
    src: int
    # Destination node id.
    dst: int
    # Weight.
    weight: float
    # Disabled connections are kept in the genome (for crossover) but not evaluated.
    enabled: bool = True


@dataclass
class NeatConfig:
    """The mutation and speciation knobs of NEAT."""

    # Chance of adding a node by splitting a connection.
    add_node_rate: float = 0.03
    # Chance of adding a connection.
    add_connection_rate: float = 0.08
    # Chance that each weight is mutated.
    weight_mutate_rate: float = 0.8
    # Of mutated weights, the chance of a small nudge (else a fresh random value).
    weight_perturb_rate: float = 0.9
    # Size of the nudge.
    weight_perturb_scale: float = 0.1
    # Range of a fresh random weight.
    weight_range: float = 1.0
    # Chance of re-enabling a disabled connection.
    enable_rate: float = 0.05
    # Speciation distance weights: excess genes, disjoint genes, weight difference.
    c_excess: float = 1.0
    c_disjoint: float = 1.0
    c_weights: float = 0.4
    # Genomes closer than this belong to one species.
    compatibility_threshold: float = 3.0
    # Activation of hidden nodes.
    activation: str = "tanh"


class InnovationTracker:
    """Hands out innovation numbers and node ids, remembering (src, dst) pairs within a generation."""

    def __init__(self, next_innovation: int = 0, next_node: int = 0) -> None:
        """Start the counters."""
        # Next innovation number.
        self.next_innovation = next_innovation
        # Next node id.
        self.next_node = next_node
        # Innovations handed out for (src, dst) this generation, so the same mutation gets the same number.
        self.seen: dict[tuple[int, int], int] = {}

    def innovation(self, src: int, dst: int) -> int:
        """The innovation number for a connection (reused within the generation)."""
        # Reuse.
        if (src, dst) in self.seen:
            return self.seen[(src, dst)]
        # New.
        number = self.next_innovation
        self.next_innovation += 1
        self.seen[(src, dst)] = number
        return number

    def node(self) -> int:
        """A fresh node id."""
        # Next.
        number = self.next_node
        self.next_node += 1
        return number

    def reset_generation(self) -> None:
        """Forget this generation's (src, dst) pairs."""
        # Clear.
        self.seen = {}


@dataclass
class NeatGenome:
    """A NEAT genome: nodes, connections, and its fitness once evaluated."""

    # The nodes.
    nodes: list[NodeGene]
    # The connections.
    connections: list[ConnectionGene]
    # Fitness after evaluation.
    fitness: float = 0.0
    # Which species it belongs to.
    species: int = -1
    # Cached node depths (computed on demand).
    _depths: dict[int, int] | None = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------ building

    @classmethod
    def minimal(
        cls, inputs: int, outputs: int, rng: np.random.Generator, config: NeatConfig, tracker: InnovationTracker
    ) -> "NeatGenome":
        """The starting genome: inputs and a bias fully connected to the outputs."""
        # Input nodes 0..inputs-1, bias node, then outputs.
        nodes = [NodeGene(i, "input") for i in range(inputs)]
        nodes.append(NodeGene(inputs, "bias"))
        nodes += [NodeGene(inputs + 1 + o, "output", config.activation) for o in range(outputs)]
        # Node ids continue after these.
        tracker.next_node = max(tracker.next_node, inputs + 1 + outputs)
        # Full connections with small random weights.
        connections = []
        for src in range(inputs + 1):
            for o in range(outputs):
                dst = inputs + 1 + o
                connections.append(
                    ConnectionGene(tracker.innovation(src, dst), src, dst, float(rng.uniform(-0.5, 0.5)))
                )
        # Done.
        return cls(nodes, connections)

    def copy(self) -> "NeatGenome":
        """A deep copy."""
        # Copy genes.
        return NeatGenome(
            [NodeGene(n.id, n.kind, n.activation) for n in self.nodes],
            [ConnectionGene(c.innovation, c.src, c.dst, c.weight, c.enabled) for c in self.connections],
            self.fitness,
            self.species,
        )

    # ------------------------------------------------------------ structure

    def depths(self) -> dict[int, int]:
        """Depth of every node: inputs and bias 0, others one more than their deepest enabled source."""
        # Cached.
        if self._depths is not None:
            return self._depths
        # Start.
        depth = {n.id: 0 for n in self.nodes}
        # Enabled connections.
        enabled = [c for c in self.connections if c.enabled]
        # Relax until stable (the graph is acyclic by construction, so this terminates).
        changed = True
        rounds = 0
        while changed and rounds < len(self.nodes) + 1:
            changed = False
            rounds += 1
            for c in enabled:
                if depth[c.dst] < depth[c.src] + 1:
                    depth[c.dst] = depth[c.src] + 1
                    changed = True
        # Cache.
        self._depths = depth
        return depth

    def invalidate(self) -> None:
        """Forget cached depths after a structural change."""
        # Clear.
        self._depths = None

    @property
    def hidden_count(self) -> int:
        """How many hidden nodes there are."""
        # Count.
        return sum(1 for n in self.nodes if n.kind == "hidden")

    @property
    def enabled_count(self) -> int:
        """How many enabled connections there are."""
        # Count.
        return sum(1 for c in self.connections if c.enabled)

    # ------------------------------------------------------------ mutation

    def mutate(self, rng: np.random.Generator, config: NeatConfig, tracker: InnovationTracker) -> None:
        """Apply NEAT's mutations in place."""
        # Weights.
        for c in self.connections:
            if rng.random() < config.weight_mutate_rate:
                if rng.random() < config.weight_perturb_rate:
                    c.weight += float(rng.normal(0.0, config.weight_perturb_scale))
                else:
                    c.weight = float(rng.uniform(-config.weight_range, config.weight_range))
        # Re-enable.
        for c in self.connections:
            if not c.enabled and rng.random() < config.enable_rate:
                c.enabled = True
                self.invalidate()
        # Add a node.
        if rng.random() < config.add_node_rate:
            self._add_node(rng, config, tracker)
        # Add a connection.
        if rng.random() < config.add_connection_rate:
            self._add_connection(rng, config, tracker)

    def _add_node(self, rng: np.random.Generator, config: NeatConfig, tracker: InnovationTracker) -> None:
        """Split an enabled connection: src -> new -> dst (weights 1 and the old weight)."""
        # Candidates.
        enabled = [c for c in self.connections if c.enabled]
        if not enabled:
            return
        # Pick one and disable it.
        old = enabled[int(rng.integers(len(enabled)))]
        old.enabled = False
        # The new node.
        node = NodeGene(tracker.node(), "hidden", config.activation)
        self.nodes.append(node)
        # Two new connections.
        self.connections.append(ConnectionGene(tracker.innovation(old.src, node.id), old.src, node.id, 1.0))
        self.connections.append(ConnectionGene(tracker.innovation(node.id, old.dst), node.id, old.dst, old.weight))
        # Structure changed.
        self.invalidate()

    def _add_connection(self, rng: np.random.Generator, config: NeatConfig, tracker: InnovationTracker) -> None:
        """Connect two unconnected nodes, from lower depth to higher, so the network stays feed-forward."""
        # Depths.
        depth = self.depths()
        # Existing pairs.
        existing = {(c.src, c.dst) for c in self.connections}
        # Try a few random pairs.
        for _ in range(20):
            a = self.nodes[int(rng.integers(len(self.nodes)))]
            b = self.nodes[int(rng.integers(len(self.nodes)))]
            # Never into inputs or the bias, never out of outputs, never self.
            if b.kind in ("input", "bias") or a.kind == "output" or a.id == b.id:
                continue
            # Feed-forward only, and only if the pair is new.
            if depth[a.id] >= depth[b.id] and not (
                a.kind in ("input", "bias") and b.kind == "hidden" and depth[b.id] == 0
            ):
                continue
            if (a.id, b.id) in existing:
                continue
            # Add.
            self.connections.append(
                ConnectionGene(
                    tracker.innovation(a.id, b.id),
                    a.id,
                    b.id,
                    float(rng.uniform(-config.weight_range, config.weight_range)),
                )
            )
            self.invalidate()
            return

    # ------------------------------------------------------------ crossover

    def crossover(self, other: "NeatGenome", rng: np.random.Generator) -> "NeatGenome":
        """Combine with another genome; matching genes are picked at random, the rest come from the fitter parent."""
        # The fitter parent is `self` by convention (callers order them).
        mine = {c.innovation: c for c in self.connections}
        theirs = {c.innovation: c for c in other.connections}
        # Child connections.
        connections = []
        for innovation, c in mine.items():
            partner = theirs.get(innovation)
            source = c if partner is None or rng.random() < 0.5 else partner
            child = ConnectionGene(source.innovation, source.src, source.dst, source.weight, source.enabled)
            # A gene disabled in either parent has a chance of staying disabled.
            if partner is not None and (not c.enabled or not partner.enabled) and rng.random() < 0.75:
                child.enabled = False
            connections.append(child)
        # Child nodes: every node referenced.
        needed = {c.src for c in connections} | {c.dst for c in connections}
        by_id = {n.id: n for n in self.nodes}
        by_id.update({n.id: n for n in other.nodes if n.id not in by_id})
        nodes = [NodeGene(n.id, n.kind, n.activation) for n in by_id.values() if n.id in needed or n.kind != "hidden"]
        # Done.
        return NeatGenome(nodes, connections)

    def distance(self, other: "NeatGenome", config: NeatConfig) -> float:
        """NEAT's compatibility distance: excess, disjoint and average weight difference of matching genes."""
        # Innovations.
        mine = {c.innovation: c for c in self.connections}
        theirs = {c.innovation: c for c in other.connections}
        # Matching.
        matching = set(mine) & set(theirs)
        # The larger genome's highest innovation.
        max_mine = max(mine) if mine else 0
        max_theirs = max(theirs) if theirs else 0
        boundary = min(max_mine, max_theirs)
        # Excess: beyond the other genome's range; disjoint: within it but unmatched.
        unmatched = (set(mine) | set(theirs)) - matching
        excess = sum(1 for i in unmatched if i > boundary)
        disjoint = len(unmatched) - excess
        # Mean weight difference.
        weight_diff = float(np.mean([abs(mine[i].weight - theirs[i].weight) for i in matching])) if matching else 0.0
        # Normalised by size (NEAT uses 1 for small genomes).
        n = max(1, max(len(mine), len(theirs)))
        n = n if n >= 20 else 1
        # Distance.
        return config.c_excess * excess / n + config.c_disjoint * disjoint / n + config.c_weights * weight_diff

    # ------------------------------------------------------------ evaluate

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Evaluate the network on one input vector; returns the output node values."""
        # Depths give the evaluation order.
        depth = self.depths()
        # Values.
        value = {n.id: 0.0 for n in self.nodes}
        # Inputs and bias.
        input_nodes = [n for n in self.nodes if n.kind == "input"]
        for n, x in zip(input_nodes, inputs, strict=False):
            value[n.id] = float(x)
        for n in self.nodes:
            if n.kind == "bias":
                value[n.id] = 1.0
        # Incoming enabled connections per node.
        incoming: dict[int, list[ConnectionGene]] = {}
        for c in self.connections:
            if c.enabled:
                incoming.setdefault(c.dst, []).append(c)
        # Evaluate in depth order.
        for n in sorted(self.nodes, key=lambda node: depth[node.id]):
            if n.kind in ("input", "bias"):
                continue
            total = sum(value[c.src] * c.weight for c in incoming.get(n.id, []))
            value[n.id] = float(ACTIVATIONS[n.activation](np.asarray(total))) if n.kind == "hidden" else total
        # Outputs in id order.
        return np.asarray([value[n.id] for n in self.nodes if n.kind == "output"], dtype=float)

    def activations(self, inputs: np.ndarray) -> dict[int, float]:
        """Every node's value for one input, for the visualiser."""
        # Same as forward, but keep everything.
        depth = self.depths()
        value = {n.id: 0.0 for n in self.nodes}
        input_nodes = [n for n in self.nodes if n.kind == "input"]
        for n, x in zip(input_nodes, inputs, strict=False):
            value[n.id] = float(x)
        for n in self.nodes:
            if n.kind == "bias":
                value[n.id] = 1.0
        incoming: dict[int, list[ConnectionGene]] = {}
        for c in self.connections:
            if c.enabled:
                incoming.setdefault(c.dst, []).append(c)
        for n in sorted(self.nodes, key=lambda node: depth[node.id]):
            if n.kind in ("input", "bias"):
                continue
            total = sum(value[c.src] * c.weight for c in incoming.get(n.id, []))
            value[n.id] = float(ACTIVATIONS[n.activation](np.asarray(total))) if n.kind == "hidden" else total
        return value

    # ------------------------------------------------------------ saving

    def to_dict(self) -> dict:
        """JSON-friendly form."""
        # Plain lists.
        return {
            "nodes": [[n.id, n.kind, n.activation] for n in self.nodes],
            "connections": [[c.innovation, c.src, c.dst, c.weight, c.enabled] for c in self.connections],
            "fitness": self.fitness,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NeatGenome":
        """Rebuild from `to_dict`."""
        # Genes.
        return cls(
            [NodeGene(int(i), k, a) for i, k, a in data["nodes"]],
            [ConnectionGene(int(i), int(s), int(d), float(w), bool(e)) for i, s, d, w, e in data["connections"]],
            float(data.get("fitness", 0.0)),
        )


class NeatBrain(Brain):
    """A brain driven by a NEAT genome: perception in, action menu scores out."""

    # Label for the results CSV.
    name = "neat"

    def __init__(self, genome: NeatGenome, chaos: float = 0.0) -> None:
        """Wrap a genome."""
        # Chaos.
        super().__init__(chaos)
        # The genome.
        self.genome_data = genome
        # Last probabilities and index, for the dashboard and the learners.
        self.last_probabilities: np.ndarray | None = None
        self.last_index = 0

    def decide_index(self, perception: Perception, rng: np.random.Generator) -> int:
        """Score the menu and pick an index."""
        # Scores.
        logits = self.genome_data.forward(perception.to_vector())
        # Pad or trim to the menu size (a genome always has MENU_SIZE outputs, but be safe).
        logits = np.resize(logits, MENU_SIZE)
        # Probabilities.
        probabilities = softmax(logits, self.chaos) if self.chaos > 0 else np.eye(MENU_SIZE)[int(np.argmax(logits))]
        self.last_probabilities = probabilities
        # Pick.
        index = int(np.argmax(probabilities)) if self.chaos <= 0 else int(rng.choice(MENU_SIZE, p=probabilities))
        self.last_index = index
        return index

    def decide(self, perception: Perception, rng: np.random.Generator) -> Action:
        """Choose an action."""
        # Index then action.
        return NeuralBrain.menu_to_action(self.decide_index(perception, rng), perception)

    def genome(self) -> np.ndarray:
        """The connection weights as a flat vector (the shape is not part of it)."""
        # Weights.
        return np.asarray([c.weight for c in self.genome_data.connections], dtype=float)

    def set_genome(self, genome: np.ndarray) -> None:
        """Replace the connection weights (same count required)."""
        # Check.
        if len(genome) != len(self.genome_data.connections):
            raise ValueError("NEAT weight vector length does not match the genome's connection count")
        for c, w in zip(self.genome_data.connections, genome, strict=False):
            c.weight = float(w)

    def describe(self) -> str:
        """A one-line summary."""
        # Counts.
        g = self.genome_data
        return (
            f"NEAT: {VECTOR_SIZE} inputs, {g.hidden_count} hidden, {MENU_SIZE} outputs, {g.enabled_count} connections"
        )
