# `neat.py`

**Source:** [hunger_games/brain/neat.py](../../hunger_games/brain/neat.py)
**Depends on:** `dataclasses` (standard library); `numpy`; [hunger_games/actions.py](../actions.md) (`Action`); [brain/base.py](base.md) (`Brain`); [brain/initializers.py](initializers.md) (`ACTIVATIONS`); [brain/neural.py](neural.md) (`MENU_SIZE`, `NeuralBrain.menu_to_action`, `softmax`); [hunger_games/perception.py](../perception.md) (`VECTOR_SIZE`, `Perception`)
**Used by:** [training/neat.py](../training/neat.md) (`InnovationTracker`, `NeatConfig`, `NeatGenome`, `NeatBrain`); [training/common.py](../training/common.md) (`build_learner` rebuilds a `NeatBrain` from a `LearnerSpec` of kind `"neat"`); [hunger_games/game.py](../game.md) (a roster entry with `brain_name == "neat"` and a dictionary genome becomes a `NeatBrain`); [ui/session.py](../ui/session.md) (draws NEAT champions as graphs, reads connection weights for the gene history); [ui/visualizer.py](../ui/visualizer.md) (`NeatGenome.activations` and `depths` lay out the graph); `tests/test_methods.py`

## Purpose

NEAT (NeuroEvolution of Augmenting Topologies, Stanley and Miikkulainen 2002) is the algorithm the Monopoly video uses. The neural brain in [neural.md](neural.md) has a fixed shape and only its weights change. A NEAT genome describes the shape too: a list of node genes and a list of connection genes. Networks start minimal, with every input wired straight to every output, and grow by mutation: a new connection between two nodes, or a new node spliced into an existing connection.

Every new connection gets an innovation number from a global counter. That number lets two genomes be lined up gene by gene, which is what makes crossover between differently shaped networks possible, and what lets the trainer measure how different two genomes are to keep species apart.

The network is kept feed-forward. A connection is only allowed from a node of lower depth to one of higher depth, so evaluation is a single pass in depth order with no cycles.

Evaluation is compiled. The first forward pass turns the gene lists into a small evaluation plan of numpy index and weight arrays, cached on the genome. Every later pass is a handful of dot products instead of a Python loop over hundreds of genes. Measured on a minimal genome, a forward pass takes about 30 microseconds; the old gene-by-gene loop took several milliseconds. The plan is dropped by `invalidate()` whenever the structure or the weights change.

This file holds the genome and a `NeatBrain` that wraps it. The evolution loop (species, fitness sharing, reproduction) is in [../training/neat.md](../training/neat.md).

## Concepts you need

**Genes.** A genome is two lists. A `NodeGene` is one neuron with an id, a kind and an activation. A `ConnectionGene` is one weighted link from a source node to a destination node, with an innovation number and an enabled flag.

**Innovation numbers.** A global counter, handed out by `InnovationTracker`. Two genomes that hold a connection gene with the same innovation number hold "the same" gene, however different the rest of their shapes are. Matching genes can be compared weight to weight; unmatched genes are called disjoint (inside the other genome's range of numbers) or excess (beyond it).

**Depth.** Inputs and the bias sit at depth 0. Every other node sits one deeper than its deepest enabled source. Depth gives the evaluation order and the feed-forward rule: new connections must go from lower depth to higher.

**Evaluation plan.** A compiled form of the genome for fast forward passes. For every non-input node in depth order it stores the indices of the node's enabled sources and their weights as numpy arrays, plus the node's kind and activation function. It also stores where the inputs, the bias and the outputs sit. Because the plan copies the weights, it must be rebuilt after any weight change, not only after structural ones.

**Disabled connections.** Splitting a connection disables it rather than deleting it. The gene stays in the genome so crossover can still line it up, and a later mutation can re-enable it. Disabled connections are left out of the plan.

**Compatibility distance.** A number that says how different two genomes are: excess and disjoint genes count, and so does the average weight difference of matching genes. Genomes closer than a threshold belong to the same species.

**The menu.** Like the neural brain, a NEAT brain outputs one score per item of the 16-item action menu (`MENU_SIZE`) and reads the 50-value perception vector (`VECTOR_SIZE`). `NeuralBrain.menu_to_action` turns the chosen index into an `Action`.

## Walkthrough

### `NodeGene`

```python
@dataclass
class NodeGene:
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `id` | `int` | required | Unique id |
| `kind` | `str` | required | `"input"`, `"bias"`, `"hidden"` or `"output"` |
| `activation` | `str` | `"tanh"` | Activation name for hidden and output nodes (a key of `ACTIVATIONS`) |

### `ConnectionGene`

```python
@dataclass
class ConnectionGene:
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `innovation` | `int` | required | The global innovation number |
| `src` | `int` | required | Source node id |
| `dst` | `int` | required | Destination node id |
| `weight` | `float` | required | Weight |
| `enabled` | `bool` | `True` | Disabled connections are kept in the genome (for crossover) but not evaluated |

### `NeatConfig`

```python
@dataclass
class NeatConfig:
```

The mutation and speciation knobs of NEAT.

| Field | Default | Meaning |
| --- | --- | --- |
| `add_node_rate` | `0.03` | Chance of adding a node by splitting a connection |
| `add_connection_rate` | `0.08` | Chance of adding a connection |
| `weight_mutate_rate` | `0.8` | Chance that each weight is mutated |
| `weight_perturb_rate` | `0.9` | Of mutated weights, the chance of a small nudge (else a fresh random value) |
| `weight_perturb_scale` | `0.1` | Size of the nudge (standard deviation) |
| `weight_range` | `1.0` | Range of a fresh random weight, `uniform(-range, range)` |
| `enable_rate` | `0.05` | Chance of re-enabling a disabled connection |
| `c_excess` | `1.0` | Distance weight of excess genes |
| `c_disjoint` | `1.0` | Distance weight of disjoint genes |
| `c_weights` | `0.4` | Distance weight of the mean weight difference |
| `compatibility_threshold` | `3.0` | Genomes closer than this belong to one species (the trainer's starting value) |
| `activation` | `"tanh"` | Activation of hidden nodes (and the name stored on output nodes) |

### `InnovationTracker`

```python
class InnovationTracker:
```

Hands out innovation numbers and node ids, remembering `(src, dst)` pairs within a generation.

#### `__init__(next_innovation=0, next_node=0)`

```python
def __init__(self, next_innovation: int = 0, next_node: int = 0) -> None:
```

Starts the two counters and an empty `seen` dictionary.

#### `innovation(src, dst)`

```python
def innovation(self, src: int, dst: int) -> int:
```

The innovation number for a connection. If `(src, dst)` was already handed out this generation, the same number is returned, so two genomes that invent the same connection in the same generation get matching genes. Otherwise the counter is used and advanced.

#### `node()`

```python
def node(self) -> int:
```

A fresh node id. Never reused.

#### `reset_generation()`

```python
def reset_generation(self) -> None:
```

Forgets this generation's `(src, dst)` pairs. The trainer calls it once per generation before breeding.

### `NeatGenome`

```python
@dataclass
class NeatGenome:
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `nodes` | `list[NodeGene]` | required | The nodes |
| `connections` | `list[ConnectionGene]` | required | The connections |
| `fitness` | `float` | `0.0` | Fitness after evaluation |
| `species` | `int` | `-1` | Which species it belongs to |
| `_depths` | `dict[int, int] | None` | `None` (`repr=False`, `compare=False`) | Cached node depths, computed on demand |
| `_plan` | `tuple | None` | `None` (`repr=False`, `compare=False`) | Cached evaluation plan, computed on demand by `_compile` |

Both caches are excluded from `repr` and from equality, so two genomes with the same genes compare equal whether or not one has been evaluated.

#### `minimal(inputs, outputs, rng, config, tracker)` (class method)

```python
@classmethod
def minimal(cls, inputs: int, outputs: int, rng: np.random.Generator, config: NeatConfig, tracker: InnovationTracker) -> "NeatGenome":
```

The starting genome: inputs and a bias fully connected to the outputs. Node ids are `0 .. inputs-1` for the inputs, `inputs` for the bias, and `inputs + 1 + o` for output `o`. The tracker's `next_node` is raised to `inputs + 1 + outputs` so new hidden nodes come after. Every `(input or bias, output)` pair gets a connection with weight `uniform(-0.5, 0.5)` and an innovation number from the tracker.

With `VECTOR_SIZE = 50` and `MENU_SIZE = 16`: inputs are nodes 0 to 49, the bias is 50, outputs are 51 to 66, and there are `51 * 16 = 816` connections. The first population shares one tracker, so every minimal genome gets the same innovation numbers 0 to 815 for the same pairs.

#### `copy()`

```python
def copy(self) -> "NeatGenome":
```

A deep copy: new `NodeGene` and `ConnectionGene` objects, the same `fitness` and `species`. Neither cache is copied; the copy compiles its own plan on its first forward pass.

#### `depths()`

```python
def depths(self) -> dict[int, int]:
```

Depth of every node: inputs and bias 0, others one more than their deepest enabled source. Starts every node at 0 and relaxes over the enabled connections (`depth[dst] = max(depth[dst], depth[src] + 1)`) until nothing changes, at most `len(nodes) + 1` rounds. The graph is acyclic by construction, so this terminates. The result is cached in `_depths`.

A hidden node whose incoming connections are all disabled has depth 0. That case matters in `_add_connection`.

#### `invalidate()`

```python
def invalidate(self) -> None:
```

Forgets the cached depths and the evaluation plan. Called after every structural change (a new node, a new connection, a re-enabled connection) and after every weight change, because the plan holds copies of the weights. Inside this file `mutate`, `_add_node` and `_add_connection` call it; `NeatBrain.set_genome` calls it after writing new weights.

#### `_compile()`

```python
def _compile(self) -> tuple:
```

Turns the genome into arrays so a forward pass is a few numpy dot products instead of Python loops. Returns the cached `_plan` if there is one. Otherwise:

1. `index` maps every node id to its position in `nodes`. `depth = self.depths()`.
2. `incoming` groups the enabled connections by destination node id. Disabled connections are skipped here, so they never take part in evaluation.
3. `steps` is built by walking the nodes sorted by depth and skipping inputs and the bias. Each step is a tuple `(slot, sources, weights, kind, activation)`: the node's position, an `int` array of its sources' positions, a `float` array of the matching weights, the node's kind, and the function `ACTIVATIONS[node.activation]`. A node with no enabled sources gets two empty arrays.
4. `input_slots`, `bias_slots` and `output_slots` are `int` arrays of the positions of the input, bias and output nodes, in node-list order.
5. The plan `(steps, input_slots, bias_slots, output_slots, len(nodes))` is cached in `_plan` and returned.

Python's sort is stable, so nodes at the same depth keep their list order. That order does not matter for the result, because a connection always goes from a lower depth to a higher one.

For a minimal genome the plan is 16 steps (one per output), each with 51 sources.

#### `_values(inputs)`

```python
def _values(self, inputs: np.ndarray) -> np.ndarray:
```

Every node's value for one input vector, using the compiled plan:

1. Take the plan from `_compile()` and make a zero array of one value per node.
2. Write the inputs into the input slots: `values[input_slots[:len(inputs)]] = inputs[:len(input_slots)]`. Extra inputs are ignored and missing ones stay 0. Write `1.0` into the bias slots.
3. For every step in depth order: `total = values[sources] @ weights`, or `0.0` when the node has no sources. A hidden node stores `activation(total)`; an output node stores the raw `total`.
4. Return the value array, indexed by node position.

Because each hidden node is finished before any deeper node is reached, one pass in depth order is enough.

#### `hidden_count` and `enabled_count` (properties)

```python
@property
def hidden_count(self) -> int:
@property
def enabled_count(self) -> int:
```

How many hidden nodes and how many enabled connections there are. The trainer reports both in `IterationStats.extra`.

#### `mutate(rng, config, tracker)`

```python
def mutate(self, rng: np.random.Generator, config: NeatConfig, tracker: InnovationTracker) -> None:
```

Applies NEAT's mutations in place, in this order:

1. **Weights.** For every connection, with probability `weight_mutate_rate`: with probability `weight_perturb_rate` add `normal(0, weight_perturb_scale)`, otherwise replace with `uniform(-weight_range, weight_range)`. With the defaults, 80 percent of weights move each generation, 72 percent by a small nudge and 8 percent by a reset. Then `invalidate()`, because the compiled plan holds copies of the weights.
2. **Re-enable.** Every disabled connection is re-enabled with probability `enable_rate`, with `invalidate()` after each one.
3. **Add a node** with probability `add_node_rate` (`_add_node`).
4. **Add a connection** with probability `add_connection_rate` (`_add_connection`).

#### `_add_node(rng, config, tracker)`

```python
def _add_node(self, rng: np.random.Generator, config: NeatConfig, tracker: InnovationTracker) -> None:
```

Splits an enabled connection `src -> dst` into `src -> new -> dst`. Picks a random enabled connection (does nothing if there is none), disables it, appends a hidden `NodeGene` with a fresh id and `config.activation`, and appends two connections: `src -> new` with weight `1.0` and `new -> dst` with the old weight. Both get innovation numbers from the tracker. Then invalidates the caches.

**Worked example.** Take a minimal genome and suppose the chosen connection is innovation 5: node 0 (an input) to node 56 (output 5), weight `0.3`.

| Step | Genes |
| --- | --- |
| Before | `ConnectionGene(5, 0, 56, 0.3, enabled=True)` |
| Disable | `ConnectionGene(5, 0, 56, 0.3, enabled=False)` |
| New node | `NodeGene(67, "hidden", "tanh")` |
| New connections | `ConnectionGene(816, 0, 67, 1.0)` and `ConnectionGene(817, 67, 56, 0.3)` |

Node 67 now has depth 1 and output 56 has depth 2; the other outputs stay at depth 1. Before the split, node 0 contributed `0.3 * x0` to output 56. After it, the contribution is `0.3 * tanh(1.0 * x0)`. For small `x0`, `tanh(x0)` is close to `x0`, so the network behaves almost the same as before. That is the point of the weights 1 and old: a structural change that starts nearly neutral, so selection does not throw it away before it can be tuned. In the plan, node 67 becomes a new step before output 56's step, and output 56's step loses node 0 from its sources and gains node 67.

#### `_add_connection(rng, config, tracker)`

```python
def _add_connection(self, rng: np.random.Generator, config: NeatConfig, tracker: InnovationTracker) -> None:
```

Connects two unconnected nodes from lower depth to higher, so the network stays feed-forward. It tries up to 20 random pairs `(a, b)` and gives up quietly if none fits. A pair is rejected when:

- `b` is an input or the bias (nothing may feed into them), or `a` is an output (nothing may leave them), or `a` and `b` are the same node;
- `depth[a] >= depth[b]`, unless `a` is an input or the bias and `b` is a hidden node at depth 0 (a hidden node with no enabled incoming connection, which may be wired up again);
- the pair `(a.id, b.id)` already has a connection gene, enabled or not.

The first pair that passes gets a connection with weight `uniform(-weight_range, weight_range)` and an innovation number from the tracker, and the caches are invalidated.

#### `crossover(other, rng)`

```python
def crossover(self, other: "NeatGenome", rng: np.random.Generator) -> "NeatGenome":
```

Combines with another genome. By convention `self` is the fitter parent; callers order them. Both parents' connections are indexed by innovation number. For every gene of `self`:

- if `other` has a gene with the same innovation number (a matching gene), the child's copy comes from either parent with probability 0.5;
- otherwise (disjoint or excess) the child's copy comes from `self`;
- if the gene matched and was disabled in either parent, the child's copy is disabled with probability 0.75.

Genes that only `other` has are dropped. The child's nodes are every non-hidden node from either parent, plus every hidden node referenced by a child connection. The child starts with `fitness = 0.0`, `species = -1` and no caches.

#### `distance(other, config)`

```python
def distance(self, other: "NeatGenome", config: NeatConfig) -> float:
```

NEAT's compatibility distance. With `mine` and `theirs` the innovation sets:

- `matching = mine & theirs`;
- `boundary = min(max(mine), max(theirs))`;
- `excess` is the number of unmatched innovation numbers above `boundary`, `disjoint` the rest of the unmatched;
- `weight_diff` is the mean absolute weight difference over matching genes (0 if none);
- `n` is the larger genome's connection count, or 1 when both are under 20 genes.

```
distance = c_excess * excess / n + c_disjoint * disjoint / n + c_weights * weight_diff
```

Every genome here has at least 816 connections, so `n` is always the size of the larger genome. Two minimal genomes with independent `uniform(-0.5, 0.5)` weights have a mean absolute difference of about 0.33, so their distance is about `0.4 * 0.33 = 0.13`. A single added node contributes `2 / 816` to the structural part. The trainer therefore starts everyone in one species and lowers its threshold each generation until the target number of species appears.

#### `forward(inputs)`

```python
def forward(self, inputs: np.ndarray) -> np.ndarray:
```

Evaluates the network on one input vector and returns the output node values. It takes `output_slots` from `_compile()` and returns `self._values(inputs)[output_slots]`: a float array with one raw sum per output node, in node-list order. No activation is applied to outputs, which is what the softmax wants.

The first call on a genome pays for `_compile`; every later call reuses the plan until `invalidate()`. Measured at about 30 microseconds per pass for a minimal genome. That is what made long NEAT runs feasible: a 48-genome generation plays 48 games of thousands of decisions each.

#### `activations(inputs)`

```python
def activations(self, inputs: np.ndarray) -> dict[int, float]:
```

The same computation through `_values`, mapped back to node ids: `{node.id: value}` for every node in list order. The dashboard's visualiser uses it to colour the graph.

#### `to_dict()`

```python
def to_dict(self) -> dict:
```

JSON-friendly form: `{"nodes": [[id, kind, activation], ...], "connections": [[innovation, src, dst, weight, enabled], ...], "fitness": fitness}`. The species id and the caches are not saved.

#### `from_dict(data)` (class method)

```python
@classmethod
def from_dict(cls, data: dict) -> "NeatGenome":
```

Rebuilds from `to_dict`, casting ids to `int`, weights to `float` and the enabled flag to `bool`. `fitness` defaults to `0.0` when missing. The champion files the NEAT trainer writes, the `LearnerSpec` sent to workers, and the roster entries the dashboard makes all carry this dictionary. A rebuilt genome compiles its plan on its first forward pass.

### `NeatBrain`

```python
class NeatBrain(Brain):
    name = "neat"
```

A brain driven by a NEAT genome: perception in, action menu scores out. `name` is the label in the results CSV.

#### `__init__(genome, chaos=0.0)`

```python
def __init__(self, genome: NeatGenome, chaos: float = 0.0) -> None:
```

Stores the genome as `genome_data` and starts `last_probabilities = None` and `last_index = 0`. The RL episode player reads `last_index` after every decision.

#### `decide_index(perception, rng)`

```python
def decide_index(self, perception: Perception, rng: np.random.Generator) -> int:
```

Scores the menu with `genome_data.forward(perception.to_vector())`, resizes the result to `MENU_SIZE` with `np.resize` (a genome always has 16 outputs, but be safe), then picks. With `chaos > 0` the probabilities are `softmax(logits, chaos)` and the index is sampled; otherwise the probabilities are a one-hot on the argmax and the index is the argmax. Both are remembered in `last_probabilities` and `last_index`.

#### `decide(perception, rng)`

```python
def decide(self, perception: Perception, rng: np.random.Generator) -> Action:
```

`NeuralBrain.menu_to_action(self.decide_index(perception, rng), perception)`. The same menu, the same targets filled in from the perception.

#### `genome()` and `set_genome(genome)`

```python
def genome(self) -> np.ndarray:
def set_genome(self, genome: np.ndarray) -> None:
```

`genome()` returns the connection weights as a flat vector, in connection order. The shape is not part of it. `set_genome` writes weights back, raises `ValueError` when the count differs from the number of connections, and then calls `genome_data.invalidate()` so the next forward pass recompiles with the new weights. The dashboard's gene history uses these vectors; the genetic trainer does not evolve NEAT brains.

#### `describe()`

```python
def describe(self) -> str:
```

`"NEAT: 50 inputs, H hidden, 16 outputs, C connections"` with `H = hidden_count` and `C = enabled_count`.

## How to use it / experiment

**Build, mutate and evaluate a genome by hand.**

```python
import numpy as np
from hunger_games.brain.neat import InnovationTracker, NeatConfig, NeatGenome
from hunger_games.brain.neural import MENU_SIZE
from hunger_games.perception import VECTOR_SIZE

rng = np.random.default_rng(0)
tracker = InnovationTracker()
config = NeatConfig(add_node_rate=1.0, add_connection_rate=1.0)
genome = NeatGenome.minimal(VECTOR_SIZE, MENU_SIZE, rng, config, tracker)
print(genome.enabled_count)            # 816
for _ in range(5):
    genome.mutate(rng, config, tracker)
print(genome.hidden_count, genome.enabled_count)
print(genome.forward(np.zeros(VECTOR_SIZE)).shape)   # (16,)
```

Setting both rates to 1 forces a structural mutation every call, which is how `tests/test_methods.py` checks the feed-forward rule: `depth[c.src] < depth[c.dst]` for every enabled connection.

**Time a forward pass.**

```python
import timeit
x = np.ones(VECTOR_SIZE)
genome.forward(x)                                   # compiles the plan once
print(timeit.timeit(lambda: genome.forward(x), number=10000) / 10000)
```

Expect tens of microseconds. Call `genome.invalidate()` first to include the compile cost in a single pass.

**Play a genome in a game.** Give a roster entry `brain_name="neat"` and `genome=genome.to_dict()`, or wrap it yourself:

```python
from hunger_games.brain.neat import NeatBrain
brain = NeatBrain(genome, chaos=0.0)
```

**Save and load.** `json.dumps(genome.to_dict())` and `NeatGenome.from_dict(json.loads(text))`. The NEAT trainer's `champion.json` has this dictionary under `"genome"` with `"brain_name": "neat"`.

**Try other activations.** `NeatConfig(activation="relu")` gives new hidden nodes relu. Existing nodes keep the activation stored on their gene.

**Measure distance.** `genome.distance(other, config)` for two genomes from the same tracker. Watch the structural part stay tiny while networks are large; lower `c_weights` if you want structure to matter more.

## Gotchas

- **`create_brain("neat")` raises `KeyError`.** `NeatBrain` is not in `BRAIN_REGISTRY`. `build_learner` in `training/common.py` and `Game`'s roster handling (when the genome is a dictionary) know how to build one; `create_brain` does not.
- **Edit a gene by hand and the plan goes stale.** `_compile` caches the plan, and the plan holds copies of the weights. Change a `weight`, flip `enabled`, or append genes yourself and you must call `invalidate()`, or `forward` keeps using the old network. `mutate`, `_add_node`, `_add_connection` and `NeatBrain.set_genome` do this for you.
- Output nodes carry an activation name but `_values` never applies it. Outputs are raw sums, which is what the softmax wants.
- Innovation numbers are reused for the same `(src, dst)` pair within a generation, but `tracker.node()` always gives a fresh id. Two genomes that split the same connection in one generation get matching innovation numbers for the two new connections but different hidden node ids, so a crossover between them can pick `src -> new_a` from one parent and `new_b -> dst` from the other.
- `_add_connection` gives up after 20 random tries. In a large, nearly fully connected genome most pairs are rejected, so the effective add-connection rate is lower than `add_connection_rate`.
- A connection that was disabled and never re-enabled still counts in `distance` and in `genome()`. `enabled_count` is the number the trainer reports; `len(connections)` is larger.
- `_values` matches inputs to input nodes by list order, not by id. Keep the input nodes first and in order; `minimal` and `crossover` do.
- The first forward pass after any change is slower than the rest, because it rebuilds the plan. In the trainer that happens once per genome per generation, since every child is mutated before it plays.
- `crossover` keeps only `self`'s disjoint and excess genes. Call it on the fitter parent, as the trainer does, or the child loses the better parent's structure.
- The genome grows but never shrinks. There is no delete mutation; disabled genes stay forever. Long runs give long gene lists. The plan skips disabled genes, so evaluation cost grows with enabled connections and nodes, not with the gene list.
- `fitness` on a loaded genome is whatever the file held, not a fresh evaluation.
