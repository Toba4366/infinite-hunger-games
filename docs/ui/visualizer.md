# `visualizer.py`

**Source:** [hunger_games/ui/visualizer.py](../../hunger_games/ui/visualizer.py)
**Depends on:** `dearpygui.dearpygui as dpg`, `numpy`; project modules [../brain/neural.md](../brain/neural.md) (`MENU_NAMES`), [../perception.md](../perception.md) (`VECTOR_NAMES`)
**Used by:** [app.md](app.md) (`Dashboard.visualizer`, drawn on the right panel's Network tab every frame from `Session.network_snapshot`, see [session.md](session.md))

## Purpose

`visualizer.py` draws a brain as a graph of connected nodes. For a neural brain each column is a layer: the 50 perception inputs on the left with their names, the hidden layers in the middle, and the 16 action outputs on the right with their names and probabilities. For a NEAT brain the same picture is built from the genome itself: inputs and the bias node on the left, hidden nodes in columns by their depth, outputs on the right, and every enabled connection as an edge. Node colour shows the activation (blue negative, dark grey zero, red positive). Edge colour shows the sign of the weight (warm positive, cool negative) and edge brightness shows its size. When a learner tribute is selected during a live game the picture is redrawn every frame from that tribute's latest perception, so you can watch the network change as the game plays.

The file knows nothing about games or tributes. It takes a plain dictionary, the snapshot built by `Session.network_snapshot`, and draws it. A snapshot with `"graph": True` is a NEAT genome; any other snapshot is a layered network. When there is no snapshot it draws the bare architecture with every node at zero.

## Concepts you need

- **Layers and weight matrices.** A network with sizes `[50, 16, 16]` has two weight matrices: one shaped `(50, 16)` between inputs and the hidden layer, one shaped `(16, 16)` between the hidden layer and the outputs. In this file `matrix[i, j]` is the weight from source node `i` in one column to target node `j` in the next.
- **Activations.** The layered snapshot carries the value of every node: the raw input vector, then each layer's output in turn. The last entry is the output layer's logits, before the softmax. `probabilities` is the softmax of those logits and `chosen` is the index the brain picked.
- **NEAT genomes.** A NEAT genome is not layered. It is a list of nodes (kinds `input`, `bias`, `hidden`, `output`, each with an id) and a list of connections (source id, target id, weight, enabled flag). A node's **depth** is 0 for inputs and the bias, and one more than its deepest enabled source for everything else, so hidden nodes can sit at any depth and the graph is drawn in columns by that depth. The genome's `activations(inputs)` gives one value per node id. See [hunger_games/brain/neat.py](../../hunger_games/brain/neat.py).
- **Drawlist and draw layer.** `dpg.drawlist` is a blank pixel area; `dpg.add_draw_layer()` inside it is a group of drawings that can be cleared in one call with `dpg.delete_item(layer, children_only=True)`. Everything here is drawn onto one layer and cleared before each redraw.
- **RGBA colours.** Dear PyGui wants `(r, g, b, a)` with each channel 0 to 255. The alpha channel is what makes weak edges faint.
- **`np.argsort(...)[::-1][:k]`.** Sort ascending, reverse, take the first `k`: the indices of the `k` largest values. That is how the strongest incoming edges of each node are picked in the layered drawing.
- **Saturation.** `activation_color` clamps the value to `-1..1`. Inputs are scaled to about that range and `tanh` outputs are inside it, but `relu` activations and output logits can be larger; anything beyond 1 in size is drawn as full red or full blue.

## Walkthrough

### `MAX_EDGES_PER_NODE`

```python
MAX_EDGES_PER_NODE = 6
```

At most this many edges are drawn into each node of a layered network: the six with the largest absolute weight. With 50 inputs and 16 hidden nodes there are 800 weights in the first matrix; drawing all of them would be a solid block. Six per target node keeps the picture readable while still showing which inputs each hidden node listens to most. The NEAT drawing does not use it: a NEAT genome has few connections, so all of them are drawn.

### `activation_color`

`def activation_color(value: float) -> tuple[int, int, int, int]`

Clamps the value to `-1..1`. Negative values blend from dark grey `(60, 70, 90)` toward blue `(60, 70, 255)`; positive values blend toward red `(255, 60, 60)`; zero is dark grey. Alpha is always 255.

### `class NetworkVisualizer`

"Draws a network snapshot (from Session.network_snapshot) into a drawlist."

#### `NetworkVisualizer.__init__`

`def __init__(self, width: int = 520, height: int = 720) -> None`

Remembers the pixel size. `tag = "network_canvas"`, `layer = None` until `build`, `last = None` (the last snapshot, so the picture persists while paused).

#### `NetworkVisualizer.build`

`def build(self, parent) -> None`

Creates the drawlist inside `parent` with one draw layer.

#### `NetworkVisualizer.resize`

`def resize(self, width: int, height: int) -> None`

Stores `max(200, width)` and `max(200, height)` and reconfigures the drawlist if it exists.

#### `NetworkVisualizer._render_graph`

`def _render_graph(self, data: dict) -> None`

Draws a NEAT snapshot. `data["nodes"]` is a list of `(id, kind, depth, value)` and `data["edges"]` a list of `(src, dst, weight)` for enabled connections.

1. **Columns.** `max_depth` is the largest depth of any node. Every `input` and `bias` node goes in column 0; every `output` node goes in column `max_depth + 1` regardless of its own depth; a `hidden` node goes in column `max(1, depth)`. The number of columns is the largest column plus one. Column x positions run from 150 pixels (room for input labels) to `width - 120` (room for output labels), evenly spaced.
2. **Rows.** Within each column the nodes are stacked in the order they appear in the node list, evenly spaced over `height - 40` pixels starting 20 pixels down. The node radius is `max(2.5, min(7.0, (height - 40) / busiest column / 2.5))`, so the fullest column decides the dot size (with 51 nodes in column 0 and a 700-pixel drawing that is about 5 pixels).
3. **Edges.** For every edge whose ends both have positions: alpha `30 + 200 * |weight| / top`, where `top` is the largest absolute weight in the genome; warm `(230, 120, 90)` for positive, cool `(90, 140, 230)` for negative; thickness 1.0. Edges are drawn before nodes so the dots sit on top.
4. **Nodes.** One filled circle per node, fill `activation_color(value)`, outline `(20, 20, 20)`, thickness 0.8. The bias node's value is 1, so it is a red dot at the bottom of the input column.
5. **Input labels.** Only when the number of `input` nodes equals `len(VECTOR_NAMES)` (50): the perception names at size 10, 145 pixels left of each input node. The bias node gets no label.
6. **Output labels.** For each `output` node paired with `data["menu"]`: `name  NN%` from `data["probabilities"]`, in yellow when its index equals `data["chosen"]`, else grey, 12 pixels to the right.
7. **Caption.** At the top left: `NEAT: {inputs} inputs, {hidden} hidden, {outputs} outputs, {edges} connections`, where `hidden` counts nodes of kind `hidden` and `edges` counts the enabled connections.

#### `NetworkVisualizer.render`

`def render(self, snapshot: dict | None, architecture: list[int] | None = None) -> None`

Clears the layer. A new snapshot replaces `self.last`; with `None` the previous snapshot is kept, so the picture persists while paused. Then:

- If the data has `"graph"` set, `_render_graph(data)` draws it and the method returns.
- If there is no data at all and no `architecture`, a one-line hint is drawn ("Select a neural tribute during a game to watch its network.").
- If there is no data but an architecture, a fake layered snapshot of zeros is built: `layer_sizes = architecture`, zero inputs and activations, `weights = None`, zero probabilities, `chosen = -1`.

The layered drawing then proceeds: column x positions from 150 to `width - 120`; node positions per layer evenly spaced over `height - 40`; radius scaled to the busiest layer; for each weight matrix, the `MAX_EDGES_PER_NODE` strongest incoming edges of each target node, coloured by sign and brightened by size relative to that matrix's largest weight; nodes coloured by `activation_color`; input labels when the first layer has exactly 50 nodes; output labels with probabilities (the chosen one yellow); and a caption over each column: `inputs (50)`, `hidden 1 (64)`, ..., `outputs (16)`.

| Snapshot key (layered) | Type | Meaning |
| --- | --- | --- |
| `layer_sizes` | `list[int]` | Nodes per column, inputs first |
| `inputs` | array of 50 | The perception vector |
| `activations` | list of arrays | One per layer after the input, the last being the output logits |
| `weights` | list of matrices or `None` | `(fan_in, fan_out)` per layer pair; `None` in the empty diagram |
| `probabilities` | array of 16 | The softmax of the logits |
| `chosen` | `int` | Index of the action taken, `-1` for none |
| `menu` | `list[str]` | The output names, `MENU_NAMES` |

| Snapshot key (graph) | Type | Meaning |
| --- | --- | --- |
| `graph` | `True` | Marks a NEAT snapshot |
| `nodes` | `list[(id, kind, depth, value)]` | Every node of the genome with its activation |
| `edges` | `list[(src, dst, weight)]` | Enabled connections only |
| `probabilities` | array of 16 | The action probabilities |
| `chosen` | `int` | Index of the action taken |
| `menu` | `list[str]` | The output names |

## How to use it / experiment

**Watch a network live.** Train on the Train tab, press Watch agent, click a starred tribute on the arena, and open the Network tab on the right. The caption above the drawing names the tribute and its chosen action; the picture updates every frame while the game plays. Pause and use Step to watch one decision at a time.

**Watch a NEAT network grow.** Run the `neat` method with the training feed set to `live`. Each time the feed starts a game, click a starred tribute: the caption reports how many hidden nodes and connections the champion has now, and the columns between inputs and outputs fill in as the genome gains depth.

**Read the picture.** A hidden node that is bright red or bright blue is firing hard. Follow its brightest incoming edges to the left to see which inputs drive it. On the right, the yellow label is the action taken; the percentages are the softmax the brain sampled from (with chaos 0 the brain always takes the largest).

**Draw a snapshot without a game.** The class needs a Dear PyGui context but not a game:

```python
import dearpygui.dearpygui as dpg
import numpy as np
from hunger_games.brain.neural import MENU_NAMES
from hunger_games.ui.visualizer import NetworkVisualizer

dpg.create_context()
viz = NetworkVisualizer(600, 700)
with dpg.window(tag="w"):
    viz.build("w")
snapshot = {
    "layer_sizes": [50, 16, 16],
    "inputs": np.random.uniform(-1, 1, 50),
    "activations": [np.random.uniform(-1, 1, 16), np.random.uniform(-2, 2, 16)],
    "weights": [np.random.normal(0, 0.3, (50, 16)), np.random.normal(0, 0.3, (16, 16))],
    "probabilities": np.full(16, 1 / 16),
    "chosen": 3,
    "menu": MENU_NAMES,
}
viz.render(snapshot)
```

**Draw a NEAT graph by hand.** The graph mode only needs node and edge tuples:

```python
nodes = [(i, "input", 0, 0.5) for i in range(50)] + [(50, "bias", 0, 1.0)]
nodes += [(51 + o, "output", 1, 0.0) for o in range(16)] + [(70, "hidden", 1, 0.8)]
edges = [(3, 70, 1.2), (70, 52, -0.7), (50, 51, 0.3)]
viz.render({"graph": True, "nodes": nodes, "edges": edges,
            "probabilities": np.full(16, 1 / 16), "chosen": 1, "menu": MENU_NAMES})
```

**Show more or fewer edges.** Change `MAX_EDGES_PER_NODE` for layered networks. To thin a busy NEAT graph, filter `edges` by `abs(weight)` before drawing in `_render_graph`.

**Reuse the colour scale elsewhere.** `activation_color` is a plain function; the Charts tab could use it for a heat cell or the inspector for a bar.

## Gotchas

- `self.last` is sticky. Once a snapshot has been drawn, `render(None, architecture)` keeps drawing that old snapshot, not the empty diagram. Select a voting tribute after a neural one and the picture stays on the neural tribute's last state while the caption above it switches to "Architecture: ...". The same holds for a NEAT graph: it stays until a new snapshot arrives. Start a new game and select a learner tribute to refresh it.
- The picture is built from the live game, not from the frame on screen. Scrub backwards during a game and the network still shows the tribute's latest decision. See `network_snapshot` in [session.md](session.md).
- Input labels only appear when the first layer, or the number of NEAT `input` nodes, is exactly 50. A network built for a different perception size draws unlabelled inputs.
- In the NEAT drawing, outputs are always placed in the last column even when some hidden node is deeper than any output's own depth, so an edge can point from right to left when a deep hidden node feeds a shallower one. NEAT genomes are acyclic, so this is only a drawing artefact.
- A hidden node with no enabled incoming connection has depth 0 and is drawn in column 1, next to nodes that do have sources.
- The NEAT drawing draws every enabled connection, so a genome that has grown to hundreds of connections becomes a dense picture; the layered drawing caps edges at 6 per node.
- Alpha is scaled per matrix in the layered drawing and per genome in the NEAT drawing. A faint edge in one matrix and a faint edge in another are not comparable.
- Output nodes are coloured by their logits, not their probabilities. Logits above 1 all look the same shade of red.
- With 50 input nodes and a 700-pixel drawing the input rows are about 13 pixels apart, and the labels are drawn at size 10. Make the window taller if they overlap; the drawing follows `Dashboard._layout`.
- The drawlist is created with tag `network_canvas`. Building two visualizers in one context would clash on that tag.
- Redrawing means deleting and recreating every line, circle and text every frame. For a network of a few hundred nodes this is fine; for thousands the Network tab will slow the whole dashboard down.
