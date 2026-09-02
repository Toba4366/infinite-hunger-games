# `visualizer.py`

**Source:** [hunger_games/ui/visualizer.py](../../hunger_games/ui/visualizer.py)
**Depends on:** `dearpygui.dearpygui as dpg`, `numpy`; project modules [../brain/neural.md](../brain/neural.md) (`MENU_NAMES`), [../perception.md](../perception.md) (`VECTOR_NAMES`)
**Used by:** [app.md](app.md) (`Dashboard.visualizer`, drawn on the right panel's Network tab every frame from `Session.network_snapshot`, see [session.md](session.md))

## Purpose

`visualizer.py` draws a neural brain as a graph of connected nodes. Each column is a layer: the 50 perception inputs on the left with their names, the hidden layers in the middle, and the 16 action outputs on the right with their names and probabilities. Node colour shows the activation (blue negative, dark grey zero, red positive). Edge colour shows the sign of the weight (warm positive, cool negative) and edge brightness shows its size. When a neural tribute is selected during a live game the picture is redrawn every frame from that tribute's latest perception, so you can watch the hidden layers change as the game plays.

The file knows nothing about games or tributes. It takes a plain dictionary, the snapshot built by `Session.network_snapshot`, and draws it. When there is no snapshot it draws the bare architecture with every node at zero.

## Concepts you need

- **Layers and weight matrices.** A network with sizes `[50, 16, 16]` has two weight matrices: one shaped `(50, 16)` between inputs and the hidden layer, one shaped `(16, 16)` between the hidden layer and the outputs. In this file `matrix[i, j]` is the weight from source node `i` in one column to target node `j` in the next.
- **Activations.** The snapshot carries the value of every node: the raw input vector, then each layer's output in turn. The last entry is the output layer's logits, before the softmax. `probabilities` is the softmax of those logits and `chosen` is the index the brain picked.
- **Drawlist and draw layer.** `dpg.drawlist` is a blank pixel area; `dpg.add_draw_layer()` inside it is a group of drawings that can be cleared in one call with `dpg.delete_item(layer, children_only=True)`. Everything here is drawn onto one layer and cleared before each redraw.
- **RGBA colours.** Dear PyGui wants `(r, g, b, a)` with each channel 0 to 255. The alpha channel is what makes weak edges faint.
- **`np.argsort(...)[::-1][:k]`.** Sort ascending, reverse, take the first `k`: the indices of the `k` largest values. That is how the strongest incoming edges of each node are picked.
- **Saturation.** `activation_color` clamps the value to `-1..1`. Inputs are scaled to about that range and `tanh` outputs are inside it, but `relu` activations and output logits can be larger; anything beyond 1 in size is drawn as full red or full blue.

## Walkthrough

### `MAX_EDGES_PER_NODE`

```python
MAX_EDGES_PER_NODE = 6
```

At most this many edges are drawn into each node: the six with the largest absolute weight. With 50 inputs and 16 hidden nodes there are 800 weights in the first matrix; drawing all of them would be a solid block. Six per target node keeps the picture readable while still showing which inputs each hidden node listens to most.

### `activation_color`

```python
def activation_color(value: float) -> tuple[int, int, int, int]
```

Clamps `value` to `-1..1`. Negative values return `(60, 70, 90 + 165 * |v|, 255)`, a blend from dark grey toward blue. Zero and positive values return `(90 + 165 * v, 60, 60, 255)`, a blend from dark grey toward red. At `v = 0` both formulas give a dark grey, so an idle node is dark.

```python
activation_color(0.0)    # (90, 60, 60, 255)
activation_color(1.0)    # (255, 60, 60, 255)
activation_color(-1.0)   # (60, 70, 255, 255)
```

### `class NetworkVisualizer`

"Draws a network snapshot (from Session.network_snapshot) into a drawlist."

#### `NetworkVisualizer.__init__`

```python
def __init__(self, width: int = 520, height: int = 720) -> None
```

Remembers the pixel size and sets `tag = "network_canvas"`, `layer = None` (filled in by `build`) and `last = None`, the most recent snapshot. `last` is kept so the picture stays on screen while playback is paused or between frames where the tribute has not decided yet.

#### `NetworkVisualizer.build`

```python
def build(self, parent) -> None
```

Creates the drawlist with the stored size and tag inside `parent`, and one draw layer inside it. The dashboard calls this once, inside the Network tab's `network_holder` child window.

#### `NetworkVisualizer.resize`

```python
def resize(self, width: int, height: int) -> None
```

Sets the size to at least 200 by 200 and, if the drawlist exists, reconfigures its `width` and `height`. `Dashboard._layout` calls this on every window resize with the right panel's width minus 30 and the panel height minus 120. The next `render` lays the columns out for the new size.

#### `NetworkVisualizer.render`

```python
def render(self, snapshot: dict | None, architecture: list[int] | None = None) -> None
```

Redraws the whole picture. Step by step:

1. Clears the layer.
2. If `snapshot` is not `None`, stores it in `self.last`. The picture is then drawn from `self.last`.
3. If there is nothing to draw and no `architecture`, draws the text "Select a neural tribute during a game to watch its network." and returns. If there is an `architecture` (a list of layer sizes such as `[50, 16, 16]`), builds a fake snapshot with zero inputs, zero activations, no weights, zero probabilities, `chosen = -1` and `MENU_NAMES` as the menu.
4. `values` is the list of per-layer node values: the inputs first, then each entry of `activations`.
5. Columns are spread between `x = 150` and `x = width - 120`, leaving room for input labels on the left and output labels on the right. Within a column, nodes are spaced evenly over `height - 40` pixels starting 20 pixels down.
6. The node radius is `(height - 40) / max(sizes) / 2.5`, clamped to `2.5..7.0`, so the busiest layer decides how small the circles are.
7. Edges, drawn first so nodes sit on top. For each weight matrix, the absolute values are scaled by the largest one in that matrix. For each target node, the `MAX_EDGES_PER_NODE` strongest sources are drawn as lines of thickness 1. Alpha is `30 + 200 * |w| / max`, so the strongest edge in a matrix is nearly opaque and a near-zero edge is faint. Positive weights are warm `(230, 120, 90)`, negative ones cool `(90, 140, 230)`. The fake snapshot has `weights = None`, so the empty diagram has no edges.
8. Nodes: one circle per value, dark outline `(20, 20, 20)` of thickness 0.8, filled with `activation_color(value)`.
9. Input labels, only when the first layer has exactly `len(VECTOR_NAMES)` nodes: each name at size 10 in light grey, 145 pixels left of its node.
10. Output labels: for each of the first `sizes[-1]` names in `data["menu"]`, the text `"{name} {probability:4.0f}%"` at size 11, 12 pixels right of the node. The chosen action's label is yellow `(255, 230, 0)`; the rest are light grey.
11. Layer captions along the top: `inputs (50)`, `hidden 1 (16)`, `hidden 2 (...)`, `outputs (16)`, at size 11 in grey.

The snapshot is built by `Session.network_snapshot` in [session.md](session.md):

| Key | Type | Meaning |
| --- | --- | --- |
| `layer_sizes` | `list[int]` | `[50, *hidden_layers, 16]` |
| `inputs` | array of 50 | The tribute's last perception vector |
| `activations` | list of arrays | One per layer after the input, the last being the output logits |
| `weights` | list of matrices or `None` | `(fan_in, fan_out)` per layer pair; `None` in the empty diagram |
| `probabilities` | array of 16 | The softmax of the logits |
| `chosen` | `int` | Index of the action taken, `-1` for none |
| `menu` | `list[str]` | The output names, `MENU_NAMES` |

## How to use it / experiment

**Watch a network live.** Train a neural champion on the Train tab, press Watch champion, click a tribute on the arena, and open the Network tab on the right. The caption above the drawing names the tribute and its chosen action; the picture updates every frame while the game plays. Pause and use Step to watch one decision at a time.

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

**Show more or fewer edges.** Change `MAX_EDGES_PER_NODE`. Ten is still readable for one hidden layer of 16; three makes the strongest connections stand out.

**Colour by weight size instead of sign.** Replace the two-colour choice in step 7 with a single colour and let alpha carry the size. Or keep the sign and add thickness `0.5 + 2 * magnitude / top` for a stronger effect.

**Reuse the colour scale elsewhere.** `activation_color` is a plain function; the Charts tab could use it for a heat cell or the inspector for a bar.

## Gotchas

- `self.last` is sticky. Once a snapshot has been drawn, `render(None, architecture)` keeps drawing that old snapshot, not the empty diagram. Select a voting tribute after a neural one and the picture stays on the neural tribute's last state while the caption above it switches to "Architecture: ...". Start a new game and select a neural tribute to refresh it.
- The picture is built from the live game, not from the frame on screen. Scrub backwards during a game and the network still shows the tribute's latest decision. See `network_snapshot` in [session.md](session.md).
- Input labels only appear when the first layer has exactly 50 nodes. A network built for a different perception size draws unlabelled inputs.
- Only 6 edges per target node are drawn. A hidden node can be driven by a sum of many small weights that never appear on screen.
- Alpha is scaled per matrix. A faint edge in the first matrix and a faint edge in the second are not comparable; each is relative to the largest weight in its own matrix.
- Output nodes are coloured by their logits, not their probabilities. Logits above 1 all look the same shade of red.
- With 50 input nodes and a 700-pixel drawing the input rows are about 13 pixels apart, and the labels are drawn at size 10. Make the window taller if they overlap; the drawing follows `Dashboard._layout`.
- The drawlist is created with tag `network_canvas`. Building two visualizers in one context would clash on that tag.
- Redrawing means deleting and recreating every line, circle and text every frame. For a network of a few hundred nodes this is fine; for thousands the Network tab will slow the whole dashboard down.
