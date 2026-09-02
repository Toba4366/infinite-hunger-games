"""ui/visualizer.py - the neural network drawn as a graph of connected nodes.

Each column is a layer: the 50 perception inputs on the left (named), the
hidden layers in the middle, and the 16 action outputs on the right (named,
with their probabilities). Node colour shows the activation: blue for
negative, dark for zero, red for positive. Edge brightness shows the weight
size. When a neural tribute is selected during a live game, the picture
updates every frame from that tribute's latest perception, so you can watch
the hidden layers change as the game plays.
"""

# Dear PyGui.
import dearpygui.dearpygui as dpg

# numpy.
import numpy as np

# Input and output names.
from hunger_games.brain.neural import MENU_NAMES
from hunger_games.perception import VECTOR_NAMES

# At most this many edges are drawn into each node (the strongest ones), to keep the picture readable.
MAX_EDGES_PER_NODE = 6


def activation_color(value: float) -> tuple[int, int, int, int]:
    """Blue for negative, dark grey for zero, red for positive, saturating at |value| = 1."""
    # Clamp.
    v = max(-1.0, min(1.0, float(value)))
    # Negative: blend toward blue.
    if v < 0:
        return (60, 70, int(90 + 165 * -v), 255)
    # Positive: blend toward red.
    return (int(90 + 165 * v), 60, 60, 255)


class NetworkVisualizer:
    """Draws a network snapshot (from Session.network_snapshot) into a drawlist."""

    def __init__(self, width: int = 520, height: int = 720) -> None:
        """Remember the pixel size."""
        # Width.
        self.width = width
        # Height.
        self.height = height
        # The drawlist tag.
        self.tag = "network_canvas"
        # The layer everything is drawn on.
        self.layer = None
        # The last snapshot, so the picture persists while paused.
        self.last: dict | None = None

    def build(self, parent) -> None:
        """Create the drawlist."""
        # The drawlist.
        with dpg.drawlist(width=self.width, height=self.height, tag=self.tag, parent=parent):
            # One layer.
            self.layer = dpg.add_draw_layer()

    def resize(self, width: int, height: int) -> None:
        """Change the pixel size on window resize."""
        # Sizes.
        self.width, self.height = max(200, width), max(200, height)
        # The drawlist.
        if dpg.does_item_exist(self.tag):
            dpg.configure_item(self.tag, width=self.width, height=self.height)

    def render(self, snapshot: dict | None, architecture: list[int] | None = None) -> None:
        """Draw a snapshot, or an empty architecture diagram if there is no live tribute."""
        # Clear.
        dpg.delete_item(self.layer, children_only=True)
        # Use the latest live picture if none is given.
        if snapshot is not None:
            self.last = snapshot
        # What to draw.
        data = self.last
        # Nothing at all: draw the bare architecture.
        if data is None:
            # Need sizes.
            if not architecture:
                dpg.draw_text(
                    (10, 10),
                    "Select a neural tribute during a game to watch its network.",
                    color=(200, 200, 200, 255),
                    size=14,
                    parent=self.layer,
                )
                return
            # Fake zeros.
            data = {
                "layer_sizes": architecture,
                "inputs": np.zeros(architecture[0]),
                "activations": [np.zeros(n) for n in architecture[1:]],
                "weights": None,
                "probabilities": np.zeros(architecture[-1]),
                "chosen": -1,
                "menu": MENU_NAMES,
            }
        # Layer sizes and values.
        sizes = data["layer_sizes"]
        # Values per layer: inputs first, then each activation.
        values = [np.asarray(data["inputs"])] + [np.asarray(a) for a in data["activations"]]
        # Column x positions, leaving room for labels on both sides.
        left, right = 150, self.width - 120
        # Columns.
        xs = [left + (right - left) * i / max(1, len(sizes) - 1) for i in range(len(sizes))]
        # Node positions per layer.
        positions = []
        # Each layer.
        for column, size in enumerate(sizes):
            # Vertical spacing.
            spacing = (self.height - 40) / max(1, size)
            # Positions.
            positions.append([(xs[column], 20 + spacing * (i + 0.5)) for i in range(size)])
        # Node radius scaled to the busiest layer.
        radius = max(2.5, min(7.0, (self.height - 40) / max(sizes) / 2.5))
        # Edges: for each layer pair, the strongest incoming weights of each node.
        weights = data.get("weights")
        # Draw edges first so nodes sit on top.
        if weights is not None:
            for column, matrix in enumerate(weights):
                # Absolute size for brightness.
                magnitude = np.abs(matrix)
                # Scale.
                top = magnitude.max() if magnitude.size and magnitude.max() > 0 else 1.0
                # Each target node.
                for j in range(matrix.shape[1]):
                    # Strongest sources.
                    sources = np.argsort(magnitude[:, j])[::-1][:MAX_EDGES_PER_NODE]
                    # Each edge.
                    for i in sources:
                        # Brightness.
                        alpha = int(30 + 200 * magnitude[i, j] / top)
                        # Positive weights warm, negative cool.
                        color = (230, 120, 90, alpha) if matrix[i, j] > 0 else (90, 140, 230, alpha)
                        dpg.draw_line(
                            positions[column][i],
                            positions[column + 1][j],
                            color=color,
                            thickness=1.0,
                            parent=self.layer,
                        )
        # Nodes.
        for column, layer_values in enumerate(values):
            # Each node.
            for i, value in enumerate(layer_values):
                dpg.draw_circle(
                    positions[column][i],
                    radius,
                    color=(20, 20, 20, 255),
                    fill=activation_color(value),
                    thickness=0.8,
                    parent=self.layer,
                )
        # Input labels.
        if sizes[0] == len(VECTOR_NAMES):
            for i, name in enumerate(VECTOR_NAMES):
                dpg.draw_text(
                    (positions[0][i][0] - 145, positions[0][i][1] - 6),
                    name,
                    color=(190, 190, 190, 255),
                    size=10,
                    parent=self.layer,
                )
        # Output labels with probabilities.
        probabilities = np.asarray(data["probabilities"])
        # Each output.
        for i, name in enumerate(data["menu"][: sizes[-1]]):
            # Highlight the chosen action.
            chosen = i == data.get("chosen", -1)
            # Text.
            dpg.draw_text(
                (positions[-1][i][0] + 12, positions[-1][i][1] - 6),
                f"{name} {probabilities[i] * 100:4.0f}%",
                color=(255, 230, 0, 255) if chosen else (200, 200, 200, 255),
                size=11,
                parent=self.layer,
            )
        # Layer captions.
        for column, size in enumerate(sizes):
            caption = "inputs" if column == 0 else ("outputs" if column == len(sizes) - 1 else f"hidden {column}")
            dpg.draw_text(
                (xs[column] - 20, 2), f"{caption} ({size})", color=(150, 150, 150, 255), size=11, parent=self.layer
            )
