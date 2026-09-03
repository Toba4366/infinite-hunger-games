"""ui/app.py - the game makers' dashboard window.

Three panels fill the window and resize with it. Left: control tabs
(Setup, Map, Loot, Tributes, Brains, Play, Train, Research). Centre: the
arena with a transport bar (play, pause, step, run to end, rewind, speed,
scrub). Right: an inspector for the selected tribute, the neural network
visualiser, and live behaviour charts. Every control changes the
`Session`; every frame the panels are redrawn from it, so the arena always
matches the settings.
"""

# Paths for the font search.
# Timing the frame loop.
import time
from pathlib import Path

# Dear PyGui.
import dearpygui.dearpygui as dpg

# numpy for chart data.
import numpy as np

# Brain names and initializer notes.
from hunger_games.brain import BRAIN_REGISTRY
from hunger_games.brain.initializers import ACTIVATIONS, INITIALIZER_NOTES, INITIALIZERS
from hunger_games.brain.neural import MENU_NAMES, MENU_SIZE, NeuralBrain
from hunger_games.brain.voting import GENE_NAMES

# Settings and enums.
from hunger_games.config import ArenaShape, LayoutName, NeuralConfig

# District facts.
from hunger_games.districts import DISTRICT_INDUSTRIES, SEXES

# The perception vector size and names.
from hunger_games.perception import VECTOR_NAMES, VECTOR_SIZE

# Sweep settings.
from hunger_games.research.experiments import SweepConfig

# Telemetry labels.
from hunger_games.research.telemetry import NEED_BIN_LABELS

# Supply kinds and weapon names.
from hunger_games.resources import ResourceKind, weapon_name

# Terrain kinds.
from hunger_games.terrain import TerrainType

# Training settings.
from hunger_games.training import (
    CurriculumConfig,
    ImitationConfig,
    NeatTrainerConfig,
    PPOConfig,
    RLConfig,
    TrainingConfig,
)

# The canvas.
from hunger_games.ui.canvas import ArenaCanvas

# The painter's presets.
from hunger_games.ui.painter import MapPainter

# The state.
from hunger_games.ui.session import Session

# The network picture.
from hunger_games.ui.visualizer import NetworkVisualizer

# The mouse tools offered at the top of the control panel.
TOOLS = ("Select", "Paint terrain", "Place loot", "Move tribute")
# Playback speed presets: label -> ticks per second.
SPEEDS = {"Slow-mo 2/s": 2.0, "Normal 8/s": 8.0, "Fast 40/s": 40.0, "Max 400/s": 400.0}
# Config fields a researcher is likely to sweep.
SWEEPABLE = [
    "chaos",
    "max_days",
    "vision_radius",
    "landmark_radius",
    "thirst_days",
    "hunger_days",
    "sponsor_gift_chance",
    "gamemaker_enabled",
    "intervention_days",
    "quiet_days_before_intervention",
    "endgame_instinct",
    "cannon_and_sky",
    "start_thirst_min",
    "start_hunger_min",
    "start_health_min",
    "num_players",
    "terrain.water_threshold",
    "terrain.sand_size",
    "terrain.grass_size",
    "noise.scale",
    "noise.octaves",
]
# Font files to try, in order (the first that exists is used).
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Verdana.ttf",
    "/Library/Fonts/Arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
# Panel widths as fractions of the window.
LEFT_FRACTION, RIGHT_FRACTION = 0.27, 0.27


class Dashboard:
    """Builds the window and wires every widget to the session."""

    def __init__(self) -> None:
        """Create the state and the default tool settings."""
        # The state.
        self.session = Session()
        # The arena drawing.
        self.canvas = ArenaCanvas(self.session)
        # The network picture.
        self.visualizer = NetworkVisualizer()
        # The active mouse tool.
        self.tool = TOOLS[0]
        # Paint settings.
        self.brush_terrain = TerrainType.GRASS
        # Brush radius in cells.
        self.brush_radius = 2
        # Loot settings.
        self.loot_kind = ResourceKind.WEAPON
        # Stack size.
        self.loot_quantity = 1
        # Quality.
        self.loot_quality = 0.8
        # The tribute being dragged, if any.
        self.drag_id: int | None = None
        # Whether a brush stroke is in progress.
        self.painting = False
        # Start a new game automatically when the watched one ends.
        self.auto_next = False
        # Time of the last frame, for delta time.
        self._last_time = time.time()
        # How many training steps the plots have drawn.
        self._plotted_steps = -1
        # Frame counter for the slower chart refreshes.
        self._frame = 0
        # Training settings being edited.
        self.ga = TrainingConfig()
        # RL settings.
        self.rl = RLConfig()
        # Imitation settings.
        self.imitation = ImitationConfig()
        # NEAT settings.
        self.neat = NeatTrainerConfig()
        # PPO settings.
        self.ppo = PPOConfig()
        # Curriculum settings.
        self.curriculum = CurriculumConfig(enabled=False)
        # Which method the Train tab uses.
        self.method = "imitation"
        # How many events the monitor showed last time.
        self._events_shown = 0
        # A brush ring to draw when the mouse is not over the arena (used by the screenshot tool).
        self.brush_demo: tuple[int, int, int] | None = None

    # ================================================================ run

    def run(self) -> None:
        """Open the window and run until it is closed."""
        # Dear PyGui setup.
        dpg.create_context()
        # Font and theme.
        self._load_font()
        # Theme.
        self._apply_theme()
        # The window.
        dpg.create_viewport(
            title="Infinite Hunger Games - Game Makers' Dashboard",
            width=1500,
            height=920,
            min_width=1100,
            min_height=700,
        )
        # Build everything.
        self.build()
        # Mouse handlers.
        with dpg.handler_registry():
            # Held buttons (painting, dragging).
            dpg.add_mouse_down_handler(callback=self._on_mouse_down)
            # Releases end strokes and drags.
            dpg.add_mouse_release_handler(callback=self._on_mouse_release)
            # Clicks select and place.
            dpg.add_mouse_click_handler(callback=self._on_mouse_click)
        # Relayout on resize.
        dpg.set_viewport_resize_callback(lambda: self._layout())
        # Finish setup.
        dpg.setup_dearpygui()
        # Show.
        dpg.show_viewport()
        # First layout.
        self._layout()
        # The frame loop (manual so we can update the session every frame).
        while dpg.is_dearpygui_running():
            # Our per-frame work.
            self.on_frame()
            # Dear PyGui's.
            dpg.render_dearpygui_frame()
        # Clean up.
        dpg.destroy_context()

    def _load_font(self) -> None:
        """Use a readable system font if one can be found."""
        # Try each candidate.
        for candidate in FONT_CANDIDATES:
            # The first that exists wins.
            if Path(candidate).exists():
                with dpg.font_registry():
                    font = dpg.add_font(candidate, 15)
                dpg.bind_font(font)
                return

    def _apply_theme(self) -> None:
        """A dark theme with the Capitol's crimson and gold accents."""
        # The theme.
        with dpg.theme() as theme:
            # Applies to everything.
            with dpg.theme_component(dpg.mvAll):
                # Backgrounds.
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (22, 24, 30))
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (30, 33, 41))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (44, 48, 58))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (58, 63, 76))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (70, 76, 92))
                dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (30, 33, 41))
                # Accents.
                dpg.add_theme_color(dpg.mvThemeCol_Button, (150, 28, 48))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (196, 30, 58))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (120, 20, 40))
                dpg.add_theme_color(dpg.mvThemeCol_Header, (70, 52, 100))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (95, 70, 135))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (110, 80, 155))
                dpg.add_theme_color(dpg.mvThemeCol_Tab, (50, 44, 66))
                dpg.add_theme_color(dpg.mvThemeCol_TabHovered, (196, 30, 58))
                dpg.add_theme_color(dpg.mvThemeCol_TabActive, (150, 28, 48))
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (242, 214, 72))
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (255, 230, 100))
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (242, 214, 72))
                dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, (242, 214, 72))
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (150, 28, 48))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (230, 230, 235))
                dpg.add_theme_color(dpg.mvThemeCol_Border, (60, 64, 78))
                # Shapes and spacing.
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 8)
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 8)
                dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 6)
                dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 6)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 5)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 6)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 10, 10)
        # Use it everywhere.
        dpg.bind_theme(theme)

    @staticmethod
    def _tip(text: str) -> None:
        """Attach a tooltip to the most recently created widget."""
        # A tooltip window that appears on hover.
        with dpg.tooltip(dpg.last_item()):
            dpg.add_text(text, wrap=320)

    # ============================================================== layout

    def build(self) -> None:
        """Lay out the primary window with its three panels."""
        # The primary window fills the viewport.
        with dpg.window(tag="root", no_title_bar=True, no_resize=True, no_move=True, no_scrollbar=True):
            # Title bar.
            with dpg.group(horizontal=True):
                dpg.add_text("INFINITE HUNGER GAMES", color=(242, 214, 72))
                dpg.add_text("  game makers' dashboard", color=(160, 160, 170))
                dpg.add_text("", tag="status_text", color=(180, 220, 180))
            # The three panels.
            with dpg.group(horizontal=True):
                # Left: controls.
                with dpg.child_window(tag="left_panel", width=400, height=800, border=True):
                    # The mouse tool.
                    dpg.add_text("Mouse tool")
                    dpg.add_radio_button(
                        TOOLS, default_value=self.tool, horizontal=True, callback=lambda s, a: setattr(self, "tool", a)
                    )
                    self._tip(
                        "Select: click a tribute to inspect it. Paint terrain: drag on the arena. Place loot: left-click adds, right-click removes. Move tribute: drag a tribute to a new podium."
                    )
                    dpg.add_separator()
                    # Tabs (tagged so the tutorial can switch between them).
                    with dpg.tab_bar(tag="left_tabs"):
                        with dpg.tab(label="Tutorial", tag="tab_tutorial"):
                            self._build_tutorial()
                        with dpg.tab(label="Setup", tag="tab_setup"):
                            self._build_setup()
                        with dpg.tab(label="Map", tag="tab_map"):
                            self._build_map()
                        with dpg.tab(label="Loot", tag="tab_loot"):
                            self._build_loot()
                        with dpg.tab(label="Tributes", tag="tab_tributes"):
                            self._build_tributes()
                        with dpg.tab(label="Brains", tag="tab_brains"):
                            self._build_brains()
                        with dpg.tab(label="Play", tag="tab_play"):
                            self._build_play()
                        with dpg.tab(label="Train", tag="tab_train"):
                            self._build_train()
                        with dpg.tab(label="Research", tag="tab_research"):
                            self._build_research()
                # Centre: arena and transport.
                with dpg.child_window(tag="center_panel", width=700, height=800, border=True):
                    self.canvas.build("center_panel")
                    self._build_transport()
                # Right: inspector, network, charts.
                with dpg.child_window(tag="right_panel", width=380, height=800, border=True):
                    with dpg.tab_bar(tag="right_tabs"):
                        with dpg.tab(label="Inspector", tag="tab_inspector"):
                            self._build_inspector()
                        with dpg.tab(label="Network", tag="tab_network"):
                            self._build_network()
                        with dpg.tab(label="Charts", tag="tab_charts"):
                            self._build_charts()
        # Make it fill the viewport.
        dpg.set_primary_window("root", True)

    def _layout(self) -> None:
        """Size the three panels and the drawings from the window size."""
        # Window size.
        width = dpg.get_viewport_client_width()
        height = dpg.get_viewport_client_height()
        # Panel heights leave room for the title bar.
        panel_height = max(400, height - 60)
        # Widths.
        left = int(width * LEFT_FRACTION)
        right = int(width * RIGHT_FRACTION)
        center = max(300, width - left - right - 50)
        # Apply.
        dpg.configure_item("left_panel", width=left, height=panel_height)
        dpg.configure_item("center_panel", width=center, height=panel_height)
        dpg.configure_item("right_panel", width=right, height=panel_height)
        # The arena drawing fits the centre panel above the transport bar.
        self.canvas.resize(int(min(center - 24, panel_height - 190)))
        # The network drawing fits the right panel.
        self.visualizer.resize(right - 30, panel_height - 120)

    # ============================================================== frame

    def on_frame(self) -> None:
        """Advance playback and refresh everything that changes."""
        # Delta time.
        now = time.time()
        # Seconds since the last frame.
        seconds = min(0.25, now - self._last_time)
        # Remember.
        self._last_time = now
        # Count.
        self._frame += 1
        # Advance playback.
        self.session.update(seconds)
        # Auto-next: when a watched game ends while playing, start another.
        if (
            self.auto_next
            and self.session.game is not None
            and self.session.game.is_over
            and self.session.at_live_edge
            and self.session.playing is False
            and dpg.get_value("auto_next_box")
        ):
            self.session.new_game()
            self.session.playing = True
        # Brush preview: the ring follows the mouse, or sits where the screenshot tool put it.
        cell = self.canvas.mouse_cell() if self.tool == "Paint terrain" else None
        self.canvas.brush_preview = (cell[0], cell[1], self.brush_radius) if cell is not None else self.brush_demo
        # Draw.
        self.canvas.render()
        # Panels.
        self._refresh_transport()
        self._refresh_inspector()
        self._refresh_network()
        self._refresh_training()
        self._refresh_research()
        # Slower refreshes.
        if self._frame % 30 == 0:
            self._refresh_charts()
        # Status.
        dpg.set_value("status_text", "   " + self.session.status)

    # ============================================================ tutorial

    # The tutorial steps: (title, text, action name or None).
    TUTORIAL_STEPS = [
        (
            "Welcome",
            "Left: the control tabs. Centre: the arena and the transport bar. Right: the inspector, the neural "
            "network visualiser and live charts. Every control has a tooltip. Work down these steps; each 'Show me' "
            "button performs the step for you and opens the tab where it lives. The written version with pictures "
            "is docs/tutorial/README.md.",
            None,
        ),
        (
            "1. Build an arena",
            "The Setup tab chooses the shape, the loot layout and the size; the Map tab loads presets and paints. "
            "Show me loads the 'lake_island' preset.",
            "arena",
        ),
        (
            "2. Paint terrain",
            "Pick the 'Paint terrain' tool at the top, choose a terrain and a brush radius on the Map tab, then drag on "
            "the arena. A ring shows the brush. Tributes are moved off any void you paint.",
            "paint",
        ),
        (
            "3. Edit the tributes",
            "The Tributes tab lists the roster. Click a name (or a dot on the arena with the Select tool) to rename, "
            "change scores, brain, weapon, favour and starting bars. Podium presets or the 'Move tribute' tool place them.",
            "tributes",
        ),
        (
            "4. Place loot",
            "The Loot tab sets a kind, quantity and quality; with the 'Place loot' tool, left-click places and "
            "right-click removes. Weapons are red triangles, food white dots, medicine magenta crosses.",
            "loot",
        ),
        (
            "5. Play a game",
            "New game builds a game from your settings and records every tick. Play, Step, To end, Rewind, the speed "
            "slider and the frame slider are under the arena. Female tributes are circles, males squares, coloured by district.",
            "play",
        ),
        (
            "6. Inspect and watch a network think",
            "Click a tribute for its bars, weapon, kills and favour. If it has a neural brain, the Network tab draws "
            "the network as a node graph: hidden activations change every tick. Show me gives everyone a neural brain "
            "and starts a game.",
            "network",
        ),
        (
            "7. Train and watch training",
            "One network is trained; it plays the starred tributes against voting opponents. Start with 'imitation' "
            "so it copies the voting brain's instincts, then keep 'start from the current champion' ticked and pick "
            "'ppo', 'reinforce', 'genetic' or 'neat'. Turn on the curriculum to face 1, 3, 7, 11 then 23 opponents. "
            "The feed replays a real training game after every iteration, or lets the newest learner play live. "
            "Show me starts a short imitation run with the live feed.",
            "train",
        ),
        (
            "8. Research",
            "The Research tab sweeps any setting over a list of values and exports one PNG per behaviour chart of the "
            "games you watched. Run folders go to results/.",
            "research",
        ),
        (
            "9. Save and share",
            "Save config (Setup), scenario (Map), replay (Play), champion and run folder (Train). The written tutorial, "
            "the research guide and one page per source file are in docs/.",
            "files",
        ),
    ]

    def _build_tutorial(self) -> None:
        """A step-by-step walkthrough with a button that performs each step."""
        # Each step.
        for index, (title, text, action) in enumerate(self.TUTORIAL_STEPS):
            # Header.
            with dpg.collapsing_header(label=title, default_open=index < 2):
                # Text.
                dpg.add_text(text, wrap=360)
                # Button.
                if action is not None:
                    dpg.add_button(label="Show me", callback=lambda s, a, u=action: self._tutorial_action(u))

    def _tutorial_action(self, name: str) -> None:
        """Perform a tutorial step and open the tab it belongs to."""
        # Build an arena.
        if name == "arena":
            self.session.apply_preset("lake_island")
            self.session.reposition_off_void()
            dpg.set_value("left_tabs", "tab_map")
        # Painting.
        elif name == "paint":
            self.tool = "Paint terrain"
            dpg.set_value("left_tabs", "tab_map")
        # Tributes.
        elif name == "tributes":
            if self.session.tributes:
                self._select(self.session.tributes[0].player_id)
            dpg.set_value("left_tabs", "tab_tributes")
        # Loot.
        elif name == "loot":
            self.tool = "Place loot"
            dpg.set_value("left_tabs", "tab_loot")
        # Play.
        elif name == "play":
            self.tool = "Select"
            self.session.new_game()
            self._set_speed(8.0)
            self.session.playing = True
            dpg.set_value("left_tabs", "tab_play")
        # Network.
        elif name == "network":
            self.session.config.brain_name = "neural"
            dpg.set_value("cfg_brain", "neural")
            self._on_brain_all()
            self.session.new_game()
            self._set_speed(4.0)
            self.session.playing = True
            if self.session.tributes:
                self._select(self.session.tributes[0].player_id)
            dpg.set_value("right_tabs", "tab_network")
            dpg.set_value("left_tabs", "tab_brains")
        # Train.
        elif name == "train":
            # A short imitation run with the feed live; the Train tab's own settings are left alone.
            dpg.set_value("train_method", "imitation")
            self._on_method(None, "imitation")
            self.session.feed_mode = "live"
            dpg.set_value("feed_mode", "live")
            self._plotted_steps = -1
            self._events_shown = 0
            self.session.start_training(
                ImitationConfig(demonstration_games=2, epochs=4, validation_games=1, learners_per_game=6), "imitation"
            )
            dpg.set_value("left_tabs", "tab_train")
        # Research.
        elif name == "research":
            dpg.set_value("left_tabs", "tab_research")
        # Files.
        elif name == "files":
            dpg.set_value("left_tabs", "tab_play")

    # ============================================================== setup

    def _setter(self, name: str, convert=lambda v: v, react: str | None = None):
        """A callback that sets one config attribute and optionally reacts (regenerate, re-place)."""

        # The callback.
        def callback(sender, value):
            setattr(self.session.config, name, convert(value))
            if react is not None:
                self.session.apply_config_change(react)
                self._rebuild_roster_table()

        return callback

    def _build_setup(self) -> None:
        """Arena and rules settings."""
        # The config being edited.
        c = self.session.config
        # Arena.
        with dpg.collapsing_header(label="Arena", default_open=True):
            dpg.add_combo(
                [s.value for s in ArenaShape],
                label="shape",
                default_value=c.shape.value,
                callback=self._setter("shape", ArenaShape, "shape"),
                tag="cfg_shape",
            )
            self._tip(
                "open_field is the 74th games' forest; round carves the 75th games' circle. Changing it regenerates the map and moves tributes off the void."
            )
            dpg.add_combo(
                [l.value for l in LayoutName],
                label="loot layout",
                default_value=c.layout.value,
                callback=self._setter("layout", LayoutName, "layout"),
                tag="cfg_layout",
            )
            self._tip(
                "cornucopia: one pile in the middle. ring: cheap supplies at the edge, weapons in the centre (the video's redesign). Changing it re-places the podiums."
            )
            dpg.add_input_int(
                label="size (cells)",
                default_value=c.width,
                min_value=30,
                max_value=300,
                min_clamped=True,
                max_clamped=True,
                callback=self._on_size,
                tag="cfg_size",
                on_enter=True,
            )
            self._tip("Arena width and height. Press Enter to apply; the map is regenerated.")
            dpg.add_input_int(
                label="seed (-1 = random)",
                default_value=-1 if c.seed is None else c.seed,
                callback=self._setter("seed", lambda v: None if v < 0 else v),
                tag="cfg_seed",
            )
            self._tip("The same seed and settings replay the same game exactly.")
            dpg.add_slider_float(
                label="chaos",
                default_value=c.chaos,
                min_value=0.0,
                max_value=1.0,
                callback=self._setter("chaos"),
                tag="cfg_chaos",
            )
            self._tip(
                "0 = deterministic: no luck in hunting or fights and brains always pick their favourite action. 1 = very random."
            )
            dpg.add_input_int(
                label="max days",
                default_value=c.max_days,
                min_value=1,
                max_value=60,
                min_clamped=True,
                max_clamped=True,
                callback=self._setter("max_days"),
                tag="cfg_days",
            )
            self._tip("A strict cutoff: the games are a draw if more than one tribute is alive at the end.")
            dpg.add_input_int(
                label="ticks per day",
                default_value=c.ticks_per_day,
                min_value=4,
                max_value=96,
                min_clamped=True,
                max_clamped=True,
                callback=self._setter("ticks_per_day"),
                tag="cfg_tpd",
            )
        # Tributes.
        with dpg.collapsing_header(label="Tributes and starting bars", default_open=True):
            dpg.add_input_int(
                label="tributes",
                default_value=c.num_players,
                min_value=2,
                max_value=96,
                min_clamped=True,
                max_clamped=True,
                callback=self._setter("num_players", react="players"),
                tag="cfg_players",
                on_enter=True,
            )
            self._tip("Press Enter to apply; a new roster is rolled.")
            dpg.add_slider_float(
                label="min thirst",
                default_value=c.start_thirst_min,
                min_value=0.05,
                max_value=1.0,
                callback=self._setter("start_thirst_min"),
                tag="cfg_thirst",
            )
            dpg.add_slider_float(
                label="min hunger",
                default_value=c.start_hunger_min,
                min_value=0.05,
                max_value=1.0,
                callback=self._setter("start_hunger_min"),
                tag="cfg_hunger",
            )
            dpg.add_slider_float(
                label="min health",
                default_value=c.start_health_min,
                min_value=0.05,
                max_value=1.0,
                callback=self._setter("start_health_min"),
                tag="cfg_health",
            )
            with dpg.group(horizontal=True):
                dpg.add_button(label="Everyone starts full", callback=lambda: self._set_start_bars(1.0))
                dpg.add_button(label="Random above 0.5", callback=lambda: self._set_start_bars(0.5))
            self._tip(
                "Each tribute starts with every bar drawn between the minimum and full. 1.0 means everyone starts full; 0.5 means a random spread that cannot kill anyone at the start."
            )
            dpg.add_slider_int(
                label="vision radius",
                default_value=c.vision_radius,
                min_value=2,
                max_value=30,
                callback=self._setter("vision_radius"),
                tag="cfg_vision",
            )
            dpg.add_slider_int(
                label="landmark radius",
                default_value=c.landmark_radius,
                min_value=5,
                max_value=80,
                callback=self._setter("landmark_radius"),
                tag="cfg_landmark",
            )
            self._tip("How far away lakes and meadows can be spotted (bigger things show from further than people).")
            dpg.add_slider_float(
                label="days to die of thirst",
                default_value=c.thirst_days,
                min_value=1.0,
                max_value=10.0,
                callback=self._setter("thirst_days"),
                tag="cfg_thirst_days",
            )
            dpg.add_slider_float(
                label="days to starve",
                default_value=c.hunger_days,
                min_value=2.0,
                max_value=30.0,
                callback=self._setter("hunger_days"),
                tag="cfg_hunger_days",
            )
        # Knowledge and instincts.
        with dpg.collapsing_header(label="What tributes know", default_open=False):
            dpg.add_checkbox(
                label="cannon and nightly sky",
                default_value=c.cannon_and_sky,
                callback=self._setter("cannon_and_sky"),
                tag="cfg_cannon",
            )
            self._tip(
                "On: tributes know how many remain, how strong they are on average, the strongest left, and where they rank (they trained together). Off: they only know the count."
            )
            dpg.add_checkbox(
                label="endgame instinct",
                default_value=c.endgame_instinct,
                callback=self._setter("endgame_instinct"),
                tag="cfg_endgame",
            )
            self._tip(
                "On: bold tributes head for the centre once fewer than half remain. Off: games end by the circle or the day cutoff. Measured over 20 games: instinct alone 18/20 victors, circle alone 19/20, neither 0/20."
            )
        # Outside help and interference.
        with dpg.collapsing_header(label="Sponsors and game makers", default_open=False):
            dpg.add_checkbox(
                label="sponsor gifts",
                default_value=c.sponsors_enabled,
                callback=self._setter("sponsors_enabled"),
                tag="cfg_sponsors",
            )
            self._tip(
                "Parachutes of medicine, food or water for tributes in need, weighted by training score, career district and kills."
            )
            dpg.add_slider_float(
                label="gift chance / day",
                default_value=c.sponsor_gift_chance,
                min_value=0.0,
                max_value=1.0,
                callback=self._setter("sponsor_gift_chance"),
                tag="cfg_gift",
            )
            dpg.add_checkbox(
                label="game maker circle",
                default_value=c.gamemaker_enabled,
                callback=self._setter("gamemaker_enabled"),
                tag="cfg_gm",
            )
            self._tip(
                "The safe circle slowly closes after a quiet day. Tributes see it coming. It is the video's 'last resort'; turn it off to test a design that needs no intervention."
            )
            dpg.add_slider_float(
                label="quiet days before it",
                default_value=c.quiet_days_before_intervention,
                min_value=0.25,
                max_value=5.0,
                callback=self._setter("quiet_days_before_intervention"),
                tag="cfg_quiet",
            )
            dpg.add_slider_float(
                label="days to close",
                default_value=c.intervention_days,
                min_value=1.0,
                max_value=20.0,
                callback=self._setter("intervention_days"),
                tag="cfg_close",
            )
            self._tip(
                "How many days of shrinking it takes the circle to close from the edge to the centre. Bigger is gentler."
            )
            dpg.add_checkbox(
                label="podiums may stand in water",
                default_value=c.allow_water_podiums,
                callback=self._setter("allow_water_podiums"),
                tag="cfg_water_podiums",
            )
        # Actions.
        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Regenerate arena",
                callback=lambda: (self.session.generate_arena(), self.session.reposition_off_void()),
            )
            self._tip("A fresh Perlin map from the current settings.")
            dpg.add_button(label="New roster", callback=self._on_generate_roster)
            self._tip("Roll new tributes, scores and podiums.")
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Save config", callback=lambda: self._file_dialog(self._save_config, ".json", "config.json")
            )
            dpg.add_button(label="Load config", callback=lambda: self._file_dialog(self._load_config, ".json"))

    def _set_start_bars(self, value: float) -> None:
        """Set all three starting minimums at once."""
        # Config.
        self.session.config.start_thirst_min = self.session.config.start_hunger_min = (
            self.session.config.start_health_min
        ) = value
        # Widgets.
        for tag in ("cfg_thirst", "cfg_hunger", "cfg_health"):
            dpg.set_value(tag, value)

    def _on_size(self, sender, value) -> None:
        """Change the arena size (square) and regenerate the map."""
        # Config.
        self.session.config.width = self.session.config.height = int(value)
        # React.
        self.session.apply_config_change("size")

    def _on_generate_roster(self) -> None:
        """Roll a new roster and rebuild the table."""
        # Roll.
        self.session.generate_roster()
        # Table.
        self._rebuild_roster_table()

    # -------------------------------------------------------------- map

    def _build_map(self) -> None:
        """Painting tools and presets."""
        # Brush.
        with dpg.collapsing_header(label="Brush", default_open=True):
            dpg.add_radio_button(
                [t.name.lower() for t in TerrainType],
                default_value="grass",
                horizontal=True,
                callback=lambda s, a: setattr(self, "brush_terrain", TerrainType[a.upper()]),
            )
            self._tip("void = outside the arena (nobody can enter).")
            dpg.add_slider_int(
                label="brush radius",
                default_value=self.brush_radius,
                min_value=0,
                max_value=20,
                callback=lambda s, a: setattr(self, "brush_radius", a),
            )
            dpg.add_text(
                "Pick 'Paint terrain' above, then drag on the arena. The ring shows the brush.",
                wrap=360,
                color=(160, 160, 170),
            )
        # Presets and stamps.
        with dpg.collapsing_header(label="Presets and stamps", default_open=True):
            dpg.add_combo(list(MapPainter.PRESETS), label="preset", default_value="perlin", tag="map_preset")
            self._tip(
                "perlin: generated hills. flat_field / flat_round: plain meadows. quarter_quell: the 75th games' island in a sea. lake_island: a lake with an island."
            )
            dpg.add_button(
                label="Load preset",
                callback=lambda: (
                    self.session.apply_preset(dpg.get_value("map_preset")),
                    self.session.reposition_off_void(),
                ),
            )
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Carve round",
                    callback=lambda: (
                        self.session.painter.carve_round(),
                        self.session.painter.finish(),
                        self.session.reposition_off_void(),
                    ),
                )
                dpg.add_button(
                    label="Fill grass",
                    callback=lambda: (self.session.painter.fill(TerrainType.GRASS), self.session.painter.finish()),
                )
                dpg.add_button(
                    label="Fill water",
                    callback=lambda: (self.session.painter.fill(TerrainType.WATER), self.session.painter.finish()),
                )
            dpg.add_slider_int(label="stamp radius", default_value=20, min_value=2, max_value=100, tag="stamp_radius")
            with dpg.group(horizontal=True):
                dpg.add_button(label="Circle at centre", callback=self._stamp_circle)
                dpg.add_button(label="Square at centre", callback=self._stamp_square)
            self._tip("Stamps use the brush terrain.")
            dpg.add_checkbox(
                label="show tribute labels",
                default_value=True,
                callback=lambda s, a: setattr(self.canvas, "show_labels", a),
            )
            dpg.add_text("", tag="map_coverage", wrap=360, color=(160, 160, 170))
        # Files.
        with dpg.collapsing_header(label="Scenario files", default_open=True):
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Save scenario",
                    callback=lambda: self._file_dialog(self._save_scenario, ".json", "scenario.json"),
                )
                dpg.add_button(label="Load scenario", callback=lambda: self._file_dialog(self._load_scenario, ".json"))
            dpg.add_text(
                "A scenario file holds the map, the hand-placed loot and the roster.", wrap=360, color=(160, 160, 170)
            )

    def _stamp_circle(self) -> None:
        """Stamp a circle of the brush terrain at the centre."""
        # Painter.
        p = self.session.painter
        # Stamp.
        p.stamp_circle(p.width // 2, p.height // 2, dpg.get_value("stamp_radius"), self.brush_terrain)
        # Heights and podiums.
        p.finish()
        self.session.reposition_off_void()

    def _stamp_square(self) -> None:
        """Stamp a square of the brush terrain at the centre."""
        # Painter.
        p = self.session.painter
        # Half size.
        r = dpg.get_value("stamp_radius")
        # Stamp.
        p.stamp_rectangle(p.width // 2 - r, p.height // 2 - r, p.width // 2 + r, p.height // 2 + r, self.brush_terrain)
        # Heights and podiums.
        p.finish()
        self.session.reposition_off_void()

    # ------------------------------------------------------------- loot

    def _build_loot(self) -> None:
        """Hand-placed supplies."""
        # Kind.
        dpg.add_radio_button(
            ["food", "weapon", "medicine"],
            default_value="weapon",
            horizontal=True,
            callback=lambda s, a: setattr(self, "loot_kind", ResourceKind[a.upper()]),
        )
        # Quantity.
        dpg.add_slider_int(
            label="quantity",
            default_value=self.loot_quantity,
            min_value=1,
            max_value=20,
            callback=lambda s, a: setattr(self, "loot_quantity", a),
        )
        # Quality.
        dpg.add_slider_float(
            label="quality",
            default_value=self.loot_quality,
            min_value=0.0,
            max_value=1.0,
            callback=lambda s, a: setattr(self, "loot_quality", a),
        )
        self._tip(
            "Weapon quality decides reach and damage: fists 0, rock 0.2, knife 0.4, spear 0.6, sword 0.8, bow 0.9."
        )
        # Preview.
        dpg.add_text("", tag="loot_weapon_name", color=(160, 160, 170))
        # Hint.
        dpg.add_text(
            "Pick 'Place loot' above. Left-click places, right-click removes.", wrap=360, color=(160, 160, 170)
        )
        # Layout loot.
        dpg.add_checkbox(
            label="also scatter the layout's loot",
            default_value=True,
            callback=lambda s, a: setattr(self.session.scenario, "use_layout_loot", a),
        )
        self._tip("Off: only your hand-placed supplies exist.")
        # Clear.
        dpg.add_button(label="Clear hand-placed loot", callback=self.session.clear_loot)
        # Count.
        dpg.add_text("", tag="loot_count", color=(160, 160, 170))

    # --------------------------------------------------------- tributes

    def _build_tributes(self) -> None:
        """The roster table, podium presets and the editor for the selected tribute."""
        # Podium presets.
        with dpg.group(horizontal=True):
            dpg.add_combo(list(Session.PODIUM_PRESETS), default_value="edge ring", tag="podium_preset", width=160)
            dpg.add_button(
                label="Arrange podiums", callback=lambda: self.session.arrange_podiums(dpg.get_value("podium_preset"))
            )
        self._tip(
            "edge ring: along the outer edge (the video's redesign). around cornucopia: a tight circle round the middle. random. two sides: two opposing camps. Strong tributes are spread apart. Or drag tributes with the Move tool."
        )
        # The table.
        with dpg.table(
            header_row=True,
            tag="roster_table",
            height=200,
            scrollY=True,
            borders_innerH=True,
            policy=dpg.mvTable_SizingStretchProp,
        ):
            for label in ("name", "district", "sex", "score", "brain"):
                dpg.add_table_column(label=label)
        # Fill it.
        self._rebuild_roster_table()
        # Add / remove.
        with dpg.group(horizontal=True):
            dpg.add_button(label="Add tribute", callback=self._on_add_tribute)
            dpg.add_button(label="Remove selected", callback=self._on_remove_tribute)
        # Editor.
        with dpg.collapsing_header(label="Selected tribute", default_open=True, tag="editor_header"):
            # Helper: a callback that sets one field of the selected spec.
            def setter(name, convert=lambda v: v):
                def cb(s, a):
                    spec = (
                        self.session.tribute(self.session.selected_id) if self.session.selected_id is not None else None
                    )
                    if spec is not None:
                        setattr(spec, name, convert(a))
                        self._rebuild_roster_table()

                return cb

            dpg.add_input_text(label="name", tag="ed_name", callback=setter("name"))
            dpg.add_combo(
                [f"{d} {DISTRICT_INDUSTRIES[d]}" for d in range(1, 13)],
                label="district",
                tag="ed_district",
                callback=setter("district", lambda v: int(v.split()[0])),
            )
            dpg.add_combo(list(SEXES), label="sex", tag="ed_sex", callback=setter("sex"))
            dpg.add_slider_int(
                label="training score", min_value=1, max_value=12, tag="ed_score", callback=setter("training_score")
            )
            self._tip("1 to 12, as the game makers rate tributes. Raises sponsor favour and podium spacing.")
            dpg.add_slider_float(
                label="survival score",
                min_value=0.05,
                max_value=0.95,
                tag="ed_survival",
                callback=setter("survival_score"),
            )
            self._tip("Hunting aptitude against terrain difficulty (grass 0.2, water 0.6).")
            dpg.add_combo(list(BRAIN_REGISTRY), label="brain", tag="ed_brain", callback=setter("brain_name"))
            dpg.add_slider_float(
                label="granted weapon", min_value=0.0, max_value=1.0, tag="ed_weapon", callback=setter("weapon_quality")
            )
            dpg.add_slider_int(label="granted food", min_value=0, max_value=20, tag="ed_food", callback=setter("food"))
            dpg.add_slider_int(
                label="granted medkits", min_value=0, max_value=5, tag="ed_medicine", callback=setter("medicine")
            )
            dpg.add_slider_float(
                label="sponsor favour bonus",
                min_value=0.0,
                max_value=1.0,
                tag="ed_favor",
                callback=setter("favor_bonus"),
            )
            dpg.add_slider_float(
                label="start thirst (0 = config)",
                min_value=0.0,
                max_value=1.0,
                tag="ed_thirst",
                callback=setter("start_thirst", lambda v: None if v <= 0 else v),
            )
            dpg.add_slider_float(
                label="start hunger (0 = config)",
                min_value=0.0,
                max_value=1.0,
                tag="ed_hunger",
                callback=setter("start_hunger", lambda v: None if v <= 0 else v),
            )
            dpg.add_slider_float(
                label="start health (0 = config)",
                min_value=0.0,
                max_value=1.0,
                tag="ed_health",
                callback=setter("start_health", lambda v: None if v <= 0 else v),
            )
            dpg.add_button(label="Forget trained genome", callback=setter("genome", lambda v: None))

    def _rebuild_roster_table(self) -> None:
        """Refill the roster table from the session."""
        # Rows live in slot 1; columns in slot 0.
        if not dpg.does_item_exist("roster_table"):
            return
        dpg.delete_item("roster_table", children_only=True, slot=1)
        # One row per tribute.
        for spec in self.session.tributes:
            with dpg.table_row(parent="roster_table"):
                dpg.add_selectable(
                    label=spec.name,
                    span_columns=True,
                    callback=self._on_select_row,
                    user_data=spec.player_id,
                    default_value=spec.player_id == self.session.selected_id,
                )
                dpg.add_text(str(spec.district))
                dpg.add_text(spec.sex)
                dpg.add_text(str(spec.training_score))
                dpg.add_text(spec.brain_name + (" *" if spec.genome else ""))

    def _on_select_row(self, sender, value, player_id) -> None:
        """A roster row was clicked."""
        # Select.
        self._select(player_id)

    def _select(self, player_id: int | None) -> None:
        """Select a tribute and load them into the editor."""
        # Session.
        self.session.selected_id = player_id
        # Table highlight.
        self._rebuild_roster_table()
        # The spec.
        spec = self.session.tribute(player_id) if player_id is not None else None
        # Header.
        dpg.configure_item("editor_header", label="Selected tribute" + (f": {spec.name}" if spec else ": none"))
        # Nothing selected.
        if spec is None:
            return
        # Fields.
        dpg.set_value("ed_name", spec.name)
        dpg.set_value("ed_district", f"{spec.district} {DISTRICT_INDUSTRIES[spec.district]}")
        dpg.set_value("ed_sex", spec.sex)
        dpg.set_value("ed_score", spec.training_score)
        dpg.set_value("ed_survival", spec.survival_score)
        dpg.set_value("ed_brain", spec.brain_name)
        dpg.set_value("ed_weapon", spec.weapon_quality)
        dpg.set_value("ed_food", spec.food)
        dpg.set_value("ed_medicine", spec.medicine)
        dpg.set_value("ed_favor", spec.favor_bonus)
        dpg.set_value("ed_thirst", spec.start_thirst or 0.0)
        dpg.set_value("ed_hunger", spec.start_hunger or 0.0)
        dpg.set_value("ed_health", spec.start_health or 0.0)

    def _on_add_tribute(self) -> None:
        """Add a tribute and select them."""
        # Add.
        spec = self.session.add_tribute()
        # Select.
        self._select(spec.player_id)

    def _on_remove_tribute(self) -> None:
        """Remove the selected tribute."""
        # Need one.
        if self.session.selected_id is not None:
            self.session.remove_tribute(self.session.selected_id)
            self._select(None)

    # ------------------------------------------------------------ brains

    def _build_brains(self) -> None:
        """Default brain and neural-network architecture."""
        # Default brain.
        dpg.add_combo(
            list(BRAIN_REGISTRY),
            label="default brain",
            default_value=self.session.config.brain_name,
            callback=lambda s, a: setattr(self.session.config, "brain_name", a),
            tag="cfg_brain",
        )
        self._tip(
            "voting: the video's instinct-voting brain. random: a baseline. neural: the network below (untrained until you train it)."
        )
        dpg.add_button(label="Give this brain to every tribute", callback=self._on_brain_all)
        dpg.add_separator()
        # Network.
        dpg.add_text("Neural network", color=(242, 214, 72))
        n = self.session.config.neural
        # Inputs and outputs are fixed; the hidden layers are yours to choose.
        dpg.add_text(
            f"{VECTOR_SIZE} inputs (the perception) -> hidden layers -> {MENU_SIZE} outputs (the action menu)",
            wrap=360,
            color=(160, 160, 170),
        )
        # How many hidden layers.
        dpg.add_input_int(
            label="number of hidden layers",
            default_value=len(n.hidden_layers),
            min_value=1,
            max_value=6,
            min_clamped=True,
            max_clamped=True,
            callback=self._on_layer_count,
            tag="nn_layer_count",
        )
        self._tip(
            "Each hidden layer is a row of neurons between the inputs and the outputs. One layer is plenty to start; more layers can learn more complex rules but take longer to train."
        )
        # One width field per hidden layer, rebuilt when the count changes.
        with dpg.group(tag="nn_widths_group"):
            pass
        self._rebuild_width_fields(list(n.hidden_layers))
        dpg.add_combo(list(ACTIVATIONS), label="activation", default_value=n.activation, tag="nn_activation")
        self._tip("tanh pairs with Xavier, relu with He, selu with LeCun.")
        dpg.add_combo(
            list(INITIALIZERS),
            label="initializer",
            default_value=n.initializer,
            tag="nn_init",
            callback=lambda s, a: dpg.set_value("nn_init_note", INITIALIZER_NOTES[a]),
        )
        dpg.add_text(INITIALIZER_NOTES[n.initializer], tag="nn_init_note", wrap=360, color=(160, 160, 170))
        dpg.add_input_float(label="init scale", default_value=n.init_scale, tag="nn_scale", format="%.3f")
        self._tip("Only used by the constant, uniform and normal initializers.")
        dpg.add_slider_float(
            label="sparsity", default_value=n.sparsity, min_value=0.01, max_value=1.0, tag="nn_sparsity"
        )
        dpg.add_button(label="Apply network settings", callback=self._on_apply_neural)
        dpg.add_text("", tag="nn_summary", wrap=360, color=(160, 160, 170))
        # Inputs and outputs.
        with dpg.collapsing_header(label=f"Inputs ({VECTOR_SIZE}) and outputs ({MENU_SIZE})", default_open=False):
            dpg.add_text("Inputs, in order (each scaled to about -1..1):", color=(242, 214, 72))
            dpg.add_text(", ".join(f"{i}: {name}" for i, name in enumerate(VECTOR_NAMES)), wrap=360)
            dpg.add_text(
                "Outputs, in order (one score each; the highest, or a softmax sample, is taken):", color=(242, 214, 72)
            )
            dpg.add_text(", ".join(f"{i}: {name}" for i, name in enumerate(MENU_NAMES)), wrap=360)
        # Show the current summary.
        self._on_apply_neural()

    def _on_brain_all(self) -> None:
        """Give the default brain to every tribute (dropping any genomes)."""
        # Each.
        for spec in self.session.tributes:
            spec.brain_name = self.session.config.brain_name
            spec.genome = None
        # Table.
        self._rebuild_roster_table()

    def _rebuild_width_fields(self, widths: list[int]) -> None:
        """Show one 'nodes in hidden layer N' field per hidden layer."""
        # Clear the old fields.
        dpg.delete_item("nn_widths_group", children_only=True)
        # One per layer.
        for index, width in enumerate(widths):
            dpg.add_input_int(
                label=f"nodes in hidden layer {index + 1}",
                default_value=int(width),
                min_value=1,
                max_value=512,
                min_clamped=True,
                max_clamped=True,
                tag=f"nn_width_{index}",
                parent="nn_widths_group",
            )

    def _on_layer_count(self, sender, count) -> None:
        """Change how many hidden layers there are, keeping the widths already typed."""
        # The widths typed so far.
        current = self._read_widths()
        # Grow with 16-node layers or shrink.
        widths = (current + [16] * int(count))[: int(count)]
        # Rebuild.
        self._rebuild_width_fields(widths)

    def _read_widths(self) -> list[int]:
        """The widths currently typed in the per-layer fields."""
        # Collect until a field is missing.
        widths = []
        index = 0
        while dpg.does_item_exist(f"nn_width_{index}"):
            widths.append(int(dpg.get_value(f"nn_width_{index}")))
            index += 1
        # Done.
        return widths

    def _on_apply_neural(self) -> None:
        """Read the neural widgets into the config and show a summary."""
        # The layer widths.
        layers = tuple(self._read_widths())
        # Build the config.
        self.session.config.neural = NeuralConfig(
            hidden_layers=layers or (16,),
            activation=dpg.get_value("nn_activation"),
            initializer=dpg.get_value("nn_init"),
            init_scale=float(dpg.get_value("nn_scale")),
            sparsity=float(dpg.get_value("nn_sparsity")),
        )
        # A sample brain describes itself.
        dpg.set_value(
            "nn_summary",
            "Network: " + NeuralBrain(config=self.session.config.neural, rng=np.random.default_rng(0)).describe(),
        )

    # -------------------------------------------------------------- play

    def _build_play(self) -> None:
        """Playback and files."""
        # Explain.
        dpg.add_text(
            "Use the bar under the arena: New game, Play/Pause, Step, To end, Rewind, speed and scrub. Click a tribute to inspect it.",
            wrap=360,
            color=(160, 160, 170),
        )
        # Speed presets.
        with dpg.group(horizontal=True):
            for label, speed in list(SPEEDS.items())[:2]:
                dpg.add_button(label=label, callback=lambda s, a, u=speed: self._set_speed(u))
        with dpg.group(horizontal=True):
            for label, speed in list(SPEEDS.items())[2:]:
                dpg.add_button(label=label, callback=lambda s, a, u=speed: self._set_speed(u))
        # Auto next.
        dpg.add_checkbox(
            label="start a new game when this one ends (back to back)",
            default_value=False,
            tag="auto_next_box",
            callback=lambda s, a: setattr(self, "auto_next", a),
        )
        self._tip("Every finished game's behaviour is kept for the Charts tab and the Research exports.")
        dpg.add_separator()
        # Files.
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Save replay", callback=lambda: self._file_dialog(self._save_replay, ".replay", "game.replay")
            )
            dpg.add_button(label="Load replay", callback=lambda: self._file_dialog(self._load_replay, ".replay"))
        self._tip(
            "A replay holds every tick of a game; scrub it and inspect tributes. Only open replays you made yourself."
        )
        dpg.add_slider_int(label="GIF ticks per frame", default_value=2, min_value=1, max_value=12, tag="gif_step")
        dpg.add_button(
            label="Export GIF of this game", callback=lambda: self._file_dialog(self._export_gif, ".gif", "game.gif")
        )
        self._tip("Finishes the game first, then writes the file. Can take a minute.")

    def _set_speed(self, ticks_per_second: float) -> None:
        """Set the playback speed."""
        # Session.
        self.session.ticks_per_second = ticks_per_second
        # Widget.
        dpg.set_value("speed_slider", ticks_per_second)

    # ------------------------------------------------------------- train

    # The methods offered, in the order a learner should try them.
    METHODS = ("imitation", "genetic", "neat", "reinforce", "ppo")
    # One-line explanations shown under the method combo.
    METHOD_HELP = {
        "imitation": "Copies the voting brain's decisions (supervised). Start here: it gives the network instincts.",
        "genetic": "Evolves the weights of a population of networks; each plays as the learner against voting opponents.",
        "neat": "Evolves weights and the shape of the network, in species (the Monopoly video's method).",
        "reinforce": "Policy gradient with a value baseline: every action is scored by the reward function.",
        "ppo": "Clipped policy gradient with several passes per batch (the zombie video's method). The most stable reward method.",
    }

    def _build_train(self) -> None:
        """The training dashboard: one learner network against voting opponents."""
        # What is being trained.
        dpg.add_text(
            "One network is trained. It plays the starred tributes; every other tribute uses the voting brain.",
            wrap=360,
            color=(160, 160, 170),
        )
        # Method.
        dpg.add_combo(
            list(self.METHODS), label="method", default_value=self.method, callback=self._on_method, tag="train_method"
        )
        dpg.add_text(self.METHOD_HELP[self.method], tag="method_help", wrap=360, color=(160, 160, 170))
        # Options.
        dpg.add_checkbox(label="start from the current champion", default_value=True, tag="warm_start")
        self._tip(
            "Seeds the run with the last champion (or a loaded champion file). Pretrain by imitation, then evolve or reinforce from it."
        )
        dpg.add_checkbox(
            label="curriculum: opponents grow 1, 3, 7, 11, 23",
            default_value=False,
            tag="curriculum_on",
            callback=lambda s, a: setattr(self.curriculum, "enabled", a),
        )
        self._tip(
            "Like the zombie video's one-to-sixteen ladder: the learner faces few opponents first and is promoted when its recent mean score clears the threshold, or after 40 iterations."
        )
        with dpg.group(horizontal=True):
            dpg.add_text("Training feed")
            dpg.add_radio_button(
                list(Session.FEED_MODES),
                default_value=self.session.feed_mode,
                horizontal=True,
                callback=lambda s, a: setattr(self.session, "feed_mode", a),
                tag="feed_mode",
            )
        self._tip(
            "replay: after every iteration the arena replays one real training game. live: the newest learner plays a fresh game live (stars mark it) so the Network tab shows real activations."
        )
        # Controls.
        with dpg.group(horizontal=True):
            dpg.add_button(label="Start", callback=self._on_start_training, tag="train_start")
            dpg.add_button(label="Pause", callback=self._on_pause_training, tag="train_pause")
            dpg.add_button(label="Stop", callback=self.session.stop_training)
            dpg.add_button(label="Reset", callback=self._on_reset_training)
            dpg.add_button(label="Watch agent", callback=self._on_watch_champion)
        self._tip(
            "Start begins a run with the settings below. Pause holds it between iterations. Stop ends it after the current one. Reset forgets it. Watch agent gives the champion to the starred tributes and plays a game live."
        )
        dpg.add_progress_bar(default_value=0.0, overlay="", tag="train_progress", width=-1)
        dpg.add_text("", tag="train_summary", wrap=360, color=(160, 160, 170))
        # Latest scores.
        with dpg.plot(label="Latest scores (one bar per episode)", height=120, width=-1, tag="score_plot"):
            dpg.add_plot_axis(dpg.mvXAxis, label="", tag="score_x", no_tick_labels=True)
            with dpg.plot_axis(dpg.mvYAxis, label="score", tag="score_y"):
                dpg.add_bar_series([], [], label="score", tag="score_bars")
        # Event monitor.
        dpg.add_text("Event monitor", color=(242, 214, 72))
        with dpg.child_window(height=150, border=True, tag="event_monitor"):
            dpg.add_text("", tag="event_text", wrap=340)
        # The three graphs.
        with dpg.plot(label="Average score", height=130, width=-1, tag="perf_plot"):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="iteration", tag="perf_x")
            with dpg.plot_axis(dpg.mvYAxis, label="score", tag="perf_y"):
                dpg.add_line_series([], [], label="mean", tag="perf_train")
                dpg.add_line_series([], [], label="validation", tag="perf_val")
                dpg.add_line_series([], [], label="best", tag="perf_mean")
        with dpg.plot(label="Entropy (lower = more confident)", height=110, width=-1, tag="stab_plot"):
            dpg.add_plot_axis(dpg.mvXAxis, label="iteration", tag="stab_x")
            with dpg.plot_axis(dpg.mvYAxis, label="nats", tag="stab_y"):
                dpg.add_line_series([], [], label="entropy", tag="stab_entropy")
        with dpg.plot(label="Average game length (learner survival)", height=110, width=-1, tag="len_plot"):
            dpg.add_plot_axis(dpg.mvXAxis, label="iteration", tag="len_x")
            with dpg.plot_axis(dpg.mvYAxis, label="ticks", tag="len_y"):
                dpg.add_line_series([], [], label="length", tag="len_series")
        # Learning statistics.
        dpg.add_text("Learning statistics", color=(242, 214, 72))
        dpg.add_text("", tag="learn_stats", wrap=360)
        dpg.add_progress_bar(default_value=0.0, overlay="rollout", tag="rollout_bar", width=-1)
        dpg.add_text("", tag="system_stats", wrap=360, color=(160, 160, 170))
        # Champion genes.
        with dpg.plot(label="Learner genes (gold = changed since last step)", height=120, width=-1, tag="gene_plot"):
            dpg.add_plot_axis(dpg.mvXAxis, label="gene", tag="gene_x")
            with dpg.plot_axis(dpg.mvYAxis, label="value", tag="gene_y"):
                dpg.add_bar_series([], [], label="unchanged", tag="gene_same")
                dpg.add_bar_series([], [], label="changed", tag="gene_changed")
        # Time plot kept for the run folder view.
        with dpg.plot(label="Seconds per iteration", height=100, width=-1, tag="time_plot"):
            dpg.add_plot_axis(dpg.mvXAxis, label="iteration", tag="time_x")
            with dpg.plot_axis(dpg.mvYAxis, label="s", tag="time_y"):
                dpg.add_bar_series([], [], label="seconds", tag="time_bars")
        # Champion use and files.
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Champion to all", callback=lambda: (self.session.give_champion(), self._rebuild_roster_table())
            )
            dpg.add_button(label="Champion to selected", callback=self._on_champion_selected)
        with dpg.group(horizontal=True):
            dpg.add_input_text(label="", default_value="run", tag="run_name", width=120)
            dpg.add_button(
                label="Save run folder", callback=lambda: self.session.save_training_run(dpg.get_value("run_name"))
            )
        self._tip(
            "Writes results/<name>_<timestamp>/ with config, history, learning curves, events, champion and one PNG per chart plus GIFs."
        )
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Save champion", callback=lambda: self._file_dialog(self._save_champion, ".json", "champion.json")
            )
            dpg.add_button(
                label="Load champion into all", callback=lambda: self._file_dialog(self._load_champion, ".json")
            )
        # Advanced settings per method.
        with dpg.collapsing_header(label="Advanced settings", default_open=False):
            self._build_method_settings()

    def _build_method_settings(self) -> None:
        """The per-method settings groups (only the current method's group is shown)."""
        # Imitation.
        with dpg.group(tag="im_group", show=self.method == "imitation"):

            def im(name, convert=lambda v: v):
                return lambda s, a: setattr(self.imitation, name, convert(a))

            dpg.add_combo(
                ["voting"], label="teacher brain", default_value=self.imitation.teacher, callback=im("teacher")
            )
            dpg.add_input_int(
                label="demonstration games",
                default_value=self.imitation.demonstration_games,
                min_value=1,
                max_value=200,
                min_clamped=True,
                max_clamped=True,
                callback=im("demonstration_games"),
            )
            dpg.add_input_int(
                label="learn only from the top N placings (0 = all)",
                default_value=self.imitation.winners_top,
                min_value=0,
                max_value=24,
                min_clamped=True,
                max_clamped=True,
                callback=im("winners_top"),
            )
            self._tip("Show it a few winning games: keep only the decisions of tributes that placed this well.")
            dpg.add_input_int(
                label="epochs",
                default_value=self.imitation.epochs,
                min_value=1,
                max_value=1000,
                min_clamped=True,
                max_clamped=True,
                callback=im("epochs"),
            )
            dpg.add_input_int(
                label="batch size",
                default_value=self.imitation.batch_size,
                min_value=8,
                max_value=4096,
                min_clamped=True,
                max_clamped=True,
                callback=im("batch_size"),
            )
            dpg.add_input_float(
                label="learning rate",
                default_value=self.imitation.learning_rate,
                format="%.5f",
                callback=im("learning_rate"),
            )
            dpg.add_input_int(
                label="validation games",
                default_value=self.imitation.validation_games,
                min_value=0,
                max_value=20,
                min_clamped=True,
                max_clamped=True,
                callback=im("validation_games"),
            )
            dpg.add_input_int(
                label="CPU workers",
                default_value=self.imitation.workers,
                min_value=1,
                max_value=32,
                min_clamped=True,
                max_clamped=True,
                callback=im("workers"),
            )
        # Genetic.
        with dpg.group(tag="ga_group", show=self.method == "genetic"):

            def ga(name, convert=lambda v: v):
                return lambda s, a: setattr(self.ga, name, convert(a))

            dpg.add_combo(
                ["neural", "voting"],
                label="brain to evolve",
                default_value=self.ga.brain_name,
                callback=ga("brain_name"),
            )
            dpg.add_combo(
                ["voting", "self"], label="opponents", default_value=self.ga.opponents, callback=ga("opponents")
            )
            self._tip(
                "voting: each genome is the learner against the voting brain (scored by return). self: the population plays itself (scored by placement)."
            )
            dpg.add_input_int(
                label="population",
                default_value=self.ga.population_size,
                min_value=4,
                max_value=480,
                min_clamped=True,
                max_clamped=True,
                callback=ga("population_size"),
            )
            dpg.add_input_int(
                label="generations",
                default_value=self.ga.generations,
                min_value=1,
                max_value=1000,
                min_clamped=True,
                max_clamped=True,
                callback=ga("generations"),
            )
            dpg.add_input_int(
                label="games per genome",
                default_value=self.ga.rounds_per_generation,
                min_value=1,
                max_value=10,
                min_clamped=True,
                max_clamped=True,
                callback=ga("rounds_per_generation"),
            )
            dpg.add_slider_float(
                label="elite fraction",
                default_value=self.ga.elite_fraction,
                min_value=0.0,
                max_value=0.5,
                callback=ga("elite_fraction"),
            )
            dpg.add_slider_float(
                label="mutation rate",
                default_value=self.ga.mutation_rate,
                min_value=0.0,
                max_value=1.0,
                callback=ga("mutation_rate"),
            )
            dpg.add_slider_float(
                label="mutation scale",
                default_value=self.ga.mutation_scale,
                min_value=0.001,
                max_value=1.0,
                callback=ga("mutation_scale"),
            )
            dpg.add_slider_float(
                label="crossover rate",
                default_value=self.ga.crossover_rate,
                min_value=0.0,
                max_value=1.0,
                callback=ga("crossover_rate"),
            )
            dpg.add_input_int(
                label="validation games",
                default_value=self.ga.validation_games,
                min_value=0,
                max_value=20,
                min_clamped=True,
                max_clamped=True,
                callback=ga("validation_games"),
            )
            dpg.add_input_int(
                label="CPU workers",
                default_value=self.ga.workers,
                min_value=1,
                max_value=32,
                min_clamped=True,
                max_clamped=True,
                callback=ga("workers"),
            )
        # NEAT.
        with dpg.group(tag="neat_group", show=self.method == "neat"):

            def ne(name, convert=lambda v: v):
                return lambda s, a: setattr(self.neat, name, convert(a))

            dpg.add_input_int(
                label="population",
                default_value=self.neat.population_size,
                min_value=4,
                max_value=480,
                min_clamped=True,
                max_clamped=True,
                callback=ne("population_size"),
            )
            dpg.add_input_int(
                label="generations",
                default_value=self.neat.generations,
                min_value=1,
                max_value=1000,
                min_clamped=True,
                max_clamped=True,
                callback=ne("generations"),
            )
            dpg.add_input_int(
                label="target species",
                default_value=self.neat.target_species,
                min_value=1,
                max_value=40,
                min_clamped=True,
                max_clamped=True,
                callback=ne("target_species"),
            )
            dpg.add_slider_float(
                label="add node rate",
                default_value=self.neat.neat.add_node_rate,
                min_value=0.0,
                max_value=0.5,
                callback=lambda s, a: setattr(self.neat.neat, "add_node_rate", a),
            )
            dpg.add_slider_float(
                label="add connection rate",
                default_value=self.neat.neat.add_connection_rate,
                min_value=0.0,
                max_value=0.5,
                callback=lambda s, a: setattr(self.neat.neat, "add_connection_rate", a),
            )
            dpg.add_input_int(
                label="validation games",
                default_value=self.neat.validation_games,
                min_value=0,
                max_value=20,
                min_clamped=True,
                max_clamped=True,
                callback=ne("validation_games"),
            )
            dpg.add_input_int(
                label="CPU workers",
                default_value=self.neat.workers,
                min_value=1,
                max_value=32,
                min_clamped=True,
                max_clamped=True,
                callback=ne("workers"),
            )
        # REINFORCE and PPO share the reward function.
        with dpg.group(tag="rl_group", show=self.method in ("reinforce", "ppo")):

            def rl(name, convert=lambda v: v):
                def cb(s, a):
                    setattr(self.rl, name, convert(a))
                    setattr(self.ppo, name, convert(a))

                return cb

            dpg.add_input_int(
                label="epochs",
                default_value=self.rl.epochs,
                min_value=1,
                max_value=10000,
                min_clamped=True,
                max_clamped=True,
                callback=rl("epochs"),
            )
            dpg.add_input_int(
                label="games per epoch",
                default_value=self.rl.episodes_per_epoch,
                min_value=1,
                max_value=64,
                min_clamped=True,
                max_clamped=True,
                callback=rl("episodes_per_epoch"),
            )
            dpg.add_input_int(
                label="learner copies per game",
                default_value=self.rl.learners_per_game,
                min_value=1,
                max_value=24,
                min_clamped=True,
                max_clamped=True,
                callback=rl("learners_per_game"),
            )
            self._tip("The same network drives this many starred tributes at once; the rest use the voting brain.")
            dpg.add_input_float(
                label="learning rate", default_value=self.rl.learning_rate, format="%.5f", callback=rl("learning_rate")
            )
            dpg.add_slider_float(
                label="entropy bonus",
                default_value=self.rl.entropy_bonus,
                min_value=0.0,
                max_value=0.2,
                callback=rl("entropy_bonus"),
            )
            dpg.add_slider_float(
                label="PPO clip ratio",
                default_value=self.ppo.clip_ratio,
                min_value=0.05,
                max_value=0.5,
                callback=lambda s, a: setattr(self.ppo, "clip_ratio", a),
            )
            dpg.add_input_int(
                label="PPO passes per batch",
                default_value=self.ppo.update_epochs,
                min_value=1,
                max_value=20,
                min_clamped=True,
                max_clamped=True,
                callback=lambda s, a: setattr(self.ppo, "update_epochs", a),
            )
            dpg.add_input_int(
                label="validation games",
                default_value=self.rl.validation_games,
                min_value=0,
                max_value=20,
                min_clamped=True,
                max_clamped=True,
                callback=rl("validation_games"),
            )
            dpg.add_input_int(
                label="CPU workers",
                default_value=self.rl.workers,
                min_value=1,
                max_value=32,
                min_clamped=True,
                max_clamped=True,
                callback=rl("workers"),
            )
            with dpg.collapsing_header(label="Reward function", default_open=False):
                r = self.session.config.reward

                def rw(name):
                    return lambda s, a: setattr(self.session.config.reward, name, a)

                dpg.add_slider_float(
                    label="per tick alive",
                    default_value=r.survive_tick,
                    min_value=0.0,
                    max_value=0.1,
                    format="%.3f",
                    callback=rw("survive_tick"),
                )
                self._tip("The zombie video's lesson: reward survival too much and the learner just runs away.")
                dpg.add_slider_float(
                    label="win", default_value=r.win, min_value=0.0, max_value=20.0, callback=rw("win")
                )
                dpg.add_slider_float(
                    label="death", default_value=r.death, min_value=-20.0, max_value=0.0, callback=rw("death")
                )
                dpg.add_slider_float(
                    label="kill", default_value=r.kill, min_value=0.0, max_value=10.0, callback=rw("kill")
                )
                dpg.add_slider_float(
                    label="per health lost",
                    default_value=r.damage_taken,
                    min_value=-10.0,
                    max_value=0.0,
                    callback=rw("damage_taken"),
                )
                dpg.add_slider_float(
                    label="per need restored",
                    default_value=r.need_gain,
                    min_value=0.0,
                    max_value=5.0,
                    callback=rw("need_gain"),
                )
                dpg.add_slider_float(
                    label="approach water/food (dense, off by default)",
                    default_value=r.approach,
                    min_value=0.0,
                    max_value=0.5,
                    format="%.3f",
                    callback=rw("approach"),
                )
                dpg.add_slider_float(
                    label="placement",
                    default_value=r.placement,
                    min_value=0.0,
                    max_value=10.0,
                    callback=rw("placement"),
                )
                dpg.add_slider_float(
                    label="discount",
                    default_value=r.discount,
                    min_value=0.8,
                    max_value=1.0,
                    format="%.3f",
                    callback=rw("discount"),
                )
        # Curriculum settings.
        with dpg.collapsing_header(label="Curriculum settings", default_open=False):
            dpg.add_input_text(
                label="opponents per stage",
                default_value=",".join(str(o) for o in self.curriculum.opponents),
                callback=lambda s, a: setattr(
                    self.curriculum, "opponents", tuple(int(x) for x in a.split(",") if x.strip().isdigit()) or (23,)
                ),
            )
            dpg.add_slider_float(
                label="promotion threshold (mean score)",
                default_value=self.curriculum.threshold,
                min_value=-5.0,
                max_value=10.0,
                callback=lambda s, a: setattr(self.curriculum, "threshold", a),
            )
            dpg.add_input_int(
                label="max iterations per stage",
                default_value=self.curriculum.max_iterations_per_stage,
                min_value=1,
                max_value=1000,
                min_clamped=True,
                max_clamped=True,
                callback=lambda s, a: setattr(self.curriculum, "max_iterations_per_stage", a),
            )

    def _on_method(self, sender, value) -> None:
        """Switch the settings group and the help text."""
        # Remember.
        self.method = value
        dpg.set_value("method_help", self.METHOD_HELP[value])
        # Show the right group.
        for tag, methods in (
            ("im_group", ("imitation",)),
            ("ga_group", ("genetic",)),
            ("neat_group", ("neat",)),
            ("rl_group", ("reinforce", "ppo")),
        ):
            dpg.configure_item(tag, show=value in methods)

    def _current_settings(self):
        """The settings dataclass for the current method (a fresh copy)."""
        # Per method.
        if self.method == "imitation":
            return ImitationConfig(**vars(self.imitation))
        if self.method == "genetic":
            return TrainingConfig(**vars(self.ga))
        if self.method == "neat":
            return NeatTrainerConfig(**vars(self.neat))
        if self.method == "reinforce":
            return RLConfig(**vars(self.rl))
        return PPOConfig(**vars(self.ppo))

    def _on_start_training(self) -> None:
        """Start the trainer with the current settings."""
        # Reset the plots.
        self._plotted_steps = -1
        self._events_shown = 0
        # Go.
        self.session.start_training(
            self._current_settings(),
            self.method,
            bool(dpg.get_value("warm_start")),
            CurriculumConfig(**vars(self.curriculum)),
        )

    def _on_pause_training(self) -> None:
        """Toggle pause."""
        # Flip.
        self.session.pause_training(not self.session.training_paused)

    def _on_reset_training(self) -> None:
        """Forget the current run."""
        # Reset.
        self.session.reset_training()
        self._plotted_steps = -1
        self._events_shown = 0
        for tag in (
            "perf_train",
            "perf_val",
            "perf_mean",
            "stab_entropy",
            "len_series",
            "time_bars",
            "gene_same",
            "gene_changed",
            "score_bars",
        ):
            dpg.set_value(tag, [[], []])
        dpg.set_value("event_text", "")
        dpg.set_value("train_summary", "")
        dpg.set_value("learn_stats", "")

    def _on_champion_selected(self) -> None:
        """Give the champion to the selected tribute."""
        # Need one.
        if self.session.selected_id is not None:
            self.session.give_champion([self.session.selected_id])
            self._rebuild_roster_table()

    def _on_watch_champion(self) -> None:
        """Give the champion to the learner slots and start a game at normal speed."""
        # Give and play.
        if not self.session.start_champion_game(all_slots=False):
            return
        self._rebuild_roster_table()
        self._set_speed(8.0)
        dpg.set_value("right_tabs", "tab_network")

    def _refresh_training(self) -> None:
        """Update the dashboard panels from the session."""
        # Progress.
        done, total = self.session.training_progress
        running = self.session.training_running
        dpg.set_value("train_progress", done / total if total else 0.0)
        dpg.set_value("rollout_bar", done / total if total else 0.0)
        dpg.configure_item(
            "rollout_bar",
            overlay=f"rollout {done}/{total} games ({(done / total * 100) if total else 0:.0f}%)"
            if total
            else "rollout",
        )
        steps = len(self.session.training_history())
        dpg.configure_item(
            "train_progress",
            overlay=f"iteration {steps}" + (" (paused)" if self.session.training_paused else "")
            if running
            else (f"{steps} iterations done" if steps else ""),
        )
        dpg.configure_item("train_start", enabled=not running)
        dpg.configure_item("train_pause", label="Resume" if self.session.training_paused else "Pause")
        # Event monitor.
        events = self.session.training_events()
        if len(self.session.trainer.events.events) != self._events_shown if self.session.trainer is not None else False:
            self._events_shown = len(self.session.trainer.events.events)
            dpg.set_value("event_text", "\n".join(events))
        # System stats every frame is cheap.
        stats = self.session.learning_stats()
        system = self.session.system.read()
        dpg.set_value(
            "system_stats",
            f"CPU {system['cpu_percent']:.0f}%   memory {system['memory_mb']:.0f} MB ({system['memory_percent']:.0f}% of RAM)   GPU: {system['gpu']}",
        )
        dpg.set_value(
            "learn_stats",
            f"iteration {stats['iteration']}   seed {stats['seed']}   {stats['seconds_per_iteration']:.1f} s/iteration   max score {stats['max_score']:.2f}   learning time {stats['learning_time']:.0f} s\nstage {stats['stage']} ({stats['opponents']} opponents)   mean score {stats['mean_score']:.2f}   entropy {stats['entropy']:.2f}   mean length {stats['mean_length']:.0f} ticks",
        )
        # Plot only when there is something new.
        rows = self.session.training_rows()
        if len(rows) == self._plotted_steps:
            return
        self._plotted_steps = len(rows)
        if not rows:
            return
        xs = [float(r["iteration"]) for r in rows]
        dpg.set_value("perf_train", [xs, [r["mean_score"] for r in rows]])
        dpg.set_value("perf_val", [xs, [r["val_score"] for r in rows]])
        dpg.set_value("perf_mean", [xs, [r["best_score"] for r in rows]])
        dpg.set_value("stab_entropy", [xs, [r["entropy"] for r in rows]])
        dpg.set_value("len_series", [xs, [r["mean_length"] for r in rows]])
        dpg.set_value("time_bars", [xs, [r["seconds"] for r in rows]])
        scores = self.session.latest_scores()
        dpg.set_value("score_bars", [list(range(len(scores))), scores])
        # Genes.
        genes = self.session.champion_genes()
        if genes is not None:
            values, changed = genes
            n = min(len(values), 400)
            idx = np.arange(n, dtype=float)
            dpg.set_value("gene_same", [idx[~changed[:n]].tolist(), values[:n][~changed[:n]].tolist()])
            dpg.set_value("gene_changed", [idx[changed[:n]].tolist(), values[:n][changed[:n]].tolist()])
            if len(values) == len(GENE_NAMES):
                dpg.set_axis_ticks("gene_x", tuple((name, float(i)) for i, name in enumerate(GENE_NAMES)))
        # The network evolution plots on the Network tab.
        self._refresh_evolution()
        # Fit.
        for axis in (
            "perf_x",
            "perf_y",
            "stab_x",
            "stab_y",
            "len_x",
            "len_y",
            "time_x",
            "time_y",
            "gene_x",
            "gene_y",
            "score_x",
            "score_y",
        ):
            dpg.fit_axis_data(axis)
        # Summary.
        last = rows[-1]
        extra = {k[6:]: v for k, v in last.items() if k.startswith("extra_")}
        extra_text = ", ".join(
            f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in list(extra.items())[:4]
        )
        dpg.set_value(
            "train_summary",
            f"{self.session.training_method}: {len(rows)} iteration(s), mean score {last['mean_score']:.2f}, validation {last['val_score']:.2f}, win rate {last['win_rate']:.2f}. {extra_text}",
        )

    # ---------------------------------------------------------- research

    def _build_research(self) -> None:
        """Parameter sweeps and chart exports."""
        # Sweep.
        dpg.add_text("Parameter sweep", color=(242, 214, 72))
        dpg.add_combo(SWEEPABLE, label="parameter", default_value="chaos", tag="sweep_param")
        self._tip("Any config field; nested ones use a dot. Each value plays the same seeded games on the painted map.")
        dpg.add_input_text(label="values", default_value="0,0.25,0.5,0.75,1", tag="sweep_values")
        self._tip("Comma-separated. Booleans as true/false.")
        dpg.add_input_int(
            label="games per value", default_value=20, min_value=1, max_value=1000, min_clamped=True, tag="sweep_games"
        )
        dpg.add_input_int(
            label="CPU workers", default_value=1, min_value=1, max_value=32, min_clamped=True, tag="sweep_workers"
        )
        dpg.add_checkbox(label="collect behaviour telemetry", default_value=True, tag="sweep_telemetry")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Start sweep", callback=self._on_start_sweep)
            dpg.add_button(label="Stop", callback=self.session.stop_sweep)
        dpg.add_progress_bar(default_value=0.0, overlay="", tag="sweep_progress", width=-1)
        dpg.add_text("", tag="sweep_results", wrap=360, color=(160, 160, 170))
        dpg.add_separator()
        # Exports.
        dpg.add_text("Charts of the games you have watched", color=(242, 214, 72))
        dpg.add_text(
            "Every finished game watched in this session (including back-to-back ones) is measured. Export one PNG per chart for a paper.",
            wrap=360,
            color=(160, 160, 170),
        )
        dpg.add_input_text(label="folder", default_value="output/watched", tag="export_folder")
        dpg.add_button(
            label="Export behaviour charts",
            callback=lambda: self.session.export_behaviour_plots(dpg.get_value("export_folder")),
        )
        dpg.add_button(label="Forget watched games", callback=lambda: self.session.watched_summaries.clear())
        dpg.add_separator()
        # Questions.
        dpg.add_text("Answers a reviewer will ask for", color=(242, 214, 72))
        dpg.add_text(
            "Method: imitation (behaviour cloning of the voting brain), genetic algorithm (neuroevolution), NEAT (neuroevolution of topologies), REINFORCE with a value baseline, or PPO (clipped policy gradient), chosen on the Train tab, with warm starts between them and an optional opponent curriculum; experiments/run_comparison.py trains them all under one budget and runs a 75-game tournament. Rewards: the Reward function section there (the dense approach reward is off by default). Observation: a 50-value vector (Brains tab lists it), not a grid. Dashboard: custom, Dear PyGui; charts by matplotlib.",
            wrap=360,
            color=(160, 160, 170),
        )

    def _on_start_sweep(self) -> None:
        """Start a sweep from the widgets."""

        # Parse values.
        def parse(text: str):
            if text.lower() in ("true", "false"):
                return text.lower() == "true"
            try:
                return int(text)
            except ValueError:
                pass
            try:
                return float(text)
            except ValueError:
                return text

        values = [parse(v.strip()) for v in dpg.get_value("sweep_values").split(",") if v.strip()]
        # Settings.
        settings = SweepConfig(
            name=dpg.get_value("sweep_param").replace(".", "_"),
            parameter=dpg.get_value("sweep_param"),
            values=values,
            games_per_value=int(dpg.get_value("sweep_games")),
            workers=int(dpg.get_value("sweep_workers")),
            telemetry=bool(dpg.get_value("sweep_telemetry")),
        )
        # Go.
        self.session.start_sweep(settings)

    def _refresh_research(self) -> None:
        """Update the sweep progress and results text."""
        # Progress.
        done, total = self.session.sweep_progress
        dpg.set_value("sweep_progress", done / total if total else 0.0)
        dpg.configure_item("sweep_progress", overlay=f"{done}/{total} values" if self.session.sweep_running else "")
        # Results.
        if self.session.sweep is not None and self.session.sweep.rows:
            lines = [
                f"{r['value']}: victors {r['victor_rate']:.0%}, days {r['mean_days']:.1f}, pvp {r['player_vs_player_share']:.0%}, natural {r['natural_share']:.0%}"
                for r in self.session.sweep.rows
            ]
            dpg.set_value("sweep_results", "\n".join(lines))

    # --------------------------------------------------------- transport

    def _build_transport(self) -> None:
        """Buttons and sliders under the arena."""
        # Buttons.
        with dpg.group(horizontal=True):
            dpg.add_button(label="New game", callback=self._on_new_game)
            self._tip("Start a fresh game from the current settings, map and roster.")
            dpg.add_button(label="Play", callback=self._on_play, tag="play_button", width=70)
            self._tip("Play or pause. Starts a game if there is none.")
            dpg.add_button(label="Step", callback=self._on_step)
            dpg.add_button(label="To end", callback=self._on_to_end)
            self._tip("Simulate the rest of the game instantly, then scrub back to watch any part.")
            dpg.add_button(label="Rewind", callback=self.session.rewind)
        # Speed and scrub.
        dpg.add_slider_float(
            label="ticks / second",
            default_value=self.session.ticks_per_second,
            min_value=0.5,
            max_value=400.0,
            callback=lambda s, a: setattr(self.session, "ticks_per_second", a),
            tag="speed_slider",
            width=-120,
        )
        dpg.add_slider_int(
            label="frame",
            default_value=0,
            min_value=0,
            max_value=0,
            callback=lambda s, a: self.session.seek(a),
            tag="playhead",
            width=-120,
        )
        # Headline.
        dpg.add_text("", tag="headline", color=(242, 214, 72))

    def _ensure_game(self) -> None:
        """Start a game if none is loaded."""
        # Need one.
        if self.session.recording is None:
            self.session.new_game()

    def _on_new_game(self) -> None:
        """Start a new game."""
        # Start.
        self.session.new_game()

    def _on_play(self) -> None:
        """Toggle play / pause."""
        # Make sure there is a game.
        self._ensure_game()
        # Toggle.
        self.session.playing = not self.session.playing

    def _on_step(self) -> None:
        """One tick."""
        # Make sure there is a game.
        self._ensure_game()
        # Step.
        self.session.playing = False
        self.session.step_once()

    def _on_to_end(self) -> None:
        """Finish the game instantly."""
        # Make sure there is a game.
        self._ensure_game()
        # Run.
        self.session.run_to_end()

    def _refresh_transport(self) -> None:
        """Keep the scrub slider, play button and headline in step with the session."""
        # Side texts that do not need a game.
        dpg.set_value("loot_count", f"hand-placed stacks: {len(self.session.scenario.loot)}")
        dpg.set_value("loot_weapon_name", f"weapon at this quality: {weapon_name(self.loot_quality)}")
        dpg.set_value(
            "map_coverage", "coverage: " + ", ".join(f"{k} {v:.0%}" for k, v in self.session.painter.coverage().items())
        )
        # Play button label.
        dpg.configure_item("play_button", label="Pause" if self.session.playing else "Play")
        # The recording.
        rec = self.session.recording
        # Nothing loaded.
        if rec is None:
            dpg.set_value("headline", "No game yet. Press New game or Play.")
            return
        # Slider range and value.
        dpg.configure_item("playhead", max_value=max(0, rec.length - 1))
        dpg.set_value("playhead", self.session.playhead)
        # The frame.
        frame = self.session.current_frame
        # Alive count.
        alive = sum(p.alive for p in frame.players)
        # Headline.
        text = f"Day {frame.day}   tick {frame.tick}   alive {alive}/{len(frame.players)}   frame {self.session.playhead + 1}/{rec.length}"
        # Outcome.
        if rec.result is not None and self.session.playhead == rec.length - 1:
            text += f"   VICTOR: {rec.result.winner_name}" if rec.result.winner_name else "   no victor (draw)"
        # The training feed's label, if any.
        if self.session.feed_mode != "off" and self.session.feed_label:
            text = self.session.feed_label + "   |   " + text
        # Show.
        dpg.set_value("headline", text)

    # --------------------------------------------------------- inspector

    def _build_inspector(self) -> None:
        """The selected tribute's stats and the event log."""
        # Title.
        dpg.add_text("Click a tribute on the arena", tag="insp_title", color=(242, 214, 72))
        dpg.add_text("", tag="insp_facts", wrap=340)
        for name in ("thirst", "hunger", "health"):
            dpg.add_progress_bar(default_value=0.0, overlay=name, tag=f"insp_{name}", width=-1)
        dpg.add_text("", tag="insp_more", wrap=340)
        dpg.add_separator()
        dpg.add_text("Event log", color=(242, 214, 72))
        dpg.add_text("", tag="insp_log", wrap=340)

    def _refresh_inspector(self) -> None:
        """Fill the inspector from the selected tribute."""
        # Selected id.
        pid = self.session.selected_id
        # Log always.
        dpg.set_value("insp_log", "\n".join(self.session.event_log()))
        # Nothing selected.
        if pid is None:
            dpg.set_value("insp_title", "Click a tribute on the arena")
            dpg.set_value("insp_facts", "")
            dpg.set_value("insp_more", "")
            return
        # Roster facts.
        spec = self.session.tribute(pid)
        # Unknown.
        if spec is None:
            return
        # Title and facts.
        dpg.set_value("insp_title", spec.name)
        dpg.set_value(
            "insp_facts",
            f"District {spec.district} ({DISTRICT_INDUSTRIES[spec.district]}), {'female' if spec.sex == 'F' else 'male'}\nTraining score {spec.training_score}, survival {spec.survival_score:.2f}, brain {spec.brain_name}{' (trained)' if spec.genome else ''}",
        )
        # Live state.
        frame = self.session.current_frame
        # Editing: show granted items.
        if frame is None:
            dpg.set_value("insp_thirst", spec.start_thirst or self.session.config.start_thirst_min)
            dpg.set_value("insp_hunger", spec.start_hunger or self.session.config.start_hunger_min)
            dpg.set_value("insp_health", spec.start_health or self.session.config.start_health_min)
            dpg.set_value(
                "insp_more",
                f"Granted: {weapon_name(spec.weapon_quality)}, {spec.food} food, {spec.medicine} medkits, favour +{spec.favor_bonus:.2f}\nPodium {spec.podium}",
            )
            return
        # The snapshot.
        snap = next((p for p in frame.players if p.player_id == pid), None)
        # Missing.
        if snap is None:
            return
        # Bars.
        dpg.set_value("insp_thirst", snap.thirst)
        dpg.set_value("insp_hunger", snap.hunger)
        dpg.set_value("insp_health", snap.health)
        # Outcome text.
        outcome = "alive"
        if not snap.alive and self.session.recording is not None:
            for f in self.session.recording.frames[: self.session.playhead + 1]:
                for e in f.eliminations:
                    if e.victim_id == pid:
                        outcome = f"eliminated day {e.day} ({e.weapon}{' by ' + e.killer_name if e.killer_name else ''}), placed {e.placement}"
        # More.
        dpg.set_value(
            "insp_more",
            f"{outcome}\nWeapon: {weapon_name(snap.weapon_quality)} ({snap.weapon_quality:.2f}), reach {1 if snap.weapon_quality < 0.6 else (2 if snap.weapon_quality < 0.9 else 3)}\nFood {snap.food}, medkits {snap.medicine}, kills {snap.kills}\nSponsor favour {snap.favor:.2f}\nLast action: {snap.last_action or '-'}",
        )

    # ----------------------------------------------------------- network

    def _build_network(self) -> None:
        """The neural network visualiser."""
        # Explain.
        dpg.add_text(
            "Select a neural tribute during a live game to watch its layers react.", wrap=340, color=(160, 160, 170)
        )
        dpg.add_text("", tag="network_caption", wrap=340)
        # Evolution over training.
        with dpg.collapsing_header(
            label="How the champion network changed over training", default_open=False, tag="evolution_header"
        ):
            with dpg.plot(label="Genome change per step (L2) and mean |weight|", height=150, width=-1, tag="evo_plot"):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="step", tag="evo_x")
                with dpg.plot_axis(dpg.mvYAxis, label="value", tag="evo_y"):
                    dpg.add_line_series([], [], label="change from previous step", tag="evo_change")
                    dpg.add_line_series([], [], label="mean |weight|", tag="evo_mean")
            with dpg.plot(
                label="Champion genome by step (rows = steps, columns = first 200 genes)",
                height=170,
                width=-1,
                tag="evo_heat_plot",
            ):
                dpg.add_plot_axis(dpg.mvXAxis, label="gene", tag="evo_heat_x")
                dpg.add_plot_axis(dpg.mvYAxis, label="step", tag="evo_heat_y")
            dpg.bind_colormap("evo_heat_plot", dpg.mvPlotColormap_RdBu)
        # The drawing, in its own scrollable holder.
        with dpg.child_window(tag="network_holder", border=False, height=-1):
            self.visualizer.build("network_holder")

    def _refresh_evolution(self) -> None:
        """Update the genome-evolution plots when a new training step exists."""
        # The data.
        data = self.session.network_evolution()
        # Nothing.
        if data is None:
            return
        # Lines.
        dpg.set_value("evo_change", [data["steps"], data["change"]])
        dpg.set_value("evo_mean", [data["steps"], data["mean_abs"]])
        dpg.fit_axis_data("evo_x")
        dpg.fit_axis_data("evo_y")
        # The heat series is recreated because its row count grows.
        if dpg.does_item_exist("evo_heat_series"):
            dpg.delete_item("evo_heat_series")
        genes = np.asarray(data["genes"], dtype=float)
        limit = float(np.abs(genes).max()) or 1.0
        dpg.add_heat_series(
            genes.ravel().tolist(),
            genes.shape[0],
            genes.shape[1],
            scale_min=-limit,
            scale_max=limit,
            format="",
            parent="evo_heat_y",
            tag="evo_heat_series",
            bounds_min=(0, 0),
            bounds_max=(genes.shape[1], genes.shape[0]),
        )
        dpg.fit_axis_data("evo_heat_x")
        dpg.fit_axis_data("evo_heat_y")

    def _refresh_network(self) -> None:
        """Redraw the network for the selected tribute."""
        # Snapshot.
        snapshot = self.session.network_snapshot(self.session.selected_id)
        # Architecture for the empty diagram.
        architecture = [VECTOR_SIZE, *self.session.config.neural.hidden_layers, MENU_SIZE]
        # Draw.
        self.visualizer.render(snapshot, architecture)
        # Caption.
        if snapshot is not None:
            chosen = snapshot["menu"][snapshot["chosen"]]
            dpg.set_value(
                "network_caption",
                f"Live: {self.session.tribute(self.session.selected_id).name}. Chosen: {chosen}. Red = positive activation, blue = negative.",
            )
        else:
            dpg.set_value(
                "network_caption",
                f"Architecture: {' -> '.join(str(s) for s in architecture)} ({self.session.config.neural.activation}, {self.session.config.neural.initializer}).",
            )

    # ------------------------------------------------------------ charts

    def _build_charts(self) -> None:
        """Live behaviour charts of the games watched this session."""
        # Explain.
        dpg.add_text(
            "Behaviour of every game watched this session (updates as games play).", wrap=340, color=(160, 160, 170)
        )
        # Action distribution.
        with dpg.plot(label="Action distribution (%)", height=170, width=-1, tag="chart_actions"):
            dpg.add_plot_axis(dpg.mvXAxis, label="", tag="chart_actions_x")
            with dpg.plot_axis(dpg.mvYAxis, label="%", tag="chart_actions_y"):
                dpg.add_bar_series([], [], label="actions", tag="chart_actions_bars")
        # Instinct curves.
        with dpg.plot(label="Instinct curves (%)", height=170, width=-1, tag="chart_instinct"):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="need bar level", tag="chart_instinct_x")
            with dpg.plot_axis(dpg.mvYAxis, label="% of decisions", tag="chart_instinct_y"):
                dpg.add_line_series([], [], label="drink | thirst", tag="chart_drink")
                dpg.add_line_series([], [], label="eat | hunger", tag="chart_eat")
                dpg.add_line_series([], [], label="flee | health", tag="chart_flee")
        # Heatmap.
        with dpg.plot(label="Where tributes spend time", height=260, width=-1, tag="chart_heat", equal_aspects=True):
            dpg.add_plot_axis(dpg.mvXAxis, label="", tag="chart_heat_x", no_tick_labels=True)
            with dpg.plot_axis(dpg.mvYAxis, label="", tag="chart_heat_y", no_tick_labels=True):
                dpg.add_heat_series(
                    [0.0] * 900,
                    30,
                    30,
                    scale_min=0.0,
                    scale_max=1.0,
                    format="",
                    tag="chart_heat_series",
                    bounds_min=(0, 0),
                    bounds_max=(30, 30),
                )
        dpg.bind_colormap("chart_heat", dpg.mvPlotColormap_Viridis)

    def _refresh_charts(self) -> None:
        """Update the behaviour charts from the session's telemetry."""
        # The data.
        summary = self.session.watched_summary()
        # Nothing yet.
        if summary is None:
            return
        # Action distribution.
        counts = np.asarray(summary["action_counts"], dtype=float)
        shares = counts / max(1.0, counts.sum()) * 100
        dpg.set_value("chart_actions_bars", [list(range(len(shares))), shares.tolist()])
        dpg.set_axis_ticks("chart_actions_x", tuple((name, float(i)) for i, name in enumerate(summary["action_names"])))
        # Instinct curves.
        names = summary["action_names"]
        thirst = np.asarray(summary["action_by_thirst"], dtype=float)
        hunger = np.asarray(summary["action_by_hunger"], dtype=float)
        health = np.asarray(summary["action_by_health"], dtype=float)
        xs = list(range(5))
        dpg.set_value(
            "chart_drink", [xs, (thirst[:, names.index("drink")] / np.maximum(1.0, thirst.sum(axis=1)) * 100).tolist()]
        )
        dpg.set_value(
            "chart_eat", [xs, (hunger[:, names.index("eat")] / np.maximum(1.0, hunger.sum(axis=1)) * 100).tolist()]
        )
        dpg.set_value(
            "chart_flee", [xs, (health[:, names.index("flee")] / np.maximum(1.0, health.sum(axis=1)) * 100).tolist()]
        )
        dpg.set_axis_ticks("chart_instinct_x", tuple((label, float(i)) for i, label in enumerate(NEED_BIN_LABELS)))
        # Heatmap, flipped so row 0 is at the top like the arena.
        heat = np.asarray(summary["position_heat"], dtype=float)[::-1]
        top = heat.max() if heat.max() > 0 else 1.0
        dpg.set_value("chart_heat_series", [(heat / top).ravel().tolist()])
        # Fit.
        for axis in ("chart_actions_x", "chart_actions_y", "chart_instinct_x", "chart_instinct_y"):
            dpg.fit_axis_data(axis)

    # ------------------------------------------------------------- mouse

    def _on_mouse_down(self, sender, button) -> None:
        """Held left button: paint or drag."""
        # Left only.
        if button != dpg.mvMouseButton_Left:
            return
        # Cell under the mouse.
        cell = self.canvas.mouse_cell()
        # Not over the arena.
        if cell is None:
            return
        # Paint (only while editing, not during a game).
        if self.tool == "Paint terrain" and self.session.game is None:
            self.session.paint(cell[0], cell[1], self.brush_terrain, self.brush_radius)
            self.painting = True
        # Drag a tribute.
        elif self.tool == "Move tribute" and self.session.game is None:
            if self.drag_id is None:
                self.drag_id = self.session.tribute_at(*cell)
            if self.drag_id is not None:
                self.session.move_tribute(self.drag_id, *cell)

    def _on_mouse_release(self, sender, button) -> None:
        """Released: end strokes and drags."""
        # End a stroke.
        if self.painting:
            self.session.finish_painting()
            self.session.reposition_off_void()
            self.painting = False
        # Drop a tribute.
        self.drag_id = None

    def _on_mouse_click(self, sender, button) -> None:
        """A click: select, or place / remove loot."""
        # Cell under the mouse.
        cell = self.canvas.mouse_cell()
        # Not over the arena.
        if cell is None:
            return
        # Select (Select tool, or right click in any tool but loot).
        if self.tool == "Select" or (button == dpg.mvMouseButton_Right and self.tool != "Place loot"):
            self._select(self.session.tribute_at(*cell))
        # Loot (only while editing).
        elif self.tool == "Place loot" and self.session.game is None:
            if button == dpg.mvMouseButton_Left:
                self.session.place_loot(cell[0], cell[1], self.loot_kind, self.loot_quantity, self.loot_quality)
            else:
                self.session.remove_loot(*cell)

    # ------------------------------------------------------------- files

    def _file_dialog(self, callback, extension: str, default_filename: str | None = None) -> None:
        """Open a file dialog and call `callback(path)` with the chosen path."""

        # The dialog's own callback unpacks the path.
        def chosen(sender, app_data):
            path = app_data.get("file_path_name", "")
            if path and not path.endswith(extension):
                path += extension
            if path:
                try:
                    callback(path)
                except Exception as error:  # noqa: BLE001 - show any problem in the status bar
                    self.session.status = f"Error: {error}"

        # Build the dialog.
        with dpg.file_dialog(
            directory_selector=False,
            show=True,
            callback=chosen,
            width=760,
            height=460,
            modal=True,
            default_filename=default_filename or "",
        ):
            dpg.add_file_extension(extension, color=(120, 220, 120, 255))
            dpg.add_file_extension(".*")

    def _save_config(self, path: str) -> None:
        """Save the config."""
        self.session.save_config(path)

    def _load_config(self, path: str) -> None:
        """Load the config and refresh the widgets that show it."""
        self.session.load_config(path)
        c = self.session.config
        for tag, value in (
            ("cfg_shape", c.shape.value),
            ("cfg_layout", c.layout.value),
            ("cfg_size", c.width),
            ("cfg_players", c.num_players),
            ("cfg_chaos", c.chaos),
            ("cfg_seed", -1 if c.seed is None else c.seed),
            ("cfg_days", c.max_days),
            ("cfg_tpd", c.ticks_per_day),
            ("cfg_thirst", c.start_thirst_min),
            ("cfg_hunger", c.start_hunger_min),
            ("cfg_health", c.start_health_min),
            ("cfg_sponsors", c.sponsors_enabled),
            ("cfg_gift", c.sponsor_gift_chance),
            ("cfg_gm", c.gamemaker_enabled),
            ("cfg_quiet", c.quiet_days_before_intervention),
            ("cfg_close", c.intervention_days),
            ("cfg_water_podiums", c.allow_water_podiums),
            ("cfg_vision", c.vision_radius),
            ("cfg_landmark", c.landmark_radius),
            ("cfg_thirst_days", c.thirst_days),
            ("cfg_hunger_days", c.hunger_days),
            ("cfg_brain", c.brain_name),
            ("cfg_cannon", c.cannon_and_sky),
            ("cfg_endgame", c.endgame_instinct),
        ):
            dpg.set_value(tag, value)
        dpg.set_value("nn_layer_count", len(c.neural.hidden_layers))
        self._rebuild_width_fields(list(c.neural.hidden_layers))
        dpg.set_value("nn_activation", c.neural.activation)
        dpg.set_value("nn_init", c.neural.initializer)
        dpg.set_value("nn_scale", c.neural.init_scale)
        dpg.set_value("nn_sparsity", c.neural.sparsity)
        self._on_apply_neural()
        self.session.apply_config_change("size")
        self._rebuild_roster_table()

    def _save_scenario(self, path: str) -> None:
        """Save the scenario."""
        self.session.save_scenario(path)

    def _load_scenario(self, path: str) -> None:
        """Load a scenario and refresh the roster table."""
        self.session.load_scenario(path)
        self._rebuild_roster_table()
        dpg.set_value("cfg_size", self.session.config.width)

    def _save_replay(self, path: str) -> None:
        """Save the replay."""
        self.session.save_replay(path)

    def _load_replay(self, path: str) -> None:
        """Load a replay and show its roster."""
        self.session.load_replay(path)
        self._rebuild_roster_table()

    def _export_gif(self, path: str) -> None:
        """Export the GIF."""
        self.session.export_gif(path, step=int(dpg.get_value("gif_step")))

    def _save_champion(self, path: str) -> None:
        """Save the champion."""
        self.session.save_champion(path)

    def _load_champion(self, path: str) -> None:
        """Load a champion into every tribute."""
        self.session.load_champion_into(path)
        self._rebuild_roster_table()


def launch() -> None:
    """Open the dashboard (blocks until it is closed)."""
    # Build and run.
    Dashboard().run()
