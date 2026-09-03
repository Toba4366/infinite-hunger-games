"""ui/session.py - everything the dashboard is doing, with no GUI code in it.

The `Session` owns the config, the painted map, the roster, the current
game and its recording, the playback clock, the selected tribute and the
trainer. The Dear PyGui code in app.py only reads and writes this object,
which keeps the GUI thin and lets the logic be tested without a window.
"""

# JSON for saving configs.
import json

# Background training.
import threading

# Pausing the training loop.
import time

# Filesystem paths.
from pathlib import Path

# numpy for genomes and grids.
import numpy as np

# The world, for podium presets on the painted map.
from hunger_games.arena import Arena

# The neural brain, for the network visualiser.
from hunger_games.brain.neural import MENU_NAMES, MENU_SIZE, NeuralBrain

# The settings.
from hunger_games.config import SimulationConfig

# Tribute names.
from hunger_games.districts import SEXES, default_tribute_name

# The game.
from hunger_games.game import Game

# Recording.
from hunger_games.recorder import Frame, Recorder, Recording

# GIF export.
from hunger_games.renderer import export_recording_gif

# Behaviour measurement, sweeps and plots.
from hunger_games.research.experiments import Sweep, SweepConfig
from hunger_games.research.plots import behaviour_plots
from hunger_games.research.telemetry import BehaviorTelemetry

# Supply kinds and layouts.
from hunger_games.resources import CornucopiaLayout, ResourceKind, RingLayout

# Custom setups.
from hunger_games.scenario import LootSpec, Scenario, TributeSpec

# Terrain kinds.
from hunger_games.terrain import TerrainType

# The trainers and run folders.
from hunger_games.training import (
    Curriculum,
    CurriculumConfig,
    GeneticTrainer,
    ImitationTrainer,
    NeatTrainer,
    PPOTrainer,
    ReinforceTrainer,
    SystemMonitor,
    save_run,
)

# The painter.
from hunger_games.ui.painter import MapPainter


class Session:
    """The dashboard's state and every operation the buttons perform."""

    def __init__(self, config: SimulationConfig | None = None) -> None:
        """Start with a generated arena and a generated roster."""
        # The settings.
        self.config = config if config is not None else SimulationConfig()
        # The painted map.
        self.painter = MapPainter(self.config.width, self.config.height)
        # The custom setup being edited.
        self.scenario = Scenario(title="Dashboard scenario")
        # The running game, if any.
        self.game: Game | None = None
        # Its recorder.
        self.recorder: Recorder | None = None
        # The recording being watched (live or loaded).
        self.recording: Recording | None = None
        # Which frame of the recording is on screen.
        self.playhead = 0
        # Whether playback is running.
        self.playing = False
        # Playback speed in simulation ticks per real second.
        self.ticks_per_second = 8.0
        # Fractional ticks carried between UI frames.
        self._accumulator = 0.0
        # The tribute the user clicked on.
        self.selected_id: int | None = None
        # The trainer (genetic or reinforce) and its thread.
        self.trainer: GeneticTrainer | ReinforceTrainer | None = None
        # Which method the trainer uses: "genetic" or "reinforce".
        self.training_method = "genetic"
        # Behaviour telemetry of the game being watched.
        self.telemetry: BehaviorTelemetry | None = None
        # Telemetry summaries of every finished game watched this session (for research charts).
        self.watched_summaries: list[dict] = []
        # Whether the current game's summary has been stored yet.
        self._summary_stored = False
        # The running sweep and its thread.
        self.sweep: Sweep | None = None
        # Its thread.
        self._sweep_thread: threading.Thread | None = None
        # Sweep progress: (values done, values total).
        self.sweep_progress = (0, 0)
        # The training feed: "off", "replay" (real evaluation games) or "live" (the new champion plays live).
        self.feed_mode = "off"
        # Pause flag for the training loop.
        self._paused = False
        # How many iterations the current run should do.
        self._max_iterations = 0
        # CPU and memory readings.
        self.system = SystemMonitor()
        # How many training steps the feed has already shown.
        self._feed_steps_seen = 0
        # A label for the headline while the feed is showing something.
        self.feed_label = ""
        # The thread running it.
        self._training_thread: threading.Thread | None = None
        # Progress of the current generation: (games done, games total).
        self.training_progress = (0, 0)
        # A short message for the status bar.
        self.status = "Ready"
        # Generate the first arena and roster.
        self.generate_arena()
        # Roster.
        self.generate_roster()

    # ------------------------------------------------------------- arena

    def generate_arena(self, seed: int | None = None) -> None:
        """Generate a Perlin arena from the config into the painter."""
        # Keep the painter's size in step with the config.
        if (self.painter.width, self.painter.height) != (self.config.width, self.config.height):
            self.painter.resize(self.config.width, self.config.height)
        # Load the preset.
        self.painter.apply_preset("perlin", self.config, seed)
        # Say so.
        self.status = "Generated a new arena"

    def apply_preset(self, name: str) -> None:
        """Load a named map preset into the painter."""
        # Delegate.
        self.painter.apply_preset(name, self.config)
        # Say so.
        self.status = f"Loaded preset '{name}'"

    def paint(self, x: int, y: int, terrain: TerrainType, radius: int) -> None:
        """Paint with the brush at a cell."""
        # Only on the grid.
        if self.painter.in_bounds(x, y):
            self.painter.paint(x, y, terrain, radius)

    def finish_painting(self) -> None:
        """Recompute heights after a brush stroke ends."""
        # Delegate.
        self.painter.finish()

    # ------------------------------------------------------------ roster

    def generate_roster(self) -> None:
        """Roll a fresh roster of tributes (and podiums) from the config."""
        # A throwaway game rolls names, scores, brains and podiums for us.
        game = Game(self.config, scenario=Scenario(terrain=self.painter.terrain.tolist()))
        # Turn each player into an editable spec.
        self.scenario.tributes = [
            TributeSpec(
                player_id=p.player_id,
                name=p.name,
                district=p.district,
                sex=p.sex,
                training_score=p.training_score,
                survival_score=p.survival_score,
                brain_name=p.brain.name,
                podium=(p.x, p.y),
            )
            for p in game.players
        ]
        # Nothing selected.
        self.selected_id = None
        # Say so.
        self.status = "Generated a new roster"

    @property
    def tributes(self) -> list[TributeSpec]:
        """The editable roster."""
        # Always a list.
        return self.scenario.tributes or []

    def tribute(self, player_id: int) -> TributeSpec | None:
        """One roster entry by id."""
        # Delegate.
        return self.scenario.tribute(player_id)

    def add_tribute(self) -> TributeSpec:
        """Append a new tribute to the roster."""
        # Next id.
        player_id = max((t.player_id for t in self.tributes), default=-1) + 1
        # Cycle districts and sexes like the generated roster.
        district = (player_id // 2) % 12 + 1
        # Sex.
        sex = SEXES[player_id % 2]
        # A middling tribute in the middle of the map.
        spec = TributeSpec(
            player_id,
            default_tribute_name(district, sex),
            district,
            sex,
            6,
            0.5,
            self.config.brain_name,
            podium=(self.painter.width // 2, self.painter.height // 2),
        )
        # Add.
        self.scenario.tributes = self.tributes + [spec]
        # Keep the config's count in step.
        self.config.num_players = len(self.scenario.tributes)
        # Done.
        return spec

    def remove_tribute(self, player_id: int) -> None:
        """Remove a tribute from the roster."""
        # Filter them out.
        self.scenario.tributes = [t for t in self.tributes if t.player_id != player_id]
        # Keep the config's count in step.
        self.config.num_players = max(2, len(self.scenario.tributes))
        # Deselect if needed.
        if self.selected_id == player_id:
            self.selected_id = None

    def move_tribute(self, player_id: int, x: int, y: int) -> None:
        """Put a tribute's podium at a cell (drag and drop)."""
        # The spec.
        spec = self.tribute(player_id)
        # Only real tributes on the grid.
        if spec is not None and self.painter.in_bounds(x, y) and self.painter.terrain[y, x] != int(TerrainType.VOID):
            # Move.
            spec.podium = (x, y)

    # Podium presets offered in the dashboard.
    PODIUM_PRESETS = ("edge ring", "around cornucopia", "random", "two sides")

    def _arena_for_podiums(self) -> Arena:
        """A throwaway arena built on the painted map, for snapping and layout podiums."""
        # Build with the current config on the painted terrain.
        return Arena(self.config, np.random.default_rng(0), terrain=self.painter.terrain)

    def arrange_podiums(self, preset: str) -> None:
        """Place every podium by a named preset, spreading the strongest tributes apart."""
        # The arena.
        arena = self._arena_for_podiums()
        # How many.
        count = len(self.tributes)
        # Nothing to place.
        if count == 0:
            return
        # The positions.
        if preset == "edge ring":
            positions = RingLayout().spawn_positions(arena, count)
        elif preset == "around cornucopia":
            positions = CornucopiaLayout().spawn_positions(arena, count)
        elif preset == "random":
            rng = np.random.default_rng()
            positions = [
                arena.snap_to_podium(int(rng.integers(arena.width)), int(rng.integers(arena.height)))
                for _ in range(count)
            ]
        elif preset == "two sides":
            # Half along the left edge, half along the right, like two opposing camps.
            positions = []
            for index in range(count):
                x = 3 if index % 2 == 0 else arena.width - 4
                y = int(3 + (arena.height - 6) * (index // 2) / max(1, (count + 1) // 2 - 1))
                positions.append(arena.snap_to_podium(x, y))
        else:
            raise KeyError(f"Unknown podium preset '{preset}'. Choose from: {', '.join(self.PODIUM_PRESETS)}")
        # Strongest spread apart, as chapter 2 suggests.
        ranked = sorted(self.tributes, key=lambda t: t.training_score, reverse=True)
        top = ranked[: max(1, count // 3)]
        rest = ranked[len(top) :]
        stride = count / len(top)
        slots: list = [None] * count
        for index, spec in enumerate(top):
            slots[int(index * stride)] = spec
        remaining = iter(rest)
        for index in range(count):
            if slots[index] is None:
                slots[index] = next(remaining)
        # Assign.
        for spec, position in zip(slots, positions, strict=False):
            spec.podium = position
        # Say so.
        self.status = f"Podiums arranged: {preset}"

    def reposition_off_void(self) -> None:
        """Nudge every podium onto a legal cell (after painting or changing the arena shape)."""
        # The arena.
        arena = self._arena_for_podiums()
        # Each tribute.
        for spec in self.tributes:
            if spec.podium is None or not arena.is_walkable(*spec.podium):
                spec.podium = arena.snap_to_podium(*(spec.podium or (arena.center_x, arena.center_y)))

    def apply_config_change(self, what: str) -> None:
        """React to a setting change so the arena on screen always matches: regenerate, re-place, or both."""
        # Size or shape: a new map and podiums nudged onto it.
        if what in ("size", "shape"):
            self.generate_arena()
            self.reposition_off_void()
        # Layout: podiums follow the layout.
        elif what == "layout":
            self.arrange_podiums("edge ring" if self.config.layout.value == "ring" else "around cornucopia")
        # Player count: regenerate the roster.
        elif what == "players":
            self.generate_roster()

    def tribute_at(self, x: int, y: int, radius: float = 1.5) -> int | None:
        """The id of the tribute whose podium (or live position) is nearest to a cell, if close enough."""
        # Positions come from the current frame when a game is loaded, else from the roster.
        positions = self.positions()
        # Best match so far.
        best, best_distance = None, radius
        # Check each.
        for player_id, (px, py) in positions.items():
            # Distance.
            distance = max(abs(px - x), abs(py - y))
            # Closer than the best so far.
            if distance <= best_distance:
                best, best_distance = player_id, distance
        # Done.
        return best

    def positions(self) -> dict[int, tuple[int, int]]:
        """Where every tribute is: the current frame if watching a game, else their podiums."""
        # From the frame on screen.
        frame = self.current_frame
        # Watching.
        if frame is not None:
            return {p.player_id: (p.x, p.y) for p in frame.players if p.alive}
        # Editing.
        return {t.player_id: t.podium for t in self.tributes if t.podium is not None}

    # -------------------------------------------------------------- loot

    def place_loot(self, x: int, y: int, kind: ResourceKind, quantity: int, quality: float) -> None:
        """Put a stack of supplies at a cell (replacing any hand-placed stack there)."""
        # Only on the grid and inside the arena.
        if not self.painter.in_bounds(x, y) or self.painter.terrain[y, x] == int(TerrainType.VOID):
            return
        # Remove any existing stack at that cell.
        self.remove_loot(x, y)
        # Add the new one.
        self.scenario.loot.append(LootSpec(x, y, int(kind), quantity, quality))

    def remove_loot(self, x: int, y: int) -> None:
        """Remove hand-placed supplies at a cell."""
        # Filter.
        self.scenario.loot = [l for l in self.scenario.loot if (l.x, l.y) != (x, y)]

    def clear_loot(self) -> None:
        """Remove all hand-placed supplies."""
        # Empty the list.
        self.scenario.loot = []

    # -------------------------------------------------------------- game

    def _scenario_for_game(self) -> Scenario:
        """The scenario as the game will see it: the painted map plus the edited roster and loot."""
        # Copy so the game's scenario is frozen at start time.
        return Scenario(
            terrain=self.painter.terrain.tolist(),
            use_layout_loot=self.scenario.use_layout_loot,
            loot=list(self.scenario.loot),
            tributes=[TributeSpec(**vars(t)) for t in self.tributes] if self.tributes else None,
            title=self.scenario.title,
        )

    def new_game(self, seed: int | None = None) -> Game:
        """Start a fresh game from the current config, map and roster, and begin recording it."""
        # Keep the player count in step with the roster.
        if self.tributes:
            self.config.num_players = len(self.tributes)
        # A config copy with the requested seed.
        config = SimulationConfig(
            **{**self.config.to_dict_raw(), "seed": seed if seed is not None else self.config.seed}
        )
        # Build the game.
        self.game = Game(config, scenario=self._scenario_for_game())
        # Measure its behaviour.
        self.telemetry = BehaviorTelemetry(self.game.arena.width, self.game.arena.height).attach(self.game)
        # Not stored yet.
        self._summary_stored = False
        # Record it from tick 0.
        self.recorder = Recorder(self.game)
        # Watch that recording.
        self.recording = self.recorder.recording
        # Rewind.
        self.playhead = 0
        # Stop.
        self.playing = False
        # Reset the clock.
        self._accumulator = 0.0
        # Say so.
        self.status = f"New game, seed {self.game.seed}"
        # Done.
        return self.game

    @property
    def current_frame(self) -> Frame | None:
        """The frame at the playhead, if there is a recording."""
        # No recording.
        if self.recording is None or not self.recording.frames:
            return None
        # Clamp the playhead and fetch.
        self.playhead = max(0, min(self.playhead, self.recording.length - 1))
        # The frame.
        return self.recording.frames[self.playhead]

    @property
    def at_live_edge(self) -> bool:
        """Is the playhead on the newest frame of a game that is still running?"""
        # Need a running game.
        return self.recording is not None and self.playhead >= self.recording.length - 1

    def step_once(self) -> None:
        """Advance one tick: play a new tick if at the live edge, else move the playhead forward."""
        # Nothing loaded.
        if self.recording is None:
            return
        # At the edge of a running game: simulate a tick.
        if self.at_live_edge:
            # Only if the game is still going.
            if self.recorder is not None and self.game is not None and not self.game.is_over:
                self.recorder.step()
                # Follow it.
                self.playhead = self.recording.length - 1
                # Store the behaviour summary when the game ends.
                self._store_summary()
            # Otherwise stop playing.
            else:
                self.playing = False
        # Otherwise scrub forward.
        else:
            self.playhead += 1

    # Feed modes offered in the dashboard.
    FEED_MODES = ("off", "replay", "live")

    def _feed_ready_for_next(self) -> bool:
        """Is the arena free to show the next training step (nothing loaded, or the current game finished)?"""
        # Nothing loaded.
        if self.recording is None:
            return True
        # A replay: finished when the playhead reached its end.
        if self.game is None:
            return self.playhead >= self.recording.length - 1
        # A live game: finished when it is over and fully watched.
        return self.game.is_over and self.at_live_edge

    def _advance_feed(self) -> None:
        """Show the newest training step when the feed is on and the arena is free."""
        # Off, or nothing new.
        if self.feed_mode == "off" or self.trainer is None:
            return
        # The steps so far.
        history = self.trainer.history
        # Nothing new, or still busy.
        if len(history) <= self._feed_steps_seen or not self._feed_ready_for_next():
            return
        # Catch up to the newest step.
        self._feed_steps_seen = len(history)
        # The newest step's label.
        step_name = "generation" if self.training_method == "genetic" else "epoch"
        # Replay a real training game.
        if self.feed_mode == "replay":
            showcase = history[-1].showcase
            # None recorded (record_showcase off): nothing to show.
            if showcase is None:
                self.feed_label = f"training feed: {step_name} {len(history) - 1} has no recording"
                return
            self.load_recording(showcase)
            self.feed_label = f"training feed: replaying a real {step_name} {len(history) - 1} game"
        # Or let the newest champion play live so the Network tab shows real activations.
        else:
            self.start_champion_game(all_slots=False)
            self.feed_label = f"training feed: {step_name} {len(history) - 1} champion playing live"
        # Play at the current speed.
        self.playing = True

    def load_recording(self, recording: Recording) -> None:
        """Watch a recording object (the live game, if any, is dropped)."""
        # Adopt it.
        self.recording = recording
        # No live game.
        self.game, self.recorder, self.telemetry = None, None, None
        # Show the map it was played on.
        self.painter.load(recording.terrain, recording.heights)
        # Adopt the settings it was played with.
        self.config = recording.config
        # Rebuild the roster from the recording so names, districts and sexes match the dots.
        first = recording.frames[0].players if recording.frames else []
        starts = {p.player_id: (p.x, p.y) for p in first}
        self.scenario.tributes = [
            TributeSpec(
                e.player_id,
                e.name,
                e.district,
                e.sex,
                e.training_score,
                e.survival_score,
                e.brain,
                podium=starts.get(e.player_id),
            )
            for e in recording.roster
        ]
        # Nothing selected.
        self.selected_id = None
        # Rewind and stop.
        self.playhead = 0
        self.playing = False

    def start_champion_game(self, all_slots: bool = True) -> bool:
        """Give the champion to every tribute (or only the trainer's learner slots) and start a game."""
        # Need a champion.
        if self.trainer is None or self.trainer.champion is None:
            self.status = "No champion yet: train first"
            return False
        # Who gets it.
        ids = None if all_slots else self.trainer._learner_ids()
        # Give it.
        self.give_champion(ids)
        # A fresh game.
        self.new_game()
        # Play.
        self.playing = True
        # Done.
        return True

    def genome_history(self) -> list[np.ndarray]:
        """The learner after every training step so far, as flat vectors (NEAT: its connection weights)."""
        # Nothing.
        if self.trainer is None:
            return []
        vectors = []
        for stats in self.trainer.learning_history:
            learner = stats.learner
            vectors.append(
                np.asarray([c[3] for c in learner["connections"]], dtype=float)
                if isinstance(learner, dict)
                else np.asarray(learner, dtype=float)
            )
        # Only steps of equal length can be stacked (NEAT genomes grow).
        size = vectors[-1].size if vectors else 0
        return [v for v in vectors if v.size == size]

    def network_evolution(self, max_genes: int = 200) -> dict | None:
        """How the champion genome changed over training: change size per step and a genes-by-steps matrix."""
        # The genomes.
        genomes = self.genome_history()
        # Nothing yet.
        if not genomes:
            return None
        # Stack.
        matrix = np.stack(genomes)
        # Change from the previous step (zero for the first).
        change = [0.0] + [float(np.linalg.norm(matrix[i] - matrix[i - 1])) for i in range(1, len(matrix))]
        # Mean absolute value per step.
        mean_abs = np.abs(matrix).mean(axis=1).tolist()
        # Bundle.
        return {
            "steps": list(range(len(genomes))),
            "change": change,
            "mean_abs": mean_abs,
            "genes": matrix[:, :max_genes],
            "gene_count": matrix.shape[1],
        }

    def update(self, seconds: float) -> None:
        """Called every UI frame: show the next training-feed step if due, then advance playback."""
        # The training feed.
        self._advance_feed()
        # Only while playing.
        if not self.playing:
            return
        # Accumulate fractional ticks.
        self._accumulator += seconds * self.ticks_per_second
        # Cap how many ticks one UI frame may run so the window stays responsive.
        budget = 50
        # Run whole ticks.
        while self._accumulator >= 1.0 and budget > 0:
            # One tick.
            self.step_once()
            # Spend it.
            self._accumulator -= 1.0
            # Count it.
            budget -= 1

    def run_to_end(self) -> None:
        """Simulate the rest of the game instantly."""
        # Need a running game.
        if self.recorder is not None and self.game is not None:
            self.recorder.record_all()
            # Show the last frame.
            self.playhead = self.recording.length - 1
            # Stop.
            self.playing = False
            # Store the behaviour summary.
            self._store_summary()
            # Say so.
            self.status = "Game finished"

    def _store_summary(self) -> None:
        """Keep the finished game's telemetry summary once."""
        # Only once, and only when over.
        if self.game is not None and self.game.is_over and self.telemetry is not None and not self._summary_stored:
            self.watched_summaries.append(self.telemetry.summary())
            self._summary_stored = True

    def watched_summary(self) -> dict | None:
        """The merged behaviour of every game watched this session, plus the current one."""
        # Current game's tallies so far.
        current = [self.telemetry.summary()] if self.telemetry is not None and not self._summary_stored else []
        # Everything.
        summaries = self.watched_summaries + current
        # Merge.
        return BehaviorTelemetry.merge(summaries) if summaries else None

    def export_behaviour_plots(self, folder: str | Path) -> int:
        """Write every behaviour chart for the games watched this session; returns how many files."""
        # The data.
        summary = self.watched_summary()
        # Nothing yet.
        if summary is None:
            self.status = "No games watched yet"
            return 0
        # Write.
        written = behaviour_plots(summary, folder)
        # Say so.
        self.status = f"Wrote {len(written)} charts to {folder}"
        # Done.
        return len(written)

    def network_snapshot(self, player_id: int | None) -> dict | None:
        """What the selected tribute's neural network is doing right now, for the visualiser:
        layer sizes, the activation of every node, every weight matrix, the input values,
        and the output probabilities. None if there is no live neural tribute selected.
        """
        # Need a live game and a selection.
        if self.game is None or player_id is None or player_id not in self.game.player_by_id:
            return None
        # The tribute.
        player = self.game.player_by_id[player_id]
        # NEAT brains are drawn as a graph.
        from hunger_games.brain.neat import NeatBrain

        if isinstance(player.brain, NeatBrain) and player.last_perception is not None:
            genome = player.brain.genome_data
            inputs = player.last_perception.to_vector()
            values = genome.activations(inputs)
            depth = genome.depths()
            logits = genome.forward(inputs)
            probabilities = (
                player.brain.probabilities_of(logits)
                if hasattr(player.brain, "probabilities_of")
                else (
                    player.brain.last_probabilities
                    if player.brain.last_probabilities is not None
                    else np.ones(MENU_SIZE) / MENU_SIZE
                )
            )
            return {
                "graph": True,
                "nodes": [(n.id, n.kind, depth[n.id], values[n.id]) for n in genome.nodes],
                "edges": [(c.src, c.dst, c.weight) for c in genome.connections if c.enabled],
                "probabilities": probabilities,
                "chosen": player.brain.last_index,
                "menu": MENU_NAMES,
            }
        # Must be neural with a perception to feed.
        if not isinstance(player.brain, NeuralBrain) or player.last_perception is None:
            return None
        # The input vector.
        inputs = player.last_perception.to_vector()
        # Every layer's activation.
        activations = player.brain.network.hidden_activations(inputs)
        # Probabilities.
        probabilities = player.brain.probabilities(activations[-1])
        # Bundle.
        return {
            "layer_sizes": player.brain.layer_sizes,
            "inputs": inputs,
            "activations": activations,
            "weights": player.brain.network.weights,
            "probabilities": probabilities,
            "chosen": player.brain.last_index,
            "menu": MENU_NAMES,
        }

    def seek(self, frame_index: int) -> None:
        """Jump the playhead."""
        # Clamp inside the recording.
        if self.recording is not None:
            self.playhead = max(0, min(frame_index, self.recording.length - 1))

    def rewind(self) -> None:
        """Back to frame 0."""
        # Seek.
        self.seek(0)

    def event_log(self, last: int = 12) -> list[str]:
        """The most recent eliminations and gifts up to the playhead, newest last."""
        # Nothing loaded.
        if self.recording is None:
            return []
        # Collect lines.
        lines = []
        # Walk frames up to the playhead.
        for frame in self.recording.frames[: self.playhead + 1]:
            # Eliminations.
            for e in frame.eliminations:
                # Who did it.
                by = f" by {e.killer_name}" if e.killer_name else ""
                # The line.
                lines.append(f"Day {e.day}: {e.victim_name} out ({e.weapon}{by}), placed {e.placement}")
            # Gifts.
            for g in frame.gifts:
                lines.append(f"Day {g.day}: parachute for {g.player_name} ({g.kind})")
        # The tail.
        return lines[-last:]

    # ------------------------------------------------------------- files

    def save_scenario(self, path: str | Path) -> None:
        """Save the map, loot and roster."""
        # Include the painted terrain.
        self._scenario_for_game().save(path)
        # Say so.
        self.status = f"Saved scenario to {path}"

    def load_scenario(self, path: str | Path) -> None:
        """Load a map, loot and roster."""
        # Read.
        scenario = Scenario.load(path)
        # Adopt the map if there is one.
        if scenario.terrain is not None:
            self.painter.load(np.array(scenario.terrain, dtype=np.int8))
            # Keep the config's size in step.
            self.config.width, self.config.height = self.painter.width, self.painter.height
        # Adopt the rest.
        self.scenario = scenario
        # Player count.
        if scenario.tributes:
            self.config.num_players = len(scenario.tributes)
        # Nothing selected.
        self.selected_id = None
        # Say so.
        self.status = f"Loaded scenario from {path}"

    def save_config(self, path: str | Path) -> None:
        """Save the settings as JSON."""
        # Write.
        Path(path).write_text(json.dumps(self.config.to_dict(), indent=2))
        # Say so.
        self.status = f"Saved config to {path}"

    def load_config(self, path: str | Path) -> None:
        """Load settings from JSON."""
        # Read.
        self.config = SimulationConfig.from_dict(json.loads(Path(path).read_text()))
        # Say so.
        self.status = f"Loaded config from {path}"

    def save_replay(self, path: str | Path) -> None:
        """Save the current recording."""
        # Need one.
        if self.recording is not None:
            self.recording.save(path)
            # Say so.
            self.status = f"Saved replay to {path}"

    def load_replay(self, path: str | Path) -> None:
        """Load a recording from disk to watch (the live game, if any, is dropped)."""
        # Read and adopt.
        self.load_recording(Recording.load(path))
        # Say so.
        self.status = f"Loaded replay from {path}"

    def export_gif(self, path: str | Path, step: int = 2, fps: int = 15) -> None:
        """Write the current recording to a GIF (or MP4)."""
        # Need one.
        if self.recording is None:
            return
        # Finish the game first so the GIF has an ending.
        self.run_to_end()
        # Export.
        export_recording_gif(self.recording, path, fps=fps, step=step)
        # Say so.
        self.status = f"Saved {path}"

    # ---------------------------------------------------------- training

    @property
    def training_running(self) -> bool:
        """Is a trainer thread alive?"""
        # Check the thread.
        return self._training_thread is not None and self._training_thread.is_alive()

    # The learner brain kinds (tributes with these brains get a star on the arena).
    LEARNER_KINDS = ("neural", "neat")

    def warm_start_genome(self):
        """A genome to start the next run from: the current champion, else a trained roster genome."""
        # The current trainer's champion (a flat array, or a NEAT genome dictionary).
        if self.trainer is not None and self.trainer.champion is not None:
            champion = self.trainer.champion
            return champion if isinstance(champion, dict) else np.asarray(champion, dtype=float)
        # Else the first roster tribute carrying a learner genome (for example a loaded champion file).
        for spec in self.tributes:
            if spec.genome is not None and spec.brain_name in self.LEARNER_KINDS:
                return spec.genome if isinstance(spec.genome, dict) else np.asarray(spec.genome, dtype=float)
        # Nothing to start from.
        return None

    def start_training(
        self,
        settings,
        method: str = "genetic",
        warm_start: bool = False,
        curriculum: CurriculumConfig | None = None,
    ) -> None:
        """Start a trainer (imitation, genetic, neat, reinforce or ppo) in a background thread on the painted map."""
        # One at a time.
        if self.training_running:
            return
        # A genome to start from, if asked and available; only a matching kind fits.
        initial = self.warm_start_genome() if warm_start else None
        if initial is not None:
            wants_neat = method == "neat"
            if isinstance(initial, dict) != wants_neat:
                initial = None
            if method == "genetic" and getattr(settings, "brain_name", "neural") != "neural":
                initial = None
        # A config copy with the roster's player count.
        config = SimulationConfig(
            **{**self.config.to_dict_raw(), "num_players": max(2, len(self.tributes) or self.config.num_players)}
        )
        # The map.
        scenario = Scenario(terrain=self.painter.terrain.tolist())
        # The curriculum.
        curriculum_object = Curriculum(curriculum) if curriculum is not None and curriculum.enabled else None
        # The trainer.
        self.training_method = method
        self._feed_steps_seen = 0
        self.feed_label = ""
        self._paused = False
        builders = {
            "imitation": ImitationTrainer,
            "genetic": GeneticTrainer,
            "neat": NeatTrainer,
            "reinforce": ReinforceTrainer,
            "ppo": PPOTrainer,
        }
        self.trainer = builders[method](
            config, settings, scenario=scenario, initial_genome=initial, curriculum=curriculum_object
        )
        # How many iterations to run.
        self._max_iterations = int(getattr(settings, "epochs", getattr(settings, "generations", 0)))

        # Progress callback.
        def progress(done: int, total: int) -> None:
            self.training_progress = (done, total)

        # The loop lives here so it can pause.
        def body() -> None:
            try:
                self.trainer._stop = False
                while len(self.trainer.learning_history) < self._max_iterations and not self.trainer._stop:
                    if self._paused:
                        time.sleep(0.1)
                        continue
                    self.trainer.step(progress)
                self.status = "Training stopped" if self.trainer._stop else "Training finished"
            except Exception as error:  # noqa: BLE001 - surface any failure in the status bar
                self.status = f"Training error: {error}"

        # Start.
        self._training_thread = threading.Thread(target=body, daemon=True)
        self._training_thread.start()
        # Say so.
        self.status = f"Training ({method}{', warm start' if initial is not None else ''}{', curriculum' if curriculum_object else ''})..."

    def pause_training(self, paused: bool = True) -> None:
        """Pause or resume the loop between iterations."""
        # Flag.
        self._paused = paused
        self.status = "Training paused" if paused else "Training resumed"

    @property
    def training_paused(self) -> bool:
        """Is the loop paused?"""
        # Flag.
        return self._paused

    def reset_training(self) -> None:
        """Stop and forget the trainer, its history and the feed."""
        # Stop.
        self.stop_training()
        self._paused = False
        if self._training_thread is not None:
            self._training_thread.join(timeout=5.0)
        # Forget.
        self.trainer = None
        self.training_progress = (0, 0)
        self._feed_steps_seen = 0
        self.feed_label = ""
        self.status = "Training reset"

    def training_events(self, count: int = 14) -> list[str]:
        """The most recent training events."""
        # From the trainer's log.
        return self.trainer.events.tail(count) if self.trainer is not None else []

    def learning_stats(self) -> dict:
        """The learning statistics panel: iteration, seed, seconds per iteration, best score, total time, stage."""
        # Nothing yet.
        history = self.trainer.learning_history if self.trainer is not None else []
        if not history:
            return {
                "iteration": 0,
                "seed": self.config.seed,
                "seconds_per_iteration": 0.0,
                "max_score": 0.0,
                "learning_time": 0.0,
                "stage": 0,
                "opponents": 0,
                "mean_score": 0.0,
                "entropy": 0.0,
                "mean_length": 0.0,
            }
        last = history[-1]
        return {
            "iteration": last.iteration + 1,
            "seed": getattr(self.trainer.settings, "seed", None),
            "seconds_per_iteration": float(np.mean([s.seconds for s in history[-5:]])),
            "max_score": float(max(s.best_score for s in history)),
            "learning_time": last.cumulative_seconds,
            "stage": last.stage,
            "opponents": last.opponents,
            "mean_score": last.mean_score,
            "entropy": last.entropy,
            "mean_length": last.mean_length,
        }

    def learner_ids_on_screen(self) -> set[int]:
        """Tributes on the arena driven by a learner brain (they get a star)."""
        # From the recording's roster when watching one, else the editable roster.
        if self.recording is not None and self.game is None:
            return {e.player_id for e in self.recording.roster if e.brain in self.LEARNER_KINDS}
        return {t.player_id for t in self.tributes if t.brain_name in self.LEARNER_KINDS and t.genome is not None}

    def stop_training(self) -> None:
        """Ask the trainer to stop after the current generation."""
        # Delegate.
        if self.trainer is not None:
            self.trainer.stop()

    def training_history(self) -> list:
        """Stats objects so far (GenerationStats or EpochStats)."""
        # Empty if no trainer.
        return list(self.trainer.history) if self.trainer is not None else []

    def training_rows(self) -> list[dict]:
        """The unified learning history as plain rows (the same shape for every method)."""
        # Empty if no trainer.
        return [s.to_row() for s in self.trainer.learning_history] if self.trainer is not None else []

    def latest_scores(self) -> list[float]:
        """The scores of the newest iteration's episodes (the dashboard's score bars)."""
        # Empty if none.
        history = self.trainer.learning_history if self.trainer is not None else []
        return list(history[-1].scores) if history else []

    def champion_genes(self) -> tuple[np.ndarray, np.ndarray] | None:
        """The latest champion genome and a mask of which genes changed since the previous champion."""
        # Need history.
        if self.trainer is None or not self.trainer.history:
            return None

        # The learner after each iteration (a flat array, or a NEAT genome dictionary reduced to its weights).
        def as_vector(learner):
            if isinstance(learner, dict):
                return np.asarray([c[3] for c in learner["connections"]], dtype=float)
            return np.asarray(learner, dtype=float)

        history = self.trainer.learning_history
        latest = as_vector(history[-1].learner)
        previous = as_vector(history[-2].learner) if len(history) >= 2 else None
        if previous is not None and previous.size != latest.size:
            previous = None
        # Changed mask.
        changed = np.abs(latest - previous) > 1e-9 if previous is not None else np.ones(latest.size, dtype=bool)
        # Done.
        return latest, changed

    def save_training_run(self, name: str, results_dir: str | Path = "results") -> Path | None:
        """Write the trainer's config, history, champion and every chart to a run folder."""
        # Need history.
        if self.trainer is None or not self.trainer.history:
            self.status = "Nothing trained yet"
            return None
        # Save.
        folder = save_run(self.trainer, self.training_method, name, results_dir)
        # Say so.
        self.status = f"Saved run to {folder}"
        # Done.
        return folder

    # ------------------------------------------------------------- sweeps

    @property
    def sweep_running(self) -> bool:
        """Is a sweep thread alive?"""
        # Check.
        return self._sweep_thread is not None and self._sweep_thread.is_alive()

    def start_sweep(self, settings: SweepConfig) -> None:
        """Run a parameter sweep on the painted map in a background thread."""
        # One at a time.
        if self.sweep_running:
            return
        # The sweep.
        self.sweep = Sweep(self.config, settings, scenario=Scenario(terrain=self.painter.terrain.tolist()))
        # Progress.
        self.sweep_progress = (0, len(settings.values))

        # Body.
        def body() -> None:
            try:
                self.sweep.run(on_progress=lambda done, total: setattr(self, "sweep_progress", (done, total)))
                self.status = f"Sweep saved to {self.sweep.run_dir}"
            except Exception as error:  # noqa: BLE001
                self.status = f"Sweep error: {error}"

        # Start.
        self._sweep_thread = threading.Thread(target=body, daemon=True)
        self._sweep_thread.start()
        # Say so.
        self.status = "Sweep running..."

    def stop_sweep(self) -> None:
        """Ask the sweep to stop after the current value."""
        # Delegate.
        if self.sweep is not None:
            self.sweep.stop()

    def give_champion(self, player_ids: list[int] | None = None) -> int:
        """Load the champion genome into some tributes (all by default). Returns how many."""
        # Need a champion.
        if self.trainer is None or self.trainer.champion is None:
            return 0
        # Which tributes.
        targets = [t for t in self.tributes if player_ids is None or t.player_id in player_ids]
        # The trained brain kind.
        if self.training_method == "neat":
            brain_name = "neat"
        elif self.training_method == "genetic":
            brain_name = self.trainer.training.brain_name
        else:
            brain_name = "neural"
        champion = self.trainer.champion
        # Give it to each.
        for spec in targets:
            # The kind.
            spec.brain_name = brain_name
            # The genome (a dictionary for NEAT, a list otherwise).
            spec.genome = champion if isinstance(champion, dict) else np.asarray(champion).tolist()
        # Neural champions need the architecture they were trained with.
        self.config.neural = self.trainer.config.neural
        # Say so.
        self.status = f"Gave the champion brain to {len(targets)} tribute(s)"
        # Done.
        return len(targets)

    def save_champion(self, path: str | Path) -> None:
        """Save the champion genome."""
        # Every trainer writes the same file shape.
        if self.trainer is not None and self.trainer.champion is not None:
            self.trainer.save_champion(path)
            # Say so.
            self.status = f"Saved champion to {path}"

    def load_champion_into(self, path: str | Path, player_ids: list[int] | None = None) -> int:
        """Load a saved champion file into some tributes (all by default)."""
        # Read.
        data = GeneticTrainer.load_champion(path)
        # Adopt the architecture when it is a neural champion.
        if data.get("neural") is not None:
            self.config.neural = data["neural"]
        # Which tributes.
        targets = [t for t in self.tributes if player_ids is None or t.player_id in player_ids]
        # Give it to each.
        for spec in targets:
            spec.brain_name = data["brain_name"]
            spec.genome = data["genome"] if isinstance(data["genome"], dict) else np.asarray(data["genome"]).tolist()
        # Say so.
        self.status = f"Loaded champion into {len(targets)} tribute(s)"
        # Done.
        return len(targets)
