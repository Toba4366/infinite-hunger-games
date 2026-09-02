# Changelog

All notable changes to this project are recorded here, newest first. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Each entry was written from the working session that produced it; keep it
that way: whenever code, docs or defaults change, add a line under
**Unreleased** and move the block to a version heading when it ships.

## 0.4.0 - 2026-09-02 (tutorial and training feed)

Requested: a way to watch the training games and see the networks evolve,
an initial tutorial with images in the repo and in the dashboard, clearer
controls for the number of hidden layers and nodes, and a public GitHub
repository.

### Added
- Training feed: both trainers record one real game per step
  (`record_showcase`, on by default; `GenerationStats.showcase` and
  `EpochStats.showcase`). The dashboard's feed replays that game after every
  step, or gives the newest champion a live game so the Network tab shows
  real activations. `Session.load_recording`, `start_champion_game`,
  `genome_history` and `network_evolution` support it.
- Network tab: plots of the champion genome's change per step and a heat
  map of the genome by step, alongside the live node graph.
- Tutorial: a Tutorial tab (first in the dashboard) whose "Show me" buttons
  perform each step, and `docs/tutorial/README.md` with screenshots taken
  from the real dashboard by `hunger_games/ui/screenshots.py` (window-only
  capture on macOS).
- Brains tab: an explicit "number of hidden layers" control that reveals one
  "nodes in hidden layer N" field per layer, replacing the comma text box.
- `tests/test_feed.py`, `.gitignore` (the video transcript, run folders and
  big media stay out of the repository).

### Changed
- `load_replay` now goes through `load_recording`, so replays from any
  source rebuild the roster and config the same way.

## 0.3.0 - 2026-09-02 (research release)

Requested: a dashboard that supports research, not just play; graphs that
prove a network is learning the right behaviours; the ability to answer a
reviewer's questions and toggle the answers; well-formatted code; one PNG
per chart; a changelog; tributes that know who is left.

### Added
- `hunger_games/research/`: `telemetry.py` (actions against thirst, hunger,
  health, danger, field size and position; consumption timing; deaths by
  cause; survival after injury), `plots.py` (one PNG per chart: the chapter 3
  charts, death and position heatmaps, armed-versus-unarmed heatmaps, instinct
  curves, fight-or-flight by health, proximity against tributes remaining,
  action distribution over training, resource levels at death, training
  curves, timing, and a growing-curve GIF), `experiments.py` (parameter sweeps
  over any config field, dotted for nested ones, writing
  `results/<name>_<timestamp>/` with `config.json`, `results.csv`,
  `summary.json` and `plots/`).
- `hunger_games/training/reinforce.py`: REINFORCE with a learned value
  baseline in plain numpy, with a configurable `RewardConfig`, held-out
  validation seeds, and per-epoch policy loss, value loss, entropy, training
  and validation reward, survival, win and kill rates, timing and behaviour
  telemetry. `training/runs.py` writes any trainer's run folder.
- `hunger_games/brain/mlp.py`: a multi-layer perceptron with forward caching,
  backpropagation for every activation, an Adam optimiser, and the flat
  genome interface. `NeuralBrain` now wraps it and records its last chosen
  action and probabilities.
- `Game.decision_hooks` and `Game.tick_hooks` for research instrumentation.
- Field knowledge from the cannon and the nightly sky: tributes know how many
  remain, the field's mean and strongest training score, and their own rank
  (`cannon_and_sky`, on by default). The perception vector grew to 50 values
  and every slot is named in `VECTOR_NAMES`.
- `experiments/run_ga.py`, `run_rl.py`, `run_sweep.py`: command-line research
  scripts in the style of a course project repository.
- Dashboard: responsive three-panel layout with a dark theme and tooltips on
  every control; a working transport bar (Play starts a game if none exists);
  a brush ring on the arena while painting; podium presets (edge ring, around
  cornucopia, random, two sides); tributes moved off the void when the shape
  or map changes; settings that regenerate the arena as soon as they change;
  a Network tab drawing the neural network as a node graph with live
  activations for the selected tribute and the inputs and outputs listed; a
  Train tab with a genetic/reinforce toggle, the reward function, live
  performance, stability and timing plots, and the champion's genes with the
  ones that changed since the previous step highlighted; a Charts tab with
  live behaviour charts; a Research tab for sweeps and chart exports.
- `pyproject.toml` with a strict ruff configuration; the whole codebase is
  formatted and lint-clean (`ruff format` and `ruff check`).
- `CHANGELOG.md` (this file) and `tests/test_research.py` (finite-difference
  gradient checks for every activation, telemetry, RL, sweeps, plots).
- `runner.py` can collect telemetry (`telemetry.json`) and `analyze` writes
  individual PNGs under `output/plots/` as well as the combined `report.png`.

### Changed
- `max_days` default 18 to 24 (a strict cutoff, as requested).
- The endgame instinct became a toggle (`endgame_instinct`, off by default).
  Measured over 20 seeded games: with neither the instinct nor the circle, no
  game ends with a victor even in 24 days; the slow circle alone gives 19/20
  victors with 4% game maker deaths; the instinct alone gives 18/20 with none.
- The game maker circle is on by default but much slower (`intervention_days`
  6, up from 3), and tributes perceive it closing before it reaches them.
- `SimulationConfig.to_dict_raw` tolerates configs pickled by older versions.
- `analyze` now also writes one PNG per chart and a death heatmap.

### Fixed
- A dashboard launched before a config field was added crashed its worker
  processes; the config copy now fills missing fields with defaults.
- The training progress bar showed 0/0 until the first generation finished.
- `--gamemaker` and `--sponsors` are now on/off flag pairs
  (`--no-gamemaker`, `--no-sponsors`) whose defaults come from the config,
  so the command line no longer silently overrides a config default.
- Behaviour telemetry recorded placement 0 for survivors because its hook
  fires before the game assigns final placings; it now computes the shared
  survivor placing itself.
- The README's calibration table is measured with the shipped defaults; the
  200-game dataset in `output/` was regenerated with them (196/200 victors,
  74% player-versus-player, 23% natural, 3% game maker).

## 0.2.0 - 2026-09-02 (dashboard release)

Requested: a game makers' dashboard so casual users can paint maps, edit
tributes, place loot, drag podiums, watch games slowly, train and watch
models, and export GIFs; rarer medkits with sponsor-style healing; no
reliance on the game maker circle; podiums in water; district colours and
male/female markers; every classic neural initializer.

### Added
- `hunger_games/ui/`: Dear PyGui dashboard (`app.py`), GUI-free state
  (`session.py`), paintable map with presets including the 75th games'
  island (`painter.py`), arena drawing and mouse mapping (`canvas.py`).
- `scenario.py`: painted map, hand-placed loot and an editable roster saved as
  JSON; `Game` accepts a `Scenario`.
- `recorder.py`: tick-by-tick recordings, replay files, GIF export from
  recordings; `renderer.py` restructured around `ArenaFigure`.
- `sponsors.py`: parachutes for tributes in need, weighted by training score,
  career district (1, 2, 4) and kills, plus a game maker favour bonus.
- `training/genetic.py`: a genetic algorithm with elites, tournament
  selection, uniform crossover, Gaussian mutation and parallel evaluation.
- `brain/initializers.py`: zeros, ones, constant, uniform, normal, Xavier,
  He, LeCun (uniform and normal), orthogonal, identity, sparse; five
  activations. `NeuralConfig` sets hidden layers, activation and initializer.
- `districts.py`: industries, the requested district colours, female circles
  and male squares; tributes have a sex and default names like "D4 Female".
- `gifts.csv` in batch output; `sex`, `favor`, `gifts_received` in
  `players.csv`.

### Changed
- Medkits are rare in both layouts (5% of the Cornucopia pile, 3% of ring
  stacks); wounds below half health bleed and cannot be rested off.
- The game maker circle defaulted to off; when on, tributes see it closing.
- Podiums may stand in water (`allow_water_podiums`).
- Starting bars use `start_*_min` settings instead of the chaos dial.
- `max_days` 14 to 18; the command line reads every default from the config
  class; `--no-gamemaker` became opt-in `--gamemaker`; `--no-sponsors` added.
- The thirst instinct gained a critical-need bonus (chapter 4's "insist on
  drinking") after tributes were found dying a median 4 cells from water.

## 0.1.1 - 2026-09-02 (documentation release)

Requested: a README that lists every command and where every default came
from, and an in-depth Markdown page for every Python file including tests.

### Added
- `docs/` with one page per source file, a docs index, and `docs/output.md`.
- README command reference and tables of every default with its source.

### Fixed
- The game bookkeeping test only exercised its fallback branch on a draw.

## 0.1.0 - 2026-09-02 (initial build)

Requested: a Hunger Games simulator from chapter 4 of the "Infinite Hunger
Games" video, in beginner-friendly multi-file object-oriented Python with a
comment on every line, real Perlin noise, a brain layer deep enough to swap
in learning agents later, and a grid large enough to render.

### Added
- `config`, `noise` (Perlin with octaves and rank equalisation), `terrain`,
  `resources` (Cornucopia and ring layouts), `arena` (round or open field,
  breadth-first distance maps), `actions`, `perception`, `brain/` (interface,
  voting brain, random brain, neural brain), `player`, `gamemaker`,
  `records`, `game`, `runner` (parallel batches to CSV), `renderer`,
  `analysis` (the chapter 3 charts), a command line, and a test suite.
- Calibration against chapter 3 of the video: the Cornucopia layout
  reproduces the day-one bloodbath and the video's method split; the ring
  layout produces the flatter curve chapter 2 argues for.
