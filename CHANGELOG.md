# Changelog

All notable changes to this project are recorded here, newest first. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Each entry was written from the working session that produced it; keep it
that way: whenever code, docs or defaults change, add a line under
**Unreleased** and move the block to a version heading when it ships.

## Unreleased

### Fixed
- `pyproject.toml` declared version 0.3.0 while the changelog had shipped
  0.4.0 through 0.7.0; the package metadata now reads 0.7.0, so an install
  reports the same version the changelog does.
- The `docs/README.md` index called the perception vector 45 values. It has
  been 50 since the field-knowledge senses were added, which is what
  `perception.VECTOR_SIZE`, `docs/perception.md` and the README's dashboard
  table already say.

## 0.7.0 - 2026-09-03 (train until it wins)

Requested: promote the curriculum only when the learner actually wins,
train every brain until it consistently wins a majority of its games, prove
the README's claims, and test whether warm starts really end better.

### Added
- Game-level win rates everywhere: a game is won when a learner copy is the
  victor (with six copies only one can win, so per-copy rates were capped at
  one sixth). `IterationStats.val_win_rate`; every trainer's validation
  returns wins as well as scores.
- `CurriculumConfig.promote_on` ("win_rate" by default), `win_threshold`
  0.5, and `max_iterations_per_stage` 0 meaning no timeout.
- `ComparisonConfig.until_win_rate` / `win_window`: each variant trains until
  it wins a majority of validation games over the window at the final
  curriculum stage; `iterations_to_criterion` and `seconds_to_criterion` are
  recorded, and the report gains a criterion table and a warm-versus-cold
  table. `run_comparison.py --pairs --until-win --window`.
- NEAT genomes compile to an evaluation plan (about 30 microseconds per
  forward pass, from several milliseconds), making long NEAT runs feasible.
- `experiments/run_full.sh`: the full experiment script (methods cold and warm,
  sizes, initialisers), and `experiments/run_sensitivity.sh`: the cold-start
  sweep over learning rate, entropy bonus and episodes per epoch. Both live
  in `experiments/` because `results/` is ignored by git. The ignore pattern
  is now anchored (`/results/`) so that `docs/results/` is tracked.
- `docs/results/full_methods/paper.md`: the experiment written up as a
  research paper (abstract, background and prior work, methods, results
  with Wilson intervals, Fisher tests and trend slopes, discussion against
  the hypotheses and the source videos, limitations, next steps,
  references), with the champions' tournament games as GIFs.
- `docs/results/`: the report, numbers and charts of the full methods run
  (2026-09-03), and a "Claims and evidence" section in the README that
  points every claim at a measured number. Warm REINFORCE met the criterion
  in 62 iterations and won the tournament; no cold start met it in 1,150.
  The sizes run (16, 64x32, 128x64): imitation needs width to copy the
  teacher, PPO fine-tuning did best on the 16-node network (0.24). The
  initialisers run: zeros never learn (symmetry), Xavier beat He uniform
  after fine-tuning (0.17 against 0.05). The sensitivity sweep: cold starts
  want 16 episodes per epoch and a smaller entropy bonus; a learning rate of
  1e-2 sharpens into a poor policy.
- Watching experiments: the Train tab's "Load champion into the learner
  slots and watch" button plays a saved champion against voting opponents,
  and `run_comparison.py --save-replays-every N` keeps replays of training
  games for the Play tab.

### Fixed
- The genetic trainer now passes its validation win rate to the curriculum,
  so it can be promoted like the other methods (it previously stayed on the
  first stage forever under the win-based rule).

- The lesson curriculum. `Stage` (opponents, rules as config overrides,
  per-episode variants, a promotion metric and threshold),
  `CurriculumConfig.stages`, `CurriculumConfig.lessons()` (survive with no
  opponents and no circle, survive the rules, beat 1, 3, 7, 11 and 23,
  generalise across layouts, shapes and rules), `stage_config`,
  `episode_config` and `apply_overrides` in `training/common.py`;
  `Curriculum.observe` takes the survival share and judges each lesson on
  its own metric; every trainer rebuilds its config per lesson and per
  training episode. `run_comparison.py --curriculum lessons`,
  `Variant(curriculum="lessons")`. Tests in `tests/test_lessons.py`. The
  research guide records how the curriculum evolved from score-based
  promotion to wins to lessons. `experiments/run_lessons.sh` runs the next
  experiment on it: every method to graduation, then sizes for the best.
- `experiments/analyze_comparison.py`: Wilson intervals for tournament win
  rates, two-sided Fisher exact tests for warm-against-cold pairs and against
  imitation, regression slopes of survival, score, entropy and validation
  wins per variant, smoothed curves and an error-bar tournament chart, all
  written to `<run>/analysis/` with a `stats.md`. Tests in
  `tests/test_analysis_scripts.py`.
- `experiments/render_champions.py`: rebuilds every champion of a finished
  comparison and writes one tournament-game GIF per champion to
  `<run>/gifs/` with an index of who won and how long the copies survived.
- `run_comparison.py --set NAME=V1,V2,...` (repeatable): sweep one trainer
  setting, one variant per value for every method whose settings have that
  field, for the cold-start sensitivity question (learning rate, entropy
  bonus, episodes per epoch).
- An extension phase in the comparison: `ComparisonConfig.extended_iterations`
  and `extended_time_budget` (`run_comparison.py --extend-iterations`,
  `--extend-hours`). After every variant has had its first budget, those
  still short of the win criterion keep training with the same population or
  weights, so a slow cold start is measured for how long it really needs
  rather than cut off, and its final network still enters the tournament.
  The first-budget snapshot is kept under `runs_first_budget/`; the table
  gains `extended_iterations`; the report's criterion and warm-versus-cold
  tables show iterations trained, extended iterations and seconds. The full
  experiment script extends up to 1000 iterations or 2 hours per variant.

### Fixed
- Champions are chosen by curriculum stage first. REINFORCE and PPO kept
  the policy with the best validation return over the whole run, and the GA
  and NEAT the genome with the best training fitness, so a variant that
  climbed the ladder could send an easy-rung policy to the tournament.
  `champion_key(stage, val_win_rate, val_score)` in `training/common.py` now
  ranks candidates for all four; `GenerationStats` records `val_win_rate`
  and `stage`. Tests in `tests/test_champion.py`.
- The report's criterion table printed `70.0` and `nan` once any variant
  missed the criterion (pandas float column); it prints integers and dashes.
- A trainer's own clock kept running while it waited for the extension
  phase, stretching the time charts; the comparison now moves the trainer's
  start forward by the wait.
- The two win-rate charts plot a rolling mean over the criterion window
  instead of raw 0, 0.5 and 1 values.
- The win criterion counted the window that earned the final curriculum
  promotion as wins at the final stage, so a warm variant was declared
  "reached" the moment it was promoted to 23 opponents without playing a
  single iteration there. The window must now consist of iterations played
  at the final stage.

### Changed
- `MethodComparison.train_all` passes each iteration's `IterationStats` to the
  progress callback and `run` prints stage, opponents, validation win rate,
  mean score and seconds per iteration, plus a line when a variant reaches
  the win criterion, so a running experiment's log shows curriculum progress.

## 0.6.0 - 2026-09-02 (one learner, five methods, a research comparison)

Requested, after reading the transcripts of the zombie (PPO), Monopoly
(NEAT) and Battleship videos: train one network against the voting
strategy with a star on it; a dashboard like the zombie video's (score bars,
event monitor, average score, entropy and game-length graphs, learning
statistics with a rollout bar, pause, reset and watch buttons, CPU and
memory); a curriculum that grows the opposition; PPO and NEAT; imitation
from winning games; and research that compares training methods,
initialisations and network sizes, ending in a tournament.

### Added
- `training/common.py`: the shared `IterationStats` every method fills
  (scores, mean, best, entropy, mean game length, win rate, validation
  score, time, curriculum stage), an `EventLog`, a `Curriculum` (opponents
  1, 3, 7, 11, 23 with promotion on a score threshold or a timeout), a
  `SystemMonitor` (psutil), and `LearnerSpec` so worker processes can
  rebuild any learner brain.
- `training/ppo.py`: PPO with a clipped surrogate, several passes per
  batch and GAE, on top of the REINFORCE machinery.
- `brain/neat.py` and `training/neat.py`: NEAT genomes (node and
  connection genes, innovation numbers, feed-forward by depth), mutation,
  crossover, compatibility distance, species with fitness sharing,
  stagnation and an adaptive threshold; a `NeatBrain`; a champion file
  with `brain_name: neat`.
- Every trainer takes `curriculum=` and `initial_genome=`, exposes
  `learning_history`, `events`, `settings`, `step()`, `learner_spec()` and
  `champion_spec()`; the genetic trainer's default opponents are now the
  voting brain (`opponents="voting"`, scored by episode return) with
  `"self"` keeping the old tournament; imitation can learn from winners
  only (`winners_top`).
- `research/comparison.py` and `experiments/run_comparison.py`: train
  variants (methods, sizes, initializers, warm starts, curriculum) under one
  budget, run a 75-game tournament of the champions, and write results.csv,
  a LaTeX table, overlay learning curves, tournament charts, lines-of-code
  and training-time charts, and a generated report.md.
- Dashboard: the Train tab is now the training dashboard (method combo with
  help text, curriculum toggle, Start, Pause, Stop, Reset, Watch agent,
  score bars, event monitor, average score, entropy and game-length graphs,
  learning statistics with the rollout bar, CPU and memory); learners wear a
  gold star on the arena; the Network tab draws NEAT genomes as graphs.
- Run folders gain `learning.json` (the shared curves), `events.txt` and
  shared learning-curve PNGs.

### Changed
- `save_run` uses every trainer's `settings` and `save_champion`.
- The dashboard's default method is imitation; the tutorial's train step
  runs a short imitation with the live feed.

### Fixed
- `run_comparison.py --warm` with `--sizes` or `--initializers` now
  warm-starts each variant from the imitation variant with the same suffix
  (imitation variants are ordered first automatically); before, the lookup
  missed and every variant started cold.
- NEAT's stagnation rule now really protects the champion's species (the
  old check compared against a copy and never matched).
- PPO champion files are labelled `"method": "ppo"`.
- `create_brain("neat", ...)` builds a minimal random NEAT genome, so a
  roster tribute set to NEAT without a genome plays instead of crashing.

## 0.5.0 - 2026-09-02 (instincts by imitation)

Requested: neural tributes kept dying of thirst during training; instincts
should come from pretraining a network and using it as the initialisation,
not from dense rewards.

### Added
- `training/imitation.py`: behaviour cloning of the voting brain. Records
  (perception, action) pairs from teacher games (teacher at chaos 0 for clean
  labels), trains the network with cross-entropy and Adam, logs train and
  validation loss and accuracy, plays a greedy validation game per epoch
  (survival, win rate, telemetry, showcase), and saves a champion file.
- Warm starts: `GeneticTrainer` and `ReinforceTrainer` accept
  `initial_genome`; the GA seeds its population with the genome and close
  relatives (a quarter of the mutation scale). The dashboard's Train tab has
  an imitation method, a "start from the current champion" checkbox, and
  accuracy and loss plots for imitation.
- `NeuralBrain.action_to_menu_index`, the inverse of `menu_to_action`.
- `RewardConfig.approach`, a dense reward for closing in on water or grass
  when needed, present as a research toggle but off by default.
- `tests/test_imitation.py`.

### Changed
- `NeuralConfig.hidden_layers` default (16,) to (64, 32): measured 64%
  versus 80% agreement with the teacher after imitation.
- Every trainer exposes `settings` and `save_champion`, so run folders and
  the dashboard treat them alike.

### Fixed
- Run-folder names typed in the dashboard are reduced to a single folder
  component before use.

### Measured
- Untrained network, 8 generations or epochs: dehydration 10 of 12 learner
  deaths, flat fitness. After 30 imitation epochs: validation accuracy 80%,
  validation survival 162 versus 85 ticks, dehydration 2 of 12.

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

### Fixed
- The training feed counter is reset when a new training run starts, so a
  second run's feed shows its steps at once.
- The tutorial's train step no longer edits the Train tab's settings behind
  the widgets; it starts a fixed short evolution and shows the genetic group.
- The brush ring can be placed by the screenshot tool even when the mouse is
  elsewhere, so the paint picture shows the brush.
- `pyobjc-framework-Quartz` is listed as a macOS requirement (window-only
  screenshots need it).

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
