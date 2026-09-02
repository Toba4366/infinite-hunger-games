# Documentation

One page per Python file. Each page follows the same layout: purpose,
the concepts a beginner needs, a walkthrough of every class and method,
how to use or extend it, and gotchas. Start with the dashboard guide if
you want to click before you read.

- [tutorial/README.md](tutorial/README.md): the illustrated walkthrough, from first launch to a trained brain.
- [ui/README.md](ui/README.md): using the game makers' dashboard.
- [research/README.md](research/README.md): the research toolkit and which chart answers which question.
- [output.md](output.md): the result files.
- [../CHANGELOG.md](../CHANGELOG.md): what changed and why, release by release.

## Package: `hunger_games/`

| Page | Source file | One line |
| --- | --- | --- |
| [config.md](config.md) | `config.py` | Every setting, with defaults |
| [noise.md](noise.md) | `noise.py` | Perlin noise height maps |
| [terrain.md](terrain.md) | `terrain.py` | Heights to terrain types, hunt and move tables |
| [districts.md](districts.md) | `districts.py` | District names, colours, sexes |
| [resources.md](resources.md) | `resources.py` | Supplies, layouts, weapons |
| [arena.md](arena.md) | `arena.py` | The world and its navigation maps |
| [actions.md](actions.md) | `actions.py` | The action vocabulary |
| [perception.md](perception.md) | `perception.py` | What a tribute senses, and the 45-value vector |
| [brain/init.md](brain/init.md) | `brain/__init__.py` | The brain registry |
| [brain/base.md](brain/base.md) | `brain/base.py` | The Brain interface and learning hooks |
| [brain/initializers.md](brain/initializers.md) | `brain/initializers.py` | Weight initializers, activations and their derivatives |
| [brain/mlp.md](brain/mlp.md) | `brain/mlp.py` | The multi-layer perceptron with backpropagation and Adam |
| [brain/voting.md](brain/voting.md) | `brain/voting.py` | The chapter 4 voting brain |
| [brain/random_brain.md](brain/random_brain.md) | `brain/random_brain.py` | The dice-rolling baseline |
| [brain/neural.md](brain/neural.md) | `brain/neural.py` | The multi-layer numpy network |
| [player.md](player.md) | `player.py` | The body |
| [sponsors.md](sponsors.md) | `sponsors.py` | Parachutes for favoured tributes |
| [gamemaker.md](gamemaker.md) | `gamemaker.py` | The shrinking safe circle |
| [scenario.md](scenario.md) | `scenario.py` | Painted map, loot and roster as JSON |
| [records.md](records.md) | `records.py` | The spreadsheet rows |
| [game.md](game.md) | `game.py` | The referee for one game |
| [recorder.md](recorder.md) | `recorder.py` | Tick-by-tick recordings |
| [runner.md](runner.md) | `runner.py` | Batches of games and CSV output |
| [renderer.md](renderer.md) | `renderer.py` | Drawing games and exporting GIFs |
| [analysis.md](analysis.md) | `analysis.py` | The chapter 3 charts |
| [training/init.md](training/init.md) | `training/__init__.py` | The training package |
| [training/genetic.md](training/genetic.md) | `training/genetic.py` | The genetic algorithm trainer |
| [training/reinforce.md](training/reinforce.md) | `training/reinforce.py` | REINFORCE with a value baseline |
| [training/runs.md](training/runs.md) | `training/runs.py` | Writing a training run folder |
| [research/init.md](research/init.md) | `research/__init__.py` | The research package |
| [research/telemetry.md](research/telemetry.md) | `research/telemetry.py` | Behaviour measurements during games |
| [research/plots.md](research/plots.md) | `research/plots.py` | One PNG per chart |
| [research/experiments.md](research/experiments.md) | `research/experiments.py` | Parameter sweeps and run folders |
| [ui/init.md](ui/init.md) | `ui/__init__.py` | The dashboard package |
| [ui/painter.md](ui/painter.md) | `ui/painter.py` | The paintable terrain grid |
| [ui/session.md](ui/session.md) | `ui/session.py` | The dashboard's state |
| [ui/canvas.md](ui/canvas.md) | `ui/canvas.py` | Drawing the arena |
| [ui/visualizer.md](ui/visualizer.md) | `ui/visualizer.py` | The neural network as a node graph |
| [ui/screenshots.md](ui/screenshots.md) | `ui/screenshots.py` | Taking the tutorial's pictures from the real dashboard |
| [ui/app.md](ui/app.md) | `ui/app.py` | The window, tabs and buttons |
| [ui/main.md](ui/main.md) | `ui/__main__.py` | `python -m hunger_games.ui` |
| [main.md](main.md) | `__main__.py` | The command line |
| [init.md](init.md) | `__init__.py` | The package front door |

## Scripts: `experiments/`

| Page | Source file | One line |
| --- | --- | --- |
| [experiments/run_ga.md](experiments/run_ga.md) | `experiments/run_ga.py` | A genetic-algorithm training run from the command line |
| [experiments/run_rl.md](experiments/run_rl.md) | `experiments/run_rl.py` | A policy-gradient training run from the command line |
| [experiments/run_sweep.md](experiments/run_sweep.md) | `experiments/run_sweep.py` | A parameter sweep from the command line |

## Tests: `tests/`

| Page | Source file | Guards |
| --- | --- | --- |
| [tests/test_noise.md](tests/test_noise.md) | `test_noise.py` | Noise range, repeatability, smoothness |
| [tests/test_arena.md](tests/test_arena.md) | `test_arena.py` | Thresholds, shapes, podiums, distance maps, layouts |
| [tests/test_brains.md](tests/test_brains.md) | `test_brains.py` | Perception vector, brain interface, genomes |
| [tests/test_initializers.md](tests/test_initializers.md) | `test_initializers.py` | Initializers, activations, multi-layer network |
| [tests/test_game.md](tests/test_game.md) | `test_game.py` | Full games, seeding, podium spreading |
| [tests/test_scenario.md](tests/test_scenario.md) | `test_scenario.py` | Scenarios, starting bars, sponsors, wounds |
| [tests/test_recorder_training.md](tests/test_recorder_training.md) | `test_recorder_training.py` | Recordings and the trainer |
| [tests/test_ui_session.md](tests/test_ui_session.md) | `test_ui_session.py` | The painter and the session |
| [tests/test_research.md](tests/test_research.md) | `test_research.py` | Gradient checks, telemetry, RL, sweeps, plots |
| [tests/test_feed.md](tests/test_feed.md) | `test_feed.py` | Showcase recordings and the training feed |

## Results: `output/`

| Page | Folder | Describes |
| --- | --- | --- |
| [output.md](output.md) | `output/` | The four CSV files column by column, the report charts, the GIF and snapshots |

See the top-level [README.md](../README.md) for the command reference and
the table of default settings.
