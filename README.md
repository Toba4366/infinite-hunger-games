# Infinite Hunger Games

A Hunger Games simulator built from chapter 4 of the "Infinite Hunger Games"
video (https://youtu.be/dS3tgfNN1HM?t=1013), written in beginner-friendly,
object-oriented Python with a comment on every line of code, plus a
game makers' dashboard and a research toolkit for training brains and
proving what they learned.

- **Start here:** [docs/tutorial/README.md](docs/tutorial/README.md) is the illustrated walkthrough; the dashboard's first tab is the same tutorial with buttons that perform each step.
- **Dashboard guide:** [docs/ui/README.md](docs/ui/README.md) explains every tab and control.
- **Research guide:** [docs/research/README.md](docs/research/README.md) maps every question to the chart that answers it.
- **Full documentation:** [docs/README.md](docs/README.md) has one in-depth page per source file, including the tests and scripts.
- **Changelog:** [CHANGELOG.md](CHANGELOG.md) records what changed, why, and what was measured.
- **Source:** [hunger_games/](hunger_games/), [experiments/](experiments/), [tests/](tests/).
- **Results:** [output/](output/) holds a 200-game dataset, one PNG per chart, snapshots and a sample GIF.

## Contents

1. [Install and check](#install-and-check)
2. [The dashboard](#the-dashboard)
3. [Command reference](#command-reference)
4. [Research scripts and run folders](#research-scripts-and-run-folders)
5. [Python API](#python-api)
6. [The chaos dial](#the-chaos-dial)
7. [Default settings and where they came from](#default-settings-and-where-they-came-from)
8. [How the video's ideas map to the code](#how-the-videos-ideas-map-to-the-code)
9. [Package map](#package-map)
10. [Brains, training and what tributes know](#brains-training-and-what-tributes-know)
11. [Reading the results](#reading-the-results)
12. [Keeping the project healthy](#keeping-the-project-healthy)

## Install and check

```bash
pip install -r requirements.txt      # numpy, matplotlib, pandas, pillow, dearpygui, pytest
python -m pytest tests               # 47 tests, about fifteen seconds
ruff format --check . && ruff check .   # the code is formatted and lint-clean (config in pyproject.toml)
```

Python 3.10 or newer is required.

## The dashboard

```bash
python -m hunger_games ui
```

A resizable three-panel window. Left: the control tabs. Centre: the arena
with a transport bar. Right: the inspector, the neural network visualiser and
live behaviour charts. Every control has a tooltip, and settings that change
the world regenerate the arena at once.

| Tab | What you can do |
| --- | --- |
| Setup | Arena shape, loot layout, size, seed, chaos, days, tributes, starting bars, senses, what tributes know (cannon and sky, endgame instinct), sponsors, the slow game maker circle, water podiums. Save and load configs. |
| Map | Paint water, sand, grass, rock or void with a brush shown as a ring on the arena. Stamps and presets: Perlin, flat field, flat round, the 75th games' island, a lake with an island. Save and load scenarios. |
| Loot | Click to place food, weapons or medkits with a quantity and quality. |
| Tributes | Podium presets (edge ring, around cornucopia, random, two sides) or drag tributes. Rename, change district, sex, scores, brain, grant a weapon, food, medkits or sponsor favour, set starting bars. |
| Brains | Default brain, and the neural network: the number of hidden layers, the nodes in each, activation, every initializer, with the 50 inputs and 16 outputs listed. |
| Play | Speed presets, back-to-back games, replays, GIF export. |
| Train | Genetic algorithm or REINFORCE, the reward function, live performance, stability and timing plots, the champion's genes with the changed ones highlighted, a training feed that replays a real evaluation game from every step or lets the newest champion play live, watch the champion, save the run folder. |
| Research | Parameter sweeps over any setting, and one-PNG-per-chart exports of the behaviour of every game watched. |

The right panel's Network tab draws the selected neural tribute's network as
a node graph whose hidden activations change in real time as the game plays,
and plots how the champion genome changed over training.

## Command reference

`python -m hunger_games <command> [options]`. Every value flag's default is
read from `SimulationConfig`, so the command line and the code never drift.

| Command | What it does |
| --- | --- |
| `ui` | Open the dashboard. |
| `watch` | Animate one game in a matplotlib window. `--speed N`, `--save PATH` (GIF or MP4). |
| `simulate` | Play many games to CSV. `--games N` (100), `--workers N` (1), `--output DIR` (output). |
| `analyze` | Print the headline numbers, write `report.png` and one PNG per chart under `output/plots/`. `--output DIR`, `--show`. |

Shared options for `watch` and `simulate`:

| Option | Default | What it changes |
| --- | --- | --- |
| `--shape` | `open_field` | `open_field` (74th games) or `round` (75th games). |
| `--layout` | `ring` | `ring` (the video's redesign) or `cornucopia` (the original pile). |
| `--chaos X` | 0.5 | The randomness dial. |
| `--seed N` | random | Repeatable runs. |
| `--size N` | 120 | Arena width and height in cells. |
| `--players N` | 24 | Number of tributes. |
| `--brain NAME` | `voting` | `voting`, `random` or `neural`. |
| `--days N` | 24 | Strict cutoff; a draw if more than one tribute remains. |
| `--gamemaker` / `--no-gamemaker` | on | The slowly shrinking safe circle. |
| `--sponsors` / `--no-sponsors` | on | Parachute gifts. |

Headless: prefix commands with `MPLBACKEND=Agg`.

## Research scripts and run folders

```bash
python experiments/run_ga.py --brain neural --population 48 --generations 20 --workers 4
python experiments/run_rl.py --epochs 30 --episodes 4 --learners 6 --workers 4
python experiments/run_sweep.py --parameter chaos --values 0,0.25,0.5,0.75,1 --games 50 --workers 4
python experiments/run_sweep.py --parameter terrain.water_threshold --values 0.1,0.25,0.4
```

Each writes `results/<name>_<timestamp>/` containing `config.json`,
`history.json` (or `results.csv` and `summary.json` for a sweep),
`champion.json`, and `plots/` with one PNG per chart plus a growing-curve
GIF. The dashboard's Train and Research tabs write the same folders.

## Python API

```python
from hunger_games import Game, Runner, SimulationConfig
from hunger_games.research import BehaviorTelemetry
from hunger_games.research.experiments import Sweep, SweepConfig
from hunger_games.training import GeneticTrainer, ReinforceTrainer, RLConfig, TrainingConfig, save_run

config = SimulationConfig(seed=7)

# One game with behaviour measurement.
game = Game(config)
telemetry = BehaviorTelemetry(game.arena.width, game.arena.height).attach(game)
result = game.run()
summary = telemetry.summary()            # actions against needs, heatmaps, deaths by cause

# Evolve neural brains, then write a run folder.
ga = GeneticTrainer(config, TrainingConfig(brain_name="neural", population_size=48, generations=20, workers=4))
ga.run(on_generation=lambda s: print(s.generation, s.best_fitness, s.val_fitness))
save_run(ga, "genetic", "ga")

# Policy gradient with a value baseline.
rl = ReinforceTrainer(config, RLConfig(epochs=30, episodes_per_epoch=4, learners_per_game=6, workers=4))
rl.run(on_epoch=lambda e: print(e.epoch, e.policy_loss, e.entropy, e.val_return))
save_run(rl, "reinforce", "rl")

# A sweep.
Sweep(config, SweepConfig(name="chaos", parameter="chaos", values=[0.0, 0.5, 1.0], games_per_value=50)).run()
```

Run scripts with more than one worker from a file with a `main()` guard, as
the `experiments/` scripts do; macOS starts workers by re-importing the
script.

## The chaos dial

`chaos` (0.0 to 1.0) scales hunting luck, fighting luck, terrain roughness
and how often a brain picks a lower-voted action (voting) or samples from
its softmax (neural). Starting bars have their own settings. At any chaos
level the same seed and settings replay the same game exactly.

## Default settings and where they came from

The "source" column says whether a value came from the video, the films and
books, a convention, or calibration against the video's chapter 3 data
(about 76% of eliminations player-versus-player, 15% game maker, 6% natural
causes, and about eleven deaths on day one of the 74th games).

Measured with the defaults over 20 seeded games:

| Setting | Player vs player | Natural | Game maker | Victor | Mean days |
| --- | --- | --- | --- | --- | --- |
| Default: ring, open field, slow circle | 72% | 23% | 4% | 19/20 | 18.9 |
| Round arena | 75% | 22% | 2% | 18/20 | 18.4 |
| Cornucopia layout | 89% | 11% | 0% | 19/20 | 12.6 |
| No circle, endgame instinct on | 79% | 21% | 0% | 18/20 | 17.4 |
| No circle, no instinct (strict cutoff only) | 81% | 19% | 0% | 0/20 | 24 (all draws) |

That last row is why the slow circle is on by default: a cutoff alone never
produces a victor. The endgame instinct is the intervention-free alternative
and is one toggle away.

### `SimulationConfig` ([config.py](hunger_games/config.py))

| Setting | Default | Source | Why |
| --- | --- | --- | --- |
| `width`, `height` | 120 | convention | Room for 24 tributes; one dot per cell on screen. |
| `shape`, `layout` | `OPEN_FIELD`, `RING` | the films, the video | The 74th games' forest; the redesign under test. |
| `allow_water_podiums` | `True` | the films | The 75th games' podiums stood in the sea. |
| `num_players` | 24 | the books | Two per district, one female and one male. |
| `brain_name` | `"voting"` | the video | Chapter 4's instinct-voting brain. |
| `start_thirst_min`, `start_hunger_min`, `start_health_min` | 1.0 | convention | Everyone starts full; lower a minimum for a random spread. |
| `career_districts` | (1, 2, 4) | the books | Districts that train and attract sponsors. |
| `chaos` | 0.5 | convention | Halfway. |
| `seed` | `None` | convention | Random unless asked. |
| `ticks_per_day` | 24 | convention | One tick per hour. |
| `max_days` | 24 | decision | A strict cutoff, raised from 18. |
| `vision_radius`, `landmark_radius` | 8, 30 | convention, tuned | People versus lakes and meadows. |
| `thirst_days`, `hunger_days` | 3, 7 | rule of threes, tuned | Water kills fastest. |
| `sponsors_enabled`, `sponsor_gift_chance` | `True`, 0.5 | the films | About six parachutes per game. |
| `gamemaker_enabled` | `True` | measured | See the table above; the circle is slow and rarely kills. |
| `quiet_days_before_intervention` | 1.0 | the films | A quiet day before the game makers act. |
| `intervention_days` | 6.0 | decision | The circle takes six days of shrinking to close, up from three. |
| `cannon_and_sky` | `True` | the books | Tributes know how many remain and, having trained together, how strong. |
| `endgame_instinct` | `False` | decision | Bold tributes head for the centre when few remain; off unless you want it. |
| `noise`, `terrain` | see below | video, tuned | Perlin settings and the relative thresholds. |
| `neural` | (16,), tanh, xavier_uniform | convention | One hidden layer; 50 inputs, 16 outputs, 1,088 weights. |
| `reward` | see below | convention | The reinforcement-learning reward function. |

### Noise and terrain

| Setting | Default | Source |
| --- | --- | --- |
| `noise.scale`, `octaves`, `persistence`, `lacunarity` | 40, 5, 0.5, 2.0 | tuned, standard fBm, Perlin's originals |
| `terrain.water_threshold`, `sand_size`, `grass_size` | 0.25, 0.10, 0.50 | the video (0.25), tuned |
| Hunt difficulty grass / water / sand / rock | 0.2 / 0.6 / 0.8 / 0.9 | the video (grass, water); ours (sand, rock) |
| Move success grass / water / sand / rock | 1.0 / 0.5 / 0.85 / 0.6 | tuned so chases end |

### Player body, sponsors, weapons

| Rule | Default | Source |
| --- | --- | --- |
| Drink, eat, heal, rest amounts | 0.5, 0.35, 0.4, 0.02 | tuned, convention |
| Serious wound below 0.5 health bleeds 0.004 per tick; rest does nothing to it | | the video ("even a cut") |
| Hunt: one ration per 0.1 of survival score above the difficulty | | the video |
| Fight damage 0.35 + 0.45 x weapon; strength 0.4 survival + 0.4 weapon + 0.2 health | | tuned |
| Weapon reach: fists/rock/knife 1, spear/sword 2, bow 3 | | tuned |
| Training score normal(6.5, 2.5) clamped 1..12; survival 0.6 x score/12 + 0.4 x random | | the films |
| Sponsor favour 0.5 x score/12 + 0.25 if career + min(0.25, 0.08 x kills) + bonus; gifts only when in need | | the films |
| Layout medkits: 5% of the Cornucopia pile, 3% of ring stacks | | decision |
| Weapon names fists 0, rock 0.2, knife 0.4, spear 0.6, sword 0.8, bow 0.9 | | convention |

### Voting brain genes ([brain/voting.py](hunger_games/brain/voting.py))

thirst 1.0, hunger 1.0, survival 1.0, danger 1.5, greed 0.6, aggression 0.5,
caution 0.5, urgency power 2.0; a need below 0.2 casts 20 extra votes
(chapter 4's "insist on drinking"); panic distance 4. All evolvable.

### Reward function ([config.py](hunger_games/config.py) `RewardConfig`)

| Term | Default | When |
| --- | --- | --- |
| `survive_tick` | 0.01 | Every tick alive. |
| `win` | 5.0 | Sole survivor. |
| `death` | -3.0 | On dying, once. |
| `kill` | 1.0 | Per elimination. |
| `damage_taken` | -2.0 | Per point of health lost. |
| `need_gain` | 0.5 | Per point of thirst or hunger restored while below half. |
| `placement` | 2.0 | Scaled by placing at the end: full for first, nothing for last. |
| `discount` | 0.98 | How much a reward one tick later is worth now. |

### Trainers

| Setting | Genetic (`TrainingConfig`) | REINFORCE (`RLConfig`) |
| --- | --- | --- |
| Steps | 20 generations, population 48, 2 games per genome | 30 epochs, 4 games each, 6 learners per game |
| Selection and update | elite 10%, tournament of 3, crossover 0.5, mutation rate 0.1, scale 0.1 | Adam, learning rate 1e-3 (value 3e-3), entropy bonus 0.01, gradient clip 5 |
| Validation | champion versus the default brain on 2 fixed seeds | greedy policy on 2 fixed seeds |
| Logged per step | best, mean, worst and validation fitness, seconds, telemetry | policy and value loss, entropy, train and validation return, survival, win and kill rate, seconds, telemetry |

## How the video's ideas map to the code

| Video idea | Where it lives |
| --- | --- |
| Terrain as heights from a noise function | [noise.py](hunger_games/noise.py) |
| Relative terrain thresholds | [config.py](hunger_games/config.py), [terrain.py](hunger_games/terrain.py) |
| Resources as quality, quantity and type per cell; Cornucopia versus ring | [resources.py](hunger_games/resources.py) |
| Podiums with high scorers spread apart (chapter 2) | [game.py](hunger_games/game.py) |
| Body versus brain; instincts vote | [player.py](hunger_games/player.py), [brain/voting.py](hunger_games/brain/voting.py) |
| Hunt: survival score against terrain difficulty | [player.py](hunger_games/player.py) |
| Interventions as a last resort (chapters 1 and 3) | [gamemaker.py](hunger_games/gamemaker.py), slow and toggleable |
| Even a cut is deadly; sponsors | [player.py](hunger_games/player.py), [sponsors.py](hunger_games/sponsors.py) |
| Every elimination logged; "every emergent action plotted" | [records.py](hunger_games/records.py), [research/](hunger_games/research/) |
| "A tool to fine-tune the very nature of the Hunger Games" | [ui/](hunger_games/ui/) |

## Package map

```
hunger_games/
  config.py, noise.py, terrain.py, districts.py, resources.py, arena.py
  actions.py, perception.py (50-value vector, VECTOR_NAMES)
  brain/  base.py, initializers.py, mlp.py (backprop, Adam), voting.py, random_brain.py, neural.py
  player.py, sponsors.py, gamemaker.py, scenario.py, records.py
  game.py (decision and tick hooks), recorder.py, runner.py, renderer.py, analysis.py
  training/  genetic.py, reinforce.py, runs.py
  research/  telemetry.py, plots.py, experiments.py
  ui/  painter.py, session.py, canvas.py, visualizer.py, app.py, screenshots.py
  __main__.py
experiments/  run_ga.py, run_rl.py, run_sweep.py
tests/        44 tests
docs/         one page per file, plus tutorial/, ui/README.md and research/README.md
output/       dataset, plots/, snapshots, GIF
results/      run folders written by trainers and sweeps
```

## Brains, training and what tributes know

Every brain implements `decide(perception, rng) -> Action`. The perception
is a 50-value vector (own bars and scores, terrain, directions to water,
grass and the centre, loot, the nearest threat, the game makers' circle, the
clock, and, from the cannon and the nightly sky, the size and strength of
the remaining field and the tribute's rank in it). The neural brain maps it
to 16 menu items through an `MLP` that supports both evolution (a flat
genome) and backpropagation (REINFORCE with a value baseline). The
`research/` package measures what any brain does against its needs, danger,
position and the field, and draws one PNG per chart.

## Reading the results

`output/` holds `eliminations.csv`, `players.csv`, `games.csv`, `gifts.csv`,
`report.png`, and `plots/` with one PNG per chart including a death heatmap.
Column-by-column descriptions are in [docs/output.md](docs/output.md).
Training and sweep run folders are described in
[docs/research/experiments.md](docs/research/experiments.md).

## Keeping the project healthy

After any change: `ruff format . && ruff check .`, `python -m pytest tests`,
update the matching page under `docs/` (read the whole source file first),
update this README if a command or default changed, and add a line to
`CHANGELOG.md` under Unreleased.
