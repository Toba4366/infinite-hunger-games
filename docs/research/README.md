# Proving a brain is learning the right behaviours

A guide for a researcher who wants to show, with charts, that a trained tribute is not just winning more but doing the sensible things: drinking when thirsty, fleeing when hurt, avoiding crowds early and seeking them late. Every chart below is one PNG from [plots.md](plots.md), fed by counts from [telemetry.md](telemetry.md), collected while the trainers in [../training/genetic.md](../training/genetic.md) and [../training/reinforce.md](../training/reinforce.md) play their games. Sweeps over settings come from [experiments.md](experiments.md).

The pages in this folder:

| Page | Covers |
| --- | --- |
| [init.md](init.md) | The package front door and how data flows |
| [telemetry.md](telemetry.md) | Every tally, the bins, the hooks, `summary()` and `merge()` |
| [plots.md](plots.md) | Every chart function and the three bundles |
| [experiments.md](experiments.md) | Parameter sweeps and the run folder layout |

## How the measurements are taken

Nothing in the research package changes a game. `Game` keeps two lists of listener functions, `decision_hooks` and `tick_hooks` (see [../game.md](../game.md)). `BehaviorTelemetry.attach(game)` adds one function to each.

- After every decision, `on_decision(player, perception, action)` receives the exact `Perception` the brain saw and the `Action` it chose. That is when the action-by-need tables, the combat table, the proximity sums and the item timing histograms are filled.
- At the end of every tick, `on_tick(game)` records where every living tracked tribute stands (the heatmaps), notes the first tick a tribute drops under half health, and catches newly dead tributes to record their bars and cause of death.
- When the game is over, `on_game_end(game)` writes one survival time, kill count, win flag and placing per tracked tribute.

`summary()` turns the tallies into plain lists. `merge()` adds summaries from many games or worker processes. The trainers call `merge` once per step and keep the result on the step's stats, which is how the "over training" charts get one point per generation or epoch.

Who is measured matters:

| Where | Who is tracked | Why it matters |
| --- | --- | --- |
| `Runner` batches and sweeps | Every tribute | With the default voting brain, the charts describe hand-coded behaviour |
| GA evaluation games | Every tribute, all driven by population genomes | The whole population's behaviour, not just the champion's |
| RL episodes | The 6 learner slots only | Opponents (voting brain) are excluded, so the charts show the policy alone |
| Dashboard watched games | Every tribute on the roster | Whatever brains the roster has, mixed |

## Where the numbers live

| File | Written by | Contents |
| --- | --- | --- |
| `results/<run>/history.json` | `save_run` | One row per step. GA: `generation`, `best_fitness`, `mean_fitness`, `worst_fitness`, `val_fitness`, `seconds`, `cumulative_seconds`. RL: `epoch`, `policy_loss`, `value_loss`, `entropy`, `train_return`, `val_return`, `train_survival`, `val_survival`, `win_rate`, `val_win_rate`, `kill_rate`, `seconds`, `cumulative_seconds` |
| `results/<run>/champion.json` | `save_run` | The best genome or policy, loadable from the dashboard's Train tab |
| `results/<sweep>/results.csv` | `Sweep.write` | One row per swept value with every metric |
| `results/<sweep>/summary.json` | `Sweep.write` | The rows plus one merged telemetry summary per value |
| `results/<sweep>/batches/<value>/` | `Runner.save` | The four CSV tables and `telemetry.json` for that value |
| `output/telemetry.json` | `Runner.save` | The merged summary of a `simulate` batch run with telemetry |

The telemetry summaries themselves are not in `history.json` (they are large). They live in the plots a run writes, and in `summary.json` for sweeps. To keep them for a training run, save `[s.telemetry for s in trainer.history]` yourself.

## The three questions

A reviewer asks three things. Is it performing better? Is it behaving sensibly? Is the training itself stable? Each has its own charts.

## 1. Performance

| Question | Chart (file in `plots/`) | Data | Code |
| --- | --- | --- | --- |
| Is reward per episode rising? | `reward.png`, `reward.gif` | `train_return`, `val_return` per epoch | `plots.curves`, `plots.curve_gif` via `training_run_plots` |
| Is fitness rising? (GA) | `fitness.png`, `fitness.gif` | `best_fitness`, `mean_fitness`, `val_fitness` per generation | same |
| Are tributes living longer? | `survival.png` (RL), `behaviour_over_training.png` panel 1 | `train_survival`, `val_survival`; `mean_survival_ticks` | `plots.curves`, `plots.behaviour_metrics_over_training` |
| Are they winning and killing more? | `win_kill_rate.png` (RL), `behaviour_over_training.png` panels 2 and 3 | `win_rate`, `val_win_rate`, `kill_rate` | same |

**What a good trend looks like.** Training return rises and then flattens. Validation return follows it. Validation is played with the greedy policy on fixed seeds against the config's brain, so it is the honest number. A big gap with training high and validation flat means the policy is exploiting its own sampling noise, not the game. Survival ticks should rise before win rate does, because staying alive is the first thing the reward teaches. Win rate on a 24-tribute field is small by nature; 6 learners winning 1 game in 4 is a 4 percent per-learner rate and already strong. For the GA, best fitness rising while mean fitness follows a few generations behind is healthy; mean fitness stuck near the start while best fitness jumps around means the games are too noisy, so raise `rounds_per_generation`.

## 2. Behaviour

These charts read a telemetry summary. In a training run folder the detailed ones describe the last generation or epoch; the "over training" ones use every step.

| Question | Chart | Summary keys | Code |
| --- | --- | --- | --- |
| Is the action mix changing from random to structured? | `action_distribution_over_training.png` (stacked area) | `action_counts` per step | `plots.stacked_area_over_training` |
| What does it do overall? | `action_distribution.png` | `action_counts`, `entropy` | `plots.action_distribution` |
| Where does it spend time? | `position_heatmap.png` | `position_heat` | `plots.heatmap` |
| Do armed and unarmed tributes go to different places? | `armed_vs_unarmed_heatmaps.png` | `armed_heat`, `unarmed_heat` | `plots.armed_vs_unarmed` |
| What were the bars at death? | `death_needs_over_training.png` | `mean_death_needs` per step | `plots.death_needs_over_training` |
| Does it drink when thirsty, eat when hungry, heal when hurt? | `instinct_curves.png` | `action_by_thirst`, `action_by_hunger`, `action_by_health` | `plots.need_action_curves` |
| What else does it do at each need level? | `actions_by_thirst.png`, `actions_by_hunger.png`, `actions_by_health.png` | same | `plots.action_by_need` |
| When exactly does it use items? | `consumption_timing.png` | `thirst_at_drink`, `hunger_at_eat`, `health_at_heal` | `plots.consumption_timing` |
| Does it fight when strong and flee when weak? | `fight_or_flight.png` | `combat_by_health` | `plots.fight_or_flight` |
| Does it survive after an injury? | `post_injury_ticks` in `summary.json` | `post_injury_ticks` | read the list; no dedicated chart yet |
| Does it keep its distance early and close in late? | `proximity_vs_remaining.png` | `proximity_sum`, `proximity_count` | `plots.proximity_vs_alive` |
| Does aggression rise as the field shrinks? | `actions_by_remaining.png` | `action_by_alive` | `plots.action_by_alive` |
| What is killing it? | `deaths_by_cause.png` | `deaths_by_cause` | `plots.deaths_by_cause` |

**What a good trend looks like, chart by chart.**

- **Stacked area over training.** Early epochs show nine bands of roughly equal width, because a fresh network samples almost uniformly. Later the `move` band widens, `drink` and `eat` take a steady slice, and `rest` shrinks. A single band swallowing the chart is collapse.
- **Position heatmap.** For the ring layout, a good brain lights up the water and grass and the loot ring, with a visible trail to the centre late in the game. A uniform smear means aimless wandering. A single hot cell means the brain has learned to stand still.
- **Armed versus unarmed.** Armed tributes should be brighter in the centre; unarmed ones near the edges and water. If both panels look the same the network is not using the `weapon quality` input.
- **Resource levels at death.** Thirst and hunger at death should rise over training. That sounds backwards until you remember what it means: tributes stop dying with empty bars, so the deaths that remain are fights with the bars still full. Health at death is always near zero.
- **Instinct curves.** P(drink given thirst) should be near zero at 80 to 100 percent and rise steeply below 40 percent. Same for eat and heal. A flat line is a brain that has not connected the bar to the action. A curve that is high everywhere is a brain that drinks constantly, which the reward discourages because `need_gain` only pays while a bar is under half.
- **Item usage timing.** Mass in the left-hand bins. Learning tributes cluster at low levels; a random brain is flat.
- **Fight or flight.** Flee should dominate the 0 to 40 percent health bins and attack the 80 to 100 percent bin. The crossover point moving right over training means the brain is getting more cautious.
- **Survival after injury.** Longer `post_injury_ticks` over training means the brain has learned to heal, hide or rest once hurt. Compare the list's mean between the first and last step.
- **Proximity versus tributes remaining.** A downward slope from "most alive" to "final few". The tribute keeps others at arm's length early, then closes in. Flat means it ignores the field size.
- **Actions by tributes remaining.** The attack share should grow toward "final few". The video's endgame instinct is the hand-coded version of this; a learned one should appear without the toggle.

## 3. Training stability

| Question | Chart | Data | Code |
| --- | --- | --- | --- |
| Is the policy collapsing too early? | `entropy.png` | `entropy` per epoch (policy entropy at the decisions made) | `plots.curves` via `training_run_plots` |
| Is the baseline learning? | `losses.png` | `policy_loss`, `value_loss` | same |
| How long does each step take? | `timing.png` | `seconds`, `cumulative_seconds` | `plots.timing` |
| Is behaviour entropy tracking? | `behaviour_over_training.png` panel 4 | `entropy` from telemetry | `plots.behaviour_metrics_over_training` |

**Good trends.** Policy entropy starts near `ln 16 = 2.77` nats (sixteen menu items) and falls slowly. A fast drop to near zero in a few epochs is premature collapse; raise `entropy_bonus`. Value loss should fall and then hover. Policy loss is noisy by nature and can even be negative, because advantages are normalised each epoch; look at its scale, not its sign. Seconds per step should be flat; a rising bar chart means episodes are getting longer because tributes survive longer, which is itself evidence of learning.

For the GA the dashboard's stability panel shows action entropy from telemetry rather than policy entropy, since there is no policy loss. The same falling-but-not-zero shape is what you want.

## Answers for reviewers

### RL versus evolutionary

Two trainers share the same brain and the same game.

| | REINFORCE with baseline (`training/reinforce.py`) | Genetic algorithm (`training/genetic.py`) |
| --- | --- | --- |
| What is scored | Every action, tick by tick | Whole games, by placement |
| Signal | Reward from `RewardConfig`, discounted, minus a learned value baseline, advantages normalised | Fitness `(n - placement) / (n - 1) + 0.05 kills + 0.01 days` |
| Update | Gradient step with Adam on a numpy MLP, gradient clipped at norm 5, entropy bonus | Tournament selection, uniform crossover, Gaussian mutation, elites kept |
| Opponents | The config's brain (voting by default); 6 learners per game | The population itself, so the target moves |
| Validation | Greedy policy on fixed seeds `90000 + i` | Champion versus the config's brain on the same fixed seeds |
| Works on | The neural brain only | Any brain with a genome, including the voting brain's eight genes |

REINFORCE learns faster per game because it gets a signal every tick, but the signal is noisy. The GA is slow and needs many games, but it never needs a reward function and can tune the voting brain's hand-written weights.

### How reward points are gained and lost

From `RewardConfig` in [../config.md](../config.md), attached to the decision made that tick:

| Event | Points |
| --- | --- |
| Surviving a tick | `+0.01` |
| Losing health | `-2.0` per full bar lost (so a 0.1 wound costs 0.2) |
| Restoring thirst or hunger while the bar was under half | `+0.5` per full bar restored |
| A kill | `+1.0` |
| Dying | `-3.0`, once |
| Finishing the game | `+2.0` scaled by placing, first place gets it all, last gets nothing |
| Winning | `+5.0` on top |
| Discount | `0.98` per tick when summing future rewards into a return |

The placement and win bonuses are added to the last decision of the episode, then discounted backwards through the trajectory.

### How the network sees the world

Not a grid. `Perception.to_vector()` in [../perception.md](../perception.md) produces 50 numbers scaled to roughly -1 to 1:

| Group | Count | Values |
| --- | --- | --- |
| Body | 11 | thirst, hunger, health, survival score, training score, weapon quality, reach, food carried, medkits carried, in water, hunt difficulty |
| Downhill | 2 | step direction |
| Water | 3 | direction (2) and distance |
| Grass | 3 | direction and distance |
| Centre | 3 | direction and distance from the middle |
| Loot here | 3 | kind, quantity, quality |
| Nearby loot | 4 | direction, distance, kind |
| Nearest threat | 5 | direction, distance, threat level, their health |
| Crowd | 1 | players in sight |
| Hazard | 3 | in danger zone, distance to the lethal edge, whether it is closing |
| Safe direction | 2 | step toward safety |
| Clock | 2 | day fraction, alive fraction |
| Field | 4 | known, mean strength of the rest, strongest remaining, my rank |
| Terrain underfoot | 4 | one-hot of water, sand, grass, rock |

The output is a menu of 16 items: rest, drink, eat, hunt, pick up, heal, attack nearest, flee nearest, and eight compass moves. Hidden layers default to one layer of 16 tanh units, so the whole policy is about 1,100 weights.

### How many games back each number

| Number | Games behind it, with the defaults |
| --- | --- |
| One RL epoch's `train_return` | 4 episodes times 6 learners, so 24 learner episodes |
| One RL epoch's `val_return` | 2 fixed-seed games times 6 learners, greedy |
| One GA generation's `best_fitness` | Each genome plays `rounds_per_generation = 2` games; 48 genomes over 24 slots is 4 games per round, 8 games per generation |
| One GA generation's `val_fitness` | 2 fixed-seed games with the champion in 6 slots against the voting brain |
| One sweep row | `games_per_value = 50` games, every tribute measured |

Raise `--episodes`, `--rounds` or `--games` before trusting a small difference.

### What the dashboard is built with

A custom window in Dear PyGui (`hunger_games/ui`), with live line charts drawn by Dear PyGui's own plot widgets during training and sweeps. Every exported chart is a matplotlib PNG from `plots.py`. There is no TensorBoard and no Weights and Biases; the equivalent of a run log is the `history.json` and `summary.json` in each run folder.

### Reproducing a run folder

From the command line, at the repo root:

```bash
python experiments/run_ga.py --brain neural --population 48 --generations 20 --rounds 2 --workers 4 --seed 0
python experiments/run_rl.py --epochs 30 --episodes 4 --learners 6 --lr 1e-3 --entropy 0.01 --workers 4 --seed 0
python experiments/run_sweep.py --parameter chaos --values 0,0.25,0.5,0.75,1 --games 50 --workers 4 --seed 1000
```

Each writes `results/<name>_<timestamp>/` with `config.json`, then `history.json` and `champion.json` for a training run or `results.csv` and `summary.json` for a sweep, and `plots/`. The `--seed` values make the population, the episode seeds and the games repeatable.

From the dashboard (`python -m hunger_games ui`): on the Train tab pick the method, set the same numbers, press Start training, and when it finishes type a name and press "Save run folder". On the Research tab choose a parameter, type the values, set games per value and workers, and press Start sweep. "Export behaviour charts" writes the behaviour charts for every game watched this session to a folder of your choice. A training run started from the dashboard plays on the painted map and roster, so mention that in a paper.

## A recipe for one figure set

1. Train with fixed seeds so the run can be repeated: `python experiments/run_rl.py --epochs 40 --episodes 6 --workers 4 --seed 1 --name paper_rl`.
2. Open `results/paper_rl_<timestamp>/plots/`. Take `reward.png`, `survival.png` and `entropy.png` for the performance and stability figures.
3. Take `action_distribution_over_training.png` for the headline behaviour figure. It is the one chart that shows random turning into structured.
4. Take `instinct_curves.png`, `fight_or_flight.png` and `proximity_vs_remaining.png` for the three instincts: needs, danger, field size.
5. Take `armed_vs_unarmed_heatmaps.png` to show that the ring layout is being used as designed.
6. Repeat steps 1 and 2 with two more seeds and report the mean and spread of the final validation return. One seed is an anecdote.
7. For a baseline, run the same charts on the untrained voting brain: `python -m hunger_games simulate --games 50 --seed 1000` does not collect telemetry, so use a sweep with one value instead, `python experiments/run_sweep.py --parameter chaos --values 0.5 --games 50`, and read `plots/behaviour/`.

## Comparing two brains fairly

- Use the same seeds. `Game` derives every game's seed from `config.seed + game_id`, so two batches with the same seed and settings play the same arenas.
- Measure the same tributes. Put the trained genome into the roster through a `Scenario` and set `brain_name` for the rest, or track ids with `tracked_ids`.
- Use the same number of games. Entropy and the instinct curves are ratios, but the heatmaps and histograms are counts, and `plots.heatmap` normalises by total time, which differs when tributes live longer.
- Compare validation, not training. Training numbers are sampled at temperature 1 and drift with the entropy bonus; validation is greedy on fixed seeds.
- State the field. A learner's win rate depends on who else is in the arena. The default opponent is the voting brain at `chaos=0.5`.

## What the raw summary lets you compute

The plots cover the common questions. `summary.json` and the `telemetry` on each trainer step hold more.

| Quantity | How |
| --- | --- |
| P(attack given someone in sight) by health | `combat_by_health[:, 0] / combat_by_health.sum(axis=1)` |
| Median survival after injury | `np.median(post_injury_ticks)` |
| Share of decisions spent resting when healthy | `action_by_health[4][0] / sum(action_by_health[4])` |
| Fraction of time in the centre 10 by 10 heat cells | slice `position_heat[10:20, 10:20]` and divide by the total |
| Deaths by cause as shares | divide each `deaths_by_cause` value by `death_count` |
| Placement histogram of learners | `np.bincount(placements)` (survivors show as 0, see the note below) |

## Watching it live in the dashboard

The Train tab draws three live panels while a trainer runs, using Dear PyGui's plot widgets rather than matplotlib:

| Panel | GA lines | RL lines |
| --- | --- | --- |
| Performance | best fitness, validation fitness, mean fitness | training return, validation return, win rate |
| Stability | action entropy from telemetry | policy loss, value loss, policy entropy |
| Timing | seconds per generation | seconds per epoch |

Below them a gene plot shows the champion's weights, with the ones that changed since the last step in gold (the first 400 for big networks). The Charts tab in the right-hand panel, next to Inspector and Network, shows the action distribution, the instinct curves (drink, eat and flee against the need bins) and the position heatmap of every game watched in the session, updated while a game plays. The Research tab's export button writes the full set of PNGs for those same games. None of these live panels are the paper charts; press "Save run folder" on the Train tab or "Export behaviour charts" on the Research tab for those.

If a chart looks wrong in the window, remember two differences from the PNGs: the live heatmap is scaled to its brightest cell rather than to the total, and the live instinct panel shows flee against health where the PNG shows heal.

## Checklist before claiming a brain has learned

1. Validation return or validation fitness rises over the run, on fixed seeds, against the untrained default brain.
2. The stacked area over training changes shape. If every band keeps its width, the policy has not moved.
3. At least one instinct curve is steep: P(drink given thirst) or P(eat given hunger) above 20 percent in the lowest bin and near zero in the highest.
4. Fight or flight crosses over: flee above attack when health is under 40 percent.
5. Policy entropy is above 0.5 nats at the end. Lower than that and the charts describe one habit, not a policy.
6. The same shape appears with a second seed.
7. The measured tributes are the learners. Check `tracked_ids` in the code path that produced the summary, or the "Who is measured" table above.

## Known limitations

- **Small numpy networks.** The brain is a plain MLP written in numpy with one hidden layer by default. There is no GPU path and no recurrence, so the tribute has no memory beyond what the perception carries.
- **REINFORCE variance.** Returns are noisy, rewards are sparse at the end of an episode, and only 4 episodes with 6 learners feed each update. Expect jagged curves. Use validation return on fixed seeds, more episodes per epoch, or several seeds averaged before claiming a trend.
- **Telemetry measures the voting brain too.** Sweeps and GA evaluation games track every tribute. With the default `brain_name="voting"`, part of every behaviour chart is hand-coded behaviour. Only the RL trainer restricts telemetry to its learners. For a clean comparison of a trained brain, load its genome into every tribute first.
- **Heatmaps are 30 by 30 bins.** On a 120-cell arena each bin covers 16 cells. Fine structure such as a single stream is blurred. Change `HEATMAP_CELLS` in `telemetry.py` if you need more, and note that the dashboard's heat series expects 30.
- **Placements in telemetry.** The `placements` list records 0 for survivors because the hook runs before final placings are assigned. Use `wins` and the runner's `players.csv` for exact placings.
