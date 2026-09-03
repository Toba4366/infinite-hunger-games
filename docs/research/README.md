# Proving a brain is learning, and choosing how to train it

A guide for a researcher who wants to show two things with charts. First, that a trained tribute is not just winning more but doing the sensible things: drinking when thirsty, fleeing when hurt, avoiding crowds early and seeking them late. Second, which of the five training methods makes the most sense for this game, and which is the easiest to build. Every chart is one PNG from [plots.md](plots.md), fed by counts from [telemetry.md](telemetry.md) and by the shared `IterationStats` every trainer fills, collected while the trainers in [../training/](../training/init.md) play their games. Sweeps over settings come from [experiments.md](experiments.md); the method comparison and its tournament from [comparison.md](comparison.md).

The pages in this folder:

| Page | Covers |
| --- | --- |
| [init.md](init.md) | The package front door and how data flows |
| [telemetry.md](telemetry.md) | Every tally, the bins, the hooks, `summary()` and `merge()` |
| [plots.md](plots.md) | Every chart function, the bundles and the shared learning curves |
| [experiments.md](experiments.md) | Parameter sweeps and the run folder layout |
| [comparison.md](comparison.md) | Training methods, sizes and initializers trained to a win criterion, warm against cold, then the tournament |

## One learner against the voting strategy

Every method here trains **one** network. It plays as the learner in a few of the tribute slots (6 by default, spread across the roster by `learner_ids`) while every other tribute runs the video's hand-written voting brain. That framing was borrowed from the zombie video, where one agent learns against a fixed environment, and it does three things for a researcher:

- **The opponent is fixed.** A score of 2.0 in epoch 1 and 2.0 in epoch 30 mean the same thing, because the field did not change under the learner. The genetic algorithm used to play the population against itself, where the target moves; `TrainingConfig.opponents="voting"` is now the default and `"self"` keeps the old tournament.
- **The learner is visible.** On the dashboard's arena the learner-driven tributes wear a gold star, so you can follow the network being trained while the voting tributes fill the rest of the field.
- **Every method reports the same numbers.** `IterationStats` in [../training/common.md](../training/common.md) is filled by all five trainers: `scores` (one episode return per learner episode), `mean_score`, `best_score`, `entropy` (nats), `mean_length` (ticks survived), `win_rate`, `val_score` and `val_win_rate` (greedy games on fixed seeds `90000 + i`), `seconds`, `cumulative_seconds`, the curriculum `stage` and `opponents`, and a method-specific `extra` dictionary (losses, accuracy, species counts). One score function, one opponent, one record shape: that is what lets `comparison.py` overlay imitation, evolution and policy gradients on one chart.

The score is always the episode return under `RewardConfig` (see "How reward points are gained and lost" below), even for the evolutionary methods, which use it as fitness.

**What "win rate" means with several learner copies.** The learner drives six tributes in one game, and a game has one victor, so at most one copy can win it. Every win rate in this project is game-level: a game counts as won when any learner copy was the victor, and the rate is the fraction of games won. Counting wins per copy would cap six copies at one sixth; counting games lets the rate reach 1.0, which is what "the learner beats this field" should mean. `win_rate` is over training games, `val_win_rate` over the greedy validation games, and the tournament's `win_rate` over its 75 seeded games.

## The five methods

| Method | What changes each iteration | Needs a teacher? | Gradients? | Evolves structure? | Typical strengths and weaknesses | Where it comes from |
| --- | --- | --- | --- | --- | --- | --- |
| Imitation ([../training/imitation.md](../training/imitation.md)) | One pass of Adam over recorded (perception, action) pairs; cross-entropy against the teacher's choice | Yes, the voting brain | Yes, backpropagation | No | Cheap, stable, gives instincts in minutes; cannot exceed the teacher and copies its mistakes | This project: the fix for fresh networks dying of thirst |
| Genetic algorithm ([../training/genetic.md](../training/genetic.md)) | Score 48 genomes by episode return, keep elites, breed by tournament selection, uniform crossover, Gaussian mutation | No | No | No, weights only | Simplest idea, works for any genome including the voting brain's 8 genes; needs many games and scales badly with weight count | Classic neuroevolution |
| NEAT ([../training/neat.md](../training/neat.md)) | Score genomes, group into species, share fitness, remove stagnant species, breed with weight, add-connection and add-node mutations | No | No | Yes, nodes and connections grow from a minimal genome | Finds small networks and protects innovation; the most machinery, slow per generation, starts tiny | The Monopoly video (knockout-tournament fitness, champion-relative fitness, 11.2 million self-play games) and Stanley and Miikkulainen 2002 |
| REINFORCE ([../training/reinforce.md](../training/reinforce.md)) | Collect 4 episodes, one gradient step on the policy toward well-rewarded actions, advantages from a learned value baseline | No | Yes | No | Learns from every tick; one pass per batch and noisy returns make the curves jagged | Williams 1992, with a value baseline |
| PPO ([../training/ppo.md](../training/ppo.md)) | Same collection, then several clipped passes over the batch with GAE advantages | No | Yes | No | Reuses each batch safely, the most stable reward method; more knobs (clip ratio, passes, minibatch, lambda) | The zombie video |

All five share the same brain input (the 50-value perception vector), the same 16-item action menu, the same opponents, the same validation seeds and the same run folder layout. Imitation, REINFORCE and PPO train a fixed-shape `NeuralBrain`; the genetic algorithm can train that or the voting brain; NEAT trains its own `NeatBrain` whose shape is part of the genome.

## The curriculum

The zombie video starts its agent against one zombie and works up to sixteen. Ours does the same with opponents. `CurriculumConfig` in [../training/common.md](../training/common.md):

| Setting | Default | Meaning |
| --- | --- | --- |
| `opponents` | `(1, 3, 7, 11, 23)` | Voting opponents per stage; the learner's own copies are added on top |
| `promote_on` | `"win_rate"` | Judge promotion on the win rate (the learner must actually win games) or on `"score"` |
| `win_threshold` | `0.5` | Promote when the win rate of the last `window` iterations averages at least this: a majority of games |
| `threshold` | `3.0` | The bar when judging on score instead |
| `window` | `5` | Iterations averaged for the promotion test |
| `max_iterations_per_stage` | `0` | Promote anyway after this many iterations in a stage; `0` means never |

**Promotion rule.** After every iteration the trainer calls `curriculum.observe(mean_score, win_rate)`, passing the validation win rate when it plays validation games and the training win rate otherwise. It returns `True`, and the trainer logs a `curriculum` event, when the learner has won at least half of those games over the last five iterations. There is no timeout by default: a learner that cannot beat one opponent stays against one opponent, which is a finding, not a bug. The trainer then rebuilds its config with `num_players = learners + opponents`, so the first stage is a 7-tribute game (6 learners and 1 opponent) and the last is the full 29 (6 and 23). The stage and opponent count are on every `IterationStats`, and `plots/curriculum.png` in a run folder draws the ladder. Genetic, NEAT, REINFORCE and PPO all judge promotion on validation wins. Imitation ignores the curriculum because it never plays training games.

Why it helps: a fresh network against 23 voting tributes dies in the bloodbath before its rewards say anything. Against one opponent it survives long enough for the survival and needs terms to teach it, and the harder stages arrive once it can actually win the easier ones. Why it complicates comparison: the mean score drops at every promotion, because the field got harder. Read `score_by_method.png` together with `curriculum.png` (or `curriculum_by_method.png` in a comparison), or compare `val_score`, which is always played on the full roster.

## How the measurements are taken

Nothing in the research package changes a game. `Game` keeps two lists of listener functions, `decision_hooks` and `tick_hooks` (see [../game.md](../game.md)). `BehaviorTelemetry.attach(game)` adds one function to each.

- After every decision, `on_decision(player, perception, action)` receives the exact `Perception` the brain saw and the `Action` it chose. That is when the action-by-need tables, the combat table, the proximity sums and the item timing histograms are filled.
- At the end of every tick, `on_tick(game)` records where every living tracked tribute stands (the heatmaps), notes the first tick a tribute drops under half health, and catches newly dead tributes to record their bars and cause of death.
- When the game is over, `on_game_end(game)` writes one survival time, kill count, win flag and placing per tracked tribute.

`summary()` turns the tallies into plain lists. `merge()` adds summaries from many games or worker processes. The trainers call `merge` once per step and keep the result on the step's stats, which is how the "over training" charts get one point per iteration.

Who is measured matters:

| Where | Who is tracked | Why it matters |
| --- | --- | --- |
| `Runner` batches and sweeps | Every tribute | With the default voting brain, the charts describe hand-coded behaviour |
| GA in `"self"` mode | Every tribute, all driven by population genomes | The whole population's behaviour, not just the champion's |
| GA in `"voting"` mode, NEAT, REINFORCE, PPO | The learner slots only | Opponents are excluded, so the charts show the learner alone |
| Imitation validation games | The student slots only | The greedy student alone, against the config's brain |
| The tournament | The learner slots only | Every champion measured the same way |
| Dashboard watched games | Every tribute on the roster | Whatever brains the roster has, mixed |

## Where the numbers live

| File | Written by | Contents |
| --- | --- | --- |
| `results/<run>/learning.json` | `save_run` | One `IterationStats` row per iteration, the same keys for every method, with `extra_*` columns for the method's own numbers |
| `results/<run>/history.json` | `save_run` | The method's own per-step record (GA `GenerationStats`, RL `EpochStats`, imitation `ImitationStats`, NEAT the same rows as `learning.json`) |
| `results/<run>/events.txt` | `save_run` | The event monitor's log: rollouts, evolution, curriculum promotions, records |
| `results/<run>/champion.json` | `save_run` | The best genome or policy, loadable from the dashboard's Train tab; `brain_name` is `neural`, `voting` or `neat` |
| `results/<comparison>/results.csv`, `summary.json`, `results_table.tex`, `report.md` | `MethodComparison.write` | One row per variant with training, criterion and tournament numbers; every learning curve; the generated report |
| `results/<comparison>/runs/<variant>/` | `save_run` | Each variant's own run folder |
| `results/<sweep>/results.csv`, `summary.json` | `Sweep.write` | One row per swept value with every metric, plus merged telemetry |
| `results/<sweep>/batches/<value>/` | `Runner.save` | The four CSV tables and `telemetry.json` for that value |
| `output/telemetry.json` | `Runner.save` | The merged summary of a `simulate` batch run with telemetry |

The telemetry summaries themselves are not in the JSON histories (they are large). They live in the plots a run writes, and in `summary.json` for sweeps. To keep them for a training run, save `[s.telemetry for s in trainer.learning_history]` yourself.

## The three questions

A reviewer asks three things. Is it performing better? Is it behaving sensibly? Is the training itself stable? Each has its own charts. Every run folder now has the shared set (`score.png`, `entropy_shared.png`, `game_length.png`, `win_rate_shared.png`, `score_vs_time.png`, `curriculum.png`, `score.gif`) next to the method's own.

## 1. Performance

| Question | Chart (file in `plots/`) | Data | Code |
| --- | --- | --- | --- |
| Is the score rising? (any method) | `score.png`, `score.gif` | `mean_score`, `best_score`, `val_score` per iteration | `plots.curves` via `learning_curve_plots` |
| Is reward per episode rising? (RL) | `reward.png`, `reward.gif` | `train_return`, `val_return` per epoch | `plots.curves`, `plots.curve_gif` via `training_run_plots` |
| Is fitness rising? (GA) | `fitness.png`, `fitness.gif` | `best_fitness`, `mean_fitness`, `val_fitness` per generation | same |
| Does the student copy the teacher? (imitation) | `accuracy.png` | `train_accuracy`, `val_accuracy` per epoch | same |
| Are tributes living longer? | `game_length.png`, `survival.png` (RL and imitation), `behaviour_over_training.png` panel 1 | `mean_length`; `train_survival`, `val_survival`; `mean_survival_ticks` | `plots.curves`, `plots.behaviour_metrics_over_training` |
| Are they winning and killing more? | `win_rate_shared.png`, `win_kill_rate.png` (RL), `win_rate.png` (imitation), `behaviour_over_training.png` panels 2 and 3 | `win_rate`, `val_win_rate`, `kill_rate` | same |

**What a good trend looks like.** Mean score rises and then flattens. Validation score follows it. Validation is played with the greedy policy on fixed seeds against the config's brain, so it is the honest number. A big gap with training high and validation flat means the policy is exploiting its own sampling noise, not the game. Survival ticks should rise before win rate does, because staying alive is the first thing the reward teaches. Win rate is game-level, so the question it answers is "how often does one of my six copies come out on top of a 24-tribute field"; a fresh network is near zero, and a majority of games won is the bar the curriculum and the comparison use. It is a coarse number per iteration (two validation games give 0, 0.5 or 1), so read it over a window. For the GA and NEAT, best score rising while mean score follows a few generations behind is healthy; mean stuck near the start while best jumps around means the games are too noisy, so raise `rounds_per_generation`. For imitation, validation accuracy climbing with training accuracy and then flattening is what you want; with the default 64 by 32 network it flattens near 80 percent.

## 2. Behaviour

These charts read a telemetry summary. In a training run folder the detailed ones describe the last iteration; the "over training" ones use every step.

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

- **Stacked area over training.** Early iterations show bands of roughly equal width, because a fresh network samples almost uniformly. Later the `move` band widens, `drink` and `eat` take a steady slice, and `rest` shrinks. A single band swallowing the chart is collapse. A warm-started run begins with the structured shape already; look for it to change without falling apart.
- **Position heatmap.** For the ring layout, a good brain lights up the water and grass and the loot ring, with a visible trail to the centre late in the game. A uniform smear means aimless wandering. A single hot cell means the brain has learned to stand still.
- **Armed versus unarmed.** Armed tributes should be brighter in the centre; unarmed ones near the edges and water. If both panels look the same the network is not using the `weapon quality` input.
- **Resource levels at death.** Thirst and hunger at death should rise over training. That sounds backwards until you remember what it means: tributes stop dying with empty bars, so the deaths that remain are fights with the bars still full. Health at death is always near zero.
- **Instinct curves.** P(drink given thirst) should be near zero at 80 to 100 percent and rise steeply below 40 percent. Same for eat and heal. A flat line is a brain that has not connected the bar to the action. A curve that is high everywhere is a brain that drinks constantly, which the reward discourages because `need_gain` only pays while a bar is under half.
- **Item usage timing.** Mass in the left-hand bins. Learning tributes cluster at low levels; a random brain is flat.
- **Fight or flight.** Flee should dominate the 0 to 40 percent health bins and attack the 80 to 100 percent bin. The crossover point moving right over training means the brain is getting more cautious.
- **Survival after injury.** Longer `post_injury_ticks` over training means the brain has learned to heal, hide or rest once hurt. Compare the list's mean between the first and last step.
- **Proximity versus tributes remaining.** A downward slope from "most alive" to "final few". The tribute keeps others at arm's length early, then closes in. Flat means it ignores the field size.
- **Actions by tributes remaining.** The attack share should grow toward "final few". The video's endgame instinct is the hand-coded version of this; a learned one should appear without the toggle.
- **Deaths by cause.** The clearest single number for a fresh network. Measured on the default setup: untrained, 10 of 12 learner deaths were dehydration; after imitation pretraining, 2 of 12.

## 3. Training stability

| Question | Chart | Data | Code |
| --- | --- | --- | --- |
| Is the policy collapsing too early? | `entropy_shared.png` (any method), `entropy.png` (RL) | `entropy` per iteration | `plots.curves` |
| Is the baseline learning? | `losses.png` (RL and PPO) | `policy_loss`, `value_loss` | same |
| Is the student overfitting? | `losses.png`, `losses.gif` (imitation) | `train_loss`, `val_loss` | same |
| Are species surviving? (NEAT) | `extra_species`, `extra_hidden_nodes`, `extra_connections` in `learning.json` | per generation | read the rows |
| How long does each step take? | `timing.png`, `score_vs_time.png` | `seconds`, `cumulative_seconds` | `plots.timing`, `plots.curves` |
| Is behaviour entropy tracking? | `behaviour_over_training.png` panel 4 | `entropy` from telemetry | `plots.behaviour_metrics_over_training` |

**Good trends.** Policy entropy starts near `ln 16 = 2.77` nats (sixteen menu items) for a fresh network, or lower for a warm start, and falls slowly. A fast drop to near zero in a few iterations is premature collapse; raise `entropy_bonus`. Value loss should fall and then hover. Policy loss is noisy by nature and can even be negative, because advantages are normalised each epoch; look at its scale, not its sign. For imitation both cross-entropy losses fall; the validation loss levelling off while the training loss keeps dropping is the moment to stop. For NEAT a species count near `target_species` (8) that neither collapses to one nor explodes is healthy; the champion's hidden node count should creep up, not jump. Seconds per step should be flat; a rising line means episodes are getting longer because tributes survive longer, which is itself evidence of learning.

For the evolutionary methods the entropy on `IterationStats` is action entropy from telemetry rather than policy entropy, since there is no policy loss. The same falling-but-not-zero shape is what you want.

## Comparing methods, sizes and initialisations, then the tournament

Learning curves alone cannot rank methods: an imitation epoch and a NEAT generation cost different amounts of time, and each method's mean score is measured on the games it chose to play. [comparison.md](comparison.md) fixes both. `experiments/run_comparison.py` ([../experiments/run_comparison.md](../experiments/run_comparison.md)) trains every variant **to the win criterion**: each one stops as soon as its validation win rate over the last `--window` iterations (5) reaches `--until-win` (0.5, a majority of games) at the final curriculum stage, or when it runs out of `--iterations` (or `--time-budget`). The iterations and seconds each variant needed are recorded. With `--extend-iterations` (and `--extend-hours`), a variant that runs out of `--iterations` without meeting the criterion is not cut off: once every variant has had its first budget, the slow ones keep training, with the same population or weights, until they meet the criterion or exhaust the extension, so the record shows how long a cold start really needs and whether its final network competes. Then it runs the tournament: every champion plays the same 75 seeded games (seeds `50000 + i`) as the learner, greedily, in 6 slots against voting opponents on the base config.

```bash
python experiments/run_comparison.py --iterations 20 --games 75 --workers 4
python experiments/run_comparison.py --methods imitation,genetic,reinforce,ppo --pairs --curriculum --iterations 200 --workers 4
python experiments/run_comparison.py --methods imitation,genetic,neat,reinforce,ppo --pairs --curriculum --iterations 150 --extend-iterations 1000 --extend-hours 2 --workers 6
python experiments/run_comparison.py --methods imitation,reinforce,ppo --warm --curriculum --iterations 30
python experiments/run_comparison.py --methods ppo --sizes 16,64x32,128x64
python experiments/run_comparison.py --methods ppo --initializers xavier_uniform,he_uniform,zeros
python experiments/run_comparison.py --iterations 30 --until-win -1
```

The last line turns the criterion off and gives every variant the same fixed budget, the old behaviour.

**Warm against cold.** `--pairs` makes two variants of every reward or evolution method except NEAT: `<method>_cold` from random weights and `<method>_warm` from the imitation champion. That is the experiment behind the claim that pretraining pays: the report's "Warm start against cold start" table puts the two tournament win rates and the two iteration counts to the criterion side by side and names the winner of each pair. Say "warm-started PPO won 0.61 of tournament games against 0.40 cold, and reached the criterion in 38 epochs against 71" and the argument is made; say "PPO is better" and it is not.

**What `results.csv` contains.** One row per variant: `variant`, `method`, `iterations`, `train_seconds`, `final_mean_score`, `best_val_score`, `reached_criterion`, `iterations_to_criterion`, `seconds_to_criterion`, `final_val_win_rate`, `tournament_score`, `tournament_win_rate`, `tournament_survival`, `tournament_kills`, `lines_of_code`. `summary.json` adds the tournament dictionary and every learning curve; `results_table.tex` is the same table for a paper.

**What `report.md` contains.** The budget sentence, a ranking table by tournament win rate (score breaks ties), three bold lines (best in the tournament, simplest to implement by lines of code, fastest to train), the "Training to the win criterion" table (reached or not, iterations and seconds to get there, final validation win rate), the warm-versus-cold table when there are pairs, one note per method on why they differ, and the chart list.

**`score_by_method.png` versus `score_by_time.png`, and the win-rate pair.** The first plots mean score against iteration and answers "which method learns most per update". The second plots the same scores against `cumulative_seconds` and answers "which method learns most per minute of CPU". They can disagree: NEAT and the genetic algorithm play 48 genomes' worth of games per iteration and look strong per iteration but weak per second; imitation plays no training games at all and looks fast on both. The time chart is the one to show when the budget is compute. `win_rate_by_method.png` and `win_rate_by_time.png` are the same two views of `val_win_rate`, and they are the charts that show the criterion being reached: a line crossing 0.5 and staying there. `curriculum_by_method.png` shows how many opponents each variant faced at each iteration, so a score dip can be matched to a promotion. `validation_by_method.png` is the honest version of the first chart, and `entropy_by_method.png` and `length_by_method.png` show whether the methods reached the same confidence and survival.

**Why 75 tournament games.** The Battleship video compares strategies head to head with win rates and asks "how much is one turn worth". Its lesson is that single games are coin flips: two strategies that differ by a real few percent look identical, or reversed, over ten games. The tournament win rate is game-level, one flag per game, so 75 games are 75 samples of it. The standard error of a 50 percent win rate over 75 games is about 6 percent, so a gap of a dozen points is real and a gap of five is not; over 10 games nothing is. The score, survival and kill columns average 450 learner episodes (75 games times 6 copies) and are steadier. The same video's second lesson is that one extra win is worth a lot at the top: a champion that wins 60 percent of games instead of 50 is a different tribute, though the two bars look close. Read `tournament_win_rate.png` with that in mind, and `tournament_mean_score.png` for the finer-grained ranking, since the return moves with survival, kills and placing, not just wins.

**Sizes and initialisers.** `--sizes 16,64x32,128x64` makes one variant per hidden-layer layout of the same method; `--initializers` makes one per weight initializer. The tournament plays every champion on the same base config, so the only difference between the rows is the network. A bigger network that wins the tournament but loses `score_by_time.png` is the usual finding. `zeros` is the control that shows an initializer matters at all: every unit starts identical and symmetric gradients keep them so.

## How to answer the research question

The question is "which way of training a brain makes the most sense for the Hunger Games, and which is easiest to implement?" The comparison gives four numbers per method: tournament win rate, iterations and seconds to the win criterion, and lines of code. Put them side by side and the argument writes itself.

| Method | Lines of code (files counted) | Why it is slower or worse |
| --- | --- | --- |
| Imitation | 525 (`imitation.py`) | Cannot exceed its teacher. It minimises cross-entropy against the voting brain, never survival or wins, so its ceiling is "as good as the video's rules" |
| Genetic algorithm | 757 (`genetic.py`) | Scales badly with weight count. Mutating 5,872 weights at random finds improvements slowly; each generation is 48 genomes times `rounds_per_generation` games for one update |
| NEAT | 981 (`neat.py` and `brain/neat.py`) | Slow per generation and starts tiny. A minimal genome has no hidden nodes and must grow them one mutation at a time; speciation, fitness sharing and stagnation add machinery to get right |
| REINFORCE | 753 (`reinforce.py`) | High variance. One gradient step per batch of 4 episodes, with returns that swing on whether a learner won the bloodbath; curves are jagged and can regress |
| PPO | 894 (`reinforce.py` and `ppo.py`) | Reuses data but has more knobs. Several clipped passes per batch make it the most stable reward method; `clip_ratio`, `update_epochs`, `minibatch_size` and `gae_lambda` all need choosing |

**Which makes the most sense.** For this game the reward methods win on the numbers, and PPO warm-started from imitation is the combination to recommend: imitation supplies instincts in minutes (a fresh network dies of thirst on day three; after imitation 2 of 12 learner deaths were dehydration instead of 10 of 12), and PPO improves on them with a stable, sample-efficient update against the fixed voting field. The `--pairs` run is what proves the warm start: the warm twin should reach the win criterion in fewer iterations and win more tournament games than the cold twin. The genetic algorithm is the right choice when there is no reward function to write or the brain has a handful of genes (it can tune the voting brain's eight); NEAT is the choice when the question is what shape the network should be. REINFORCE is the teaching baseline that PPO is measured against.

**Which is easiest to implement.** By lines of code, imitation: one loss, one optimiser, no game logic beyond recording a teacher. The genetic algorithm is next and needs no calculus at all. PPO is only 141 lines more than REINFORCE because it inherits the collection, value network, validation and curriculum. NEAT is the largest because the network itself is a data structure that has to be evolved.

**Say it with the files.** The ranking and the criterion table in `report.md`, `plots/tournament_win_rate.png` and `plots/tournament_mean_score.png` for performance, `plots/win_rate_by_time.png`, `plots/train_seconds.png` and `plots/score_by_time.png` for cost, `plots/lines_of_code.png` for effort. Repeat the run with two more `--seed` values before writing the sentence; one seed is an anecdote.

## Answers for reviewers

### Which training method

Five trainers share the same brain input, the same menu, the same opponents and the same record shape. The intended order is imitation first, then a reward method from the imitation champion; the table in "The five methods" gives the mechanics. Per method:

| | Imitation | Genetic | NEAT | REINFORCE | PPO |
| --- | --- | --- | --- | --- | --- |
| Signal | Cross-entropy of the teacher's action | Episode return (fitness) | Episode return, shared within species | Discounted return minus a learned value, normalised | Same, with GAE (`gae_lambda` 0.95) |
| Update | Adam, batches of 256, 30 epochs over about 40,000 decisions from 12 teacher games | Elites 10 percent, tournament size 3, crossover 0.5, mutation rate 0.1 and scale 0.1 | Species with survival threshold 0.3, stagnation 15, target 8 species, crossover 0.75 | One Adam step per batch of 4 episodes, gradient clipped at norm 5, entropy bonus 0.01 | 4 passes of minibatch 256 with ratio clip 0.2 |
| Validation | Held-out 20 percent of demonstrations, plus 1 greedy game on seed 90000 | Champion, 2 greedy games on seeds 90000 and 90001 | same | same | same |
| Champion | Lowest validation loss | Best fitness ever | Best fitness ever | Best validation return | Best validation return |
| Works on | The neural brain | Any genome, including the voting brain's 8 genes | NEAT genomes | The neural brain | The neural brain |

**Warm starts.** Every trainer accepts `initial_genome`. REINFORCE, PPO and imitation load it into their network; the genetic algorithm builds its population as the genome plus relatives perturbed by a quarter of the mutation scale; NEAT clones a NEAT genome and mutates the copies. In the dashboard the "start from the current champion" checkbox passes the previous run's champion into the next run; in the comparison, `warm_from` does it, and `run_comparison.py --pairs` builds a cold and a warm variant of each method so the two can be compared on the same seeds.

### How reward points are gained and lost

From `RewardConfig` in [../config.md](../config.md), attached to the decision made that tick and used by every method's score:

| Event | Points |
| --- | --- |
| Surviving a tick | `+0.01` |
| Losing health | `-2.0` per full bar lost (so a 0.1 wound costs 0.2) |
| Restoring thirst or hunger while the bar was under half | `+0.5` per full bar restored |
| Moving closer to water while thirsty, or to grass while hungry with no food | `approach` per cell, default `0.0` (off) |
| A kill | `+1.0` |
| Dying | `-3.0`, once |
| Finishing the game | `+2.0` scaled by placing: first gets it all, last gets nothing |
| Winning | `+5.0` on top |
| Discount | `0.98` per tick when summing future rewards into a return |

The placement and win bonuses are added to the last decision of the episode. The `approach` term is a dense shaping reward that is off by default, because warm-starting from an imitation champion is the preferred way to give a fresh network its instincts.

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

The output is a menu of 16 items: rest, drink, eat, hunt, pick up, heal, attack nearest, flee nearest, and eight compass moves. Hidden layers default to two tanh layers of 64 and 32 units, 5,872 weights. A single layer of 16 (1,088 weights) copies the teacher 64 percent of the time after imitation; the default reaches 80 percent. A NEAT genome starts with the 50 inputs and a bias wired straight to the 16 outputs, 816 connections, and grows from there.

### How many games back each number

| Number | Games behind it, with the defaults |
| --- | --- |
| One imitation epoch's accuracy | No new games: one pass over 80 percent of about 40,000 recorded decisions; validation on the other 20 percent |
| One imitation epoch's `val_score` and `val_win_rate` | 1 fixed-seed game with 6 students, greedy; the win rate is 0 or 1 |
| One REINFORCE or PPO epoch's `mean_score` | 4 episodes times 6 learners, 24 learner episodes; `win_rate` is over the 4 games |
| One GA generation's `mean_score` | 48 genomes times 2 rounds, each a game with 6 learner copies; `win_rate` is over those 96 games |
| One NEAT generation's `mean_score` | 48 genomes times 1 round; `win_rate` is over those 48 games |
| Any method's `val_score` | 2 fixed-seed games times 6 learners (imitation: 1 game); `val_win_rate` is over the 2 games, so 0, 0.5 or 1 |
| The win criterion | The mean of `val_win_rate` over 5 iterations: 10 validation games (imitation: 5) |
| One tournament row | 75 games; score, survival and kills average 450 learner episodes, the win rate averages the 75 games |
| One sweep row | `games_per_value = 50` games, every tribute measured |

Raise `--episodes`, `--rounds`, `--games`, `--window` or `validation_games` before trusting a small difference.

### What the dashboard is built with

A custom window in Dear PyGui (`hunger_games/ui`), modelled on the zombie video's training view: score bars (one per episode of the latest iteration), an event monitor, the average score, entropy and average game length graphs, learning statistics with a rollout progress bar, CPU and memory from `psutil`, a gene plot with changed weights in gold, seconds per iteration, and Start, Pause, Stop, Reset and Watch agent buttons. Live charts are Dear PyGui's own plot widgets; every exported chart is a matplotlib PNG from `plots.py`. There is no TensorBoard and no Weights and Biases; the equivalent of a run log is `learning.json`, `history.json` and `events.txt` in each run folder.

### Reproducing a run folder

From the command line, at the repo root:

```bash
python experiments/run_ga.py --brain neural --population 48 --generations 20 --rounds 2 --workers 4 --seed 0
python experiments/run_rl.py --epochs 30 --episodes 4 --learners 6 --lr 1e-3 --entropy 0.01 --workers 4 --seed 0
python experiments/run_sweep.py --parameter chaos --values 0,0.25,0.5,0.75,1 --games 50 --workers 4 --seed 1000
python experiments/run_comparison.py --iterations 20 --games 75 --workers 4 --seed 0
```

Each writes `results/<name>_<timestamp>/` with `config.json`, then `history.json`, `learning.json`, `events.txt` and `champion.json` for a training run, `results.csv` and `summary.json` for a sweep or a comparison, and `plots/`. A comparison also writes `results_table.tex`, `report.md` and one run folder per variant under `runs/`. The `--seed` values make the population, the episode seeds and the games repeatable. A comparison's `config.json` records `until_win_rate` and `win_window`, so a reader can tell whether a variant stopped at the criterion or at the ceiling.

From the dashboard (`python -m hunger_games ui`): on the Train tab pick the method (imitation, genetic, neat, reinforce or ppo), tick the curriculum and "start from the current champion" if wanted, set the numbers, press Start, and when it finishes type a name and press "Save run folder". The run folder's `config.json` records the method and the trainer settings. A training run started from the dashboard plays on the painted map and roster, so mention that in a paper.

## A recipe for one figure set

1. Train with fixed seeds: `python experiments/run_rl.py --epochs 40 --episodes 6 --workers 4 --seed 1 --name paper_rl`.
2. Open `results/paper_rl_<timestamp>/plots/`. Take `score.png`, `game_length.png` and `entropy_shared.png` for the performance and stability figures.
3. Take `action_distribution_over_training.png` for the headline behaviour figure. It is the one chart that shows random turning into structured.
4. Take `instinct_curves.png`, `fight_or_flight.png` and `proximity_vs_remaining.png` for the three instincts: needs, danger, field size.
5. Take `armed_vs_unarmed_heatmaps.png` to show that the ring layout is being used as designed.
6. Run `experiments/run_comparison.py --pairs --curriculum` for the method figure: `plots/win_rate_by_time.png`, `plots/tournament_win_rate.png`, `plots/lines_of_code.png`, and the criterion and pair tables from `report.md`.
7. Repeat with two more seeds and report the mean and spread of the final validation score and win rate. One seed is an anecdote.
8. For a baseline, run the same behaviour charts on the untrained voting brain with a one-value sweep, `python experiments/run_sweep.py --parameter chaos --values 0.5 --games 50`, and read `plots/behaviour/`.

## Comparing two brains fairly

- Use the same seeds. `Game` derives every game's seed from `config.seed + game_id`, so two batches with the same seed and settings play the same arenas. The tournament does exactly this with seeds `50000 + i`.
- Measure the same tributes. Put the trained genome into the roster through a `Scenario` and set `brain_name` for the rest, or track ids with `tracked_ids`.
- Use the same number of games. Entropy and the instinct curves are ratios, but the heatmaps and histograms are counts, and `plots.heatmap` normalises by total time, which differs when tributes live longer.
- Compare validation, not training. Training numbers are sampled at temperature 1 and drift with the entropy bonus and the curriculum stage; validation is greedy on fixed seeds on the full roster.
- State the field. A learner's win rate depends on who else is in the arena. The default opponent is the voting brain at `chaos=0.5`.
- State the starting point. A warm-started run and a run from random weights are different experiments; say which champion seeded it. The `_cold` and `_warm` names in a pairs run do this for you.
- State what a win is. A game-level win (any of six copies is the victor) and a per-copy win are different numbers by a factor of up to six. This project reports game-level wins everywhere.

## What the raw summary lets you compute

| Quantity | How |
| --- | --- |
| P(attack given someone in sight) by health | `combat_by_health[:, 0] / combat_by_health.sum(axis=1)` |
| Median survival after injury | `np.median(post_injury_ticks)` |
| Share of decisions spent resting when healthy | `action_by_health[4][0] / sum(action_by_health[4])` |
| Fraction of time in the centre 10 by 10 heat cells | slice `position_heat[10:20, 10:20]` and divide by the total |
| Deaths by cause as shares | divide each `deaths_by_cause` value by `death_count` |
| Placement histogram of learners | `np.bincount(placements)` (survivors show as 0, see the limitations) |
| Score gap between two comparison variants | subtract their `tournament_score` rows in `results.csv`; the per-episode spread is in `summary.json["learning"]` |
| Iterations saved by a warm start | subtract the `_warm` row's `iterations_to_criterion` from the `_cold` row's in `results.csv` |

## Watching it live in the dashboard

The Train tab draws the same panels for every method, fed by `IterationStats`: the latest scores as bars, the event monitor, average score (mean, validation, best), entropy, average game length, the learning statistics line with the rollout bar, the system line, the gene plot and seconds per iteration. In the right-hand panel the Network tab draws a fixed-shape network or a NEAT genome as a graph, and the Charts tab shows the action distribution, the instinct curves and the position heatmap of every game watched in the session, updated while a game plays. None of these live panels are the paper charts; press "Save run folder" on the Train tab or "Export behaviour charts" on the Research tab for those.

Two differences from the PNGs: the live heatmap is scaled to its brightest cell rather than to the total, and the live instinct panel shows flee against health where the PNG shows heal.

## Checklist before claiming a brain has learned

1. Validation score rises over the run, on fixed seeds, against the untrained default brain. For an imitation run on its own, validation accuracy rises and validation loss falls; that shows copying, not learning to win.
2. The stacked area over training changes shape. If every band keeps its width, the policy has not moved.
3. At least one instinct curve is steep: P(drink given thirst) or P(eat given hunger) above 20 percent in the lowest bin and near zero in the highest.
4. Fight or flight crosses over: flee above attack when health is under 40 percent.
5. Policy entropy is above 0.5 nats at the end. Lower than that and the charts describe one habit, not a policy.
6. The same shape appears with a second seed.
7. The measured tributes are the learners. Check the "Who is measured" table above.
8. If the run was warm-started, the final numbers beat the champion it started from, not just a random network.
9. If the claim is "method A beats method B", it comes from the tournament with the same seeds, not from two training curves with different budgets.
10. If the claim is "it wins", it means a majority of validation games at the final curriculum stage (the comparison's criterion), and the win rate quoted is game-level.

## Known limitations

- **Small numpy networks.** The brain is a plain MLP written in numpy, two hidden layers of 64 and 32 by default. There is no GPU path and no recurrence, so the tribute has no memory beyond what the perception carries. NEAT genomes compile to a numpy evaluation plan (about 30 microseconds per decision), but a NEAT generation still plays 48 games.
- **REINFORCE variance.** Returns are noisy, rewards are sparse at the end of an episode, and only 4 episodes with 6 learners feed each update. Expect jagged curves. PPO is steadier but not immune.
- **Imitation is capped by its teacher.** The student learns the voting brain's rules and its mistakes. Its validation games measure survival and wins, but the loss it minimises never sees them; a better copy is not always a better tribute.
- **The curriculum changes the score's meaning.** Mean score drops at every promotion because the field grows. Compare `val_score`, which is always on the full roster, or turn the curriculum off for a clean curve.
- **The curriculum can stall, on purpose.** With no timeout a learner that never wins half of its validation games stays on stage 0, and the comparison's criterion is never tested for it.
- **Win rates are coarse per iteration.** Two validation games give 0, 0.5 or 1. The curriculum and the criterion average five iterations, which is still only ten games. Raise `validation_games` when the win rate is the claim.
- **The tournament is one field.** Every champion fights voting opponents at `chaos=0.5` on the base config. A champion that beats that field may not beat a different brain, and champions never fight each other.
- **Lines of code is a rough measure.** It counts whole files, comments included, and NEAT's total includes its genome module. It says how much there is to read, not how hard it was to get right.
- **Telemetry measures the voting brain too** in sweeps and in the GA's `"self"` mode. For a clean comparison of a trained brain, load its genome into every tribute first.
- **Heatmaps are 30 by 30 bins.** On a 120-cell arena each bin covers 16 cells. Change `HEATMAP_CELLS` in `telemetry.py` if you need more, and note that the dashboard's heat series expects 30.
- **Placements in telemetry.** The `placements` list records 0 for survivors because the hook runs before final placings are assigned. Use `wins` and the runner's `players.csv` for exact placings.
- **One seed per comparison run.** `run_comparison.py` takes one `--seed`; the spread between seeds has to be measured by running it again.
