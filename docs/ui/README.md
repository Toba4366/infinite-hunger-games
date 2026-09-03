# Using the game makers' dashboard

The dashboard is a desktop window for designing an arena, picking the tributes, watching a game play out tick by tick, training one brain against the voting strategy and watching it train, and measuring what the tributes do. It is built with Dear PyGui. This page is the reference for every control. For a guided first run with pictures taken from the real dashboard, read the [illustrated walkthrough](../tutorial/README.md), which is the written version of the Tutorial tab. The code is explained in [init.md](init.md), [main.md](main.md), [painter.md](painter.md), [session.md](session.md), [canvas.md](canvas.md), [visualizer.md](visualizer.md), [app.md](app.md) and [screenshots.md](screenshots.md).

## Launching

From the project root (the folder that contains `hunger_games/`):

```text
python -m hunger_games ui
```

`python -m hunger_games.ui` does the same thing. The window opens at 1500 by 920 pixels (it can be resized down to 1100 by 700) with a generated arena and a roster of 24 tributes on their podiums, on the Tutorial tab. A readable system font is used when one is found; the theme is dark with crimson buttons and gold sliders.

## The three panels

| Panel | Where | What it holds |
| --- | --- | --- |
| Controls | Left | The four mouse tools and nine tabs: Tutorial, Setup, Map, Loot, Tributes, Brains, Play, Train, Research |
| Arena | Centre | The map, with the transport bar (play controls) underneath |
| Analysis | Right | Three tabs: Inspector, Network, Charts |

The panels resize with the window: the left and right panels take 27 % of the width each and the arena takes the rest. The status line at the top right of the title row reports every action, including errors from file dialogs. Most controls have a tooltip; hover to read it.

## The four mouse tools

The radio buttons at the top of the left panel choose what the mouse does on the arena.

| Tool | Left button | Right button |
| --- | --- | --- |
| Select | Click a tribute to select them | Same |
| Paint terrain | Hold and drag to paint with the brush; a white ring shows the brush | Click a tribute to select them |
| Place loot | Click to place a stack of supplies | Click to remove the stack there |
| Move tribute | Press on a tribute and drag their podium | Click a tribute to select them |

Painting, placing loot and dragging only work while no game is loaded. After a game, Load replay or a restart clears it; painting still applies to the next New game. Podiums are nudged onto legal ground after every stroke, so a tribute standing where you painted void moves to the nearest walkable cell.

## Reading the arena

Female tributes are circles and male tributes are squares, filled with their district's colour and labelled `D4F`, `D4M` and so on (turn labels off on the Map tab). A gold star on a tribute means it is driven by the learner brain: the network being trained, or a champion you handed out. Weapons are red triangles, food a white dot, medicine a magenta plus. A yellow ring marks the selected tribute. While a game plays, a white parachute appears over anyone receiving a sponsor gift, a red X flashes where someone was just eliminated, and a red ring shows the game makers' safe circle when it is closing. While editing, only the stacks you placed by hand are drawn; the layout's own supplies appear once a game starts.

## Tab by tab

### Tutorial

The first tab is a walkthrough in ten folding steps, the same steps as the [illustrated walkthrough](../tutorial/README.md). The first two are open when the dashboard starts. Each step explains one part of the dashboard, and all but the first have a "Show me" button that performs the step for you with the same code the real controls use, then opens the tab where it lives.

| Step | Show me does |
| --- | --- |
| Welcome | Nothing to press; explains the panels and the buttons |
| 1. Build an arena | Loads the `lake_island` preset, moves tributes off the water, opens Map |
| 2. Paint terrain | Selects the Paint terrain tool, opens Map. Drag on the arena yourself |
| 3. Edit the tributes | Selects the first tribute, opens Tributes with the editor filled in |
| 4. Place loot | Selects the Place loot tool, opens Loot. Click the arena yourself |
| 5. Play a game | Selects the Select tool, starts a new game at 8 ticks per second, opens Play |
| 6. Inspect and watch a network think | Sets the default brain to `neural`, gives it to every tribute (dropping any trained genomes), starts a game at 4 ticks per second, selects the first tribute, opens Brains on the left and Network on the right |
| 7. Train and watch training | Switches the Train tab's method to `imitation`, sets the training feed to `live`, and starts a fixed short imitation run: 2 demonstration games, 4 epochs, 1 validation game, 6 learner slots. Opens Train |
| 8. Research | Opens Research |
| 9. Save and share | Opens Play, where the replay and GIF buttons are; the other files are on Setup, Map and Train |

Step 7 uses its own fixed numbers, never warm-starts and uses no curriculum; the Train tab's Advanced settings are left as you set them. It does nothing while another run is already going.

### Setup

Four folding sections and the buttons under them. Changing shape, size, layout or tribute count updates the arena at once.

| Section | Control | Meaning |
| --- | --- | --- |
| Arena | shape | `open_field` (the 74th games' forest) or `round` (the 75th games' circle). Regenerates the map and moves tributes off the void |
| | loot layout | `cornucopia` (one pile in the middle) or `ring` (cheap supplies at the edge, weapons in the centre, the video's redesign). Re-places the podiums |
| | size (cells) | 30 to 300, square. Press Enter; the map is regenerated |
| | seed (-1 = random) | The same seed and settings replay the same game |
| | chaos | 0 = no luck in hunting or fights and brains always pick their favourite action; 1 = very random |
| | max days, ticks per day | Game length. The games are a draw if more than one tribute is alive at the cutoff |
| Tributes and starting bars | tributes | 2 to 96. Press Enter; a new roster is rolled |
| | min thirst, min hunger, min health | Each tribute starts with every bar drawn between the minimum and full |
| | Everyone starts full, Random above 0.5 | Set all three minimums to 1.0 or 0.5 |
| | vision radius, landmark radius | How far tributes see people, and how far they spot lakes and meadows |
| | days to die of thirst, days to starve | How fast the bars drain |
| What tributes know | cannon and nightly sky | On: tributes know how many remain, how strong they are on average, the strongest left and their own rank. Off: only the count |
| | endgame instinct | On: bold tributes head for the centre once fewer than half remain. The tooltip quotes 20 test games: instinct alone 18/20 victors, circle alone 19/20, neither 0/20 |
| Sponsors and game makers | sponsor gifts, gift chance / day | Parachutes of medicine, food or water for tributes in need, weighted by training score, career district and kills |
| | game maker circle | The safe circle that slowly closes after a quiet day. Tributes see it coming |
| | quiet days before it, days to close | When it starts and how gently it closes |
| | podiums may stand in water | Allow podiums in the sea, as in the 75th games |
| Buttons | Regenerate arena, New roster | A fresh Perlin map; new names, scores, brains and podiums |
| | Save config, Load config | The settings as JSON, including the network architecture and the reward function. Loading refreshes every Setup and Brains widget and regenerates the map |

### Map

| Section | Control | Meaning |
| --- | --- | --- |
| Brush | terrain | void, water, sand, grass or rock. Void is outside the arena; nobody can enter it |
| | brush radius | 0 paints one cell; 20 is a wide brush |
| Presets and stamps | preset, Load preset | `perlin` (generated hills), `flat_field`, `flat_round`, `quarter_quell` (the 75th games' island in a sea with twelve rocky spokes), `lake_island` |
| | Carve round | Void everything outside the largest circle |
| | Fill grass, Fill water | Paint the whole map one type |
| | stamp radius, Circle at centre, Square at centre | Stamp the brush terrain in the middle |
| | show tribute labels | The `D4F` labels |
| | coverage | The share of the arena that is water, sand, grass and rock |
| Scenario files | Save scenario, Load scenario | The map, the hand-placed loot and the roster in one JSON file |

### Loot

| Control | Meaning |
| --- | --- |
| kind | food, weapon or medicine |
| quantity | 1 to 20 items in the stack |
| quality | 0 to 1. For weapons the line under it names the tier: fists 0, rock 0.2, knife 0.4, spear 0.6, sword 0.8, bow 0.9 |
| also scatter the layout's loot | Untick to play with only your hand-placed stacks |
| Clear hand-placed loot | Remove every stack you placed |

Choose the Place loot tool and click the arena. Loot cannot be placed on void. The line at the bottom counts your stacks.

### Tributes

At the top, a podium preset and "Arrange podiums": `edge ring` (along the outer edge, the video's redesign), `around cornucopia` (a tight circle round the middle), `random`, or `two sides` (two opposing camps). Every preset spreads the strongest tributes apart, as the video's chapter 2 suggests. Or drag tributes with the Move tool.

The table lists the roster; click a name to select. A `*` after the brain name means the tribute carries a trained genome. Add tribute appends a middling tribute at the centre of the map; Remove selected removes the selected one.

| Editor control | Meaning |
| --- | --- |
| name, district, sex | Rename them; 1 Luxury to 12 Mining; F or M |
| training score | 1 to 12, as the game makers rate tributes. Raises sponsor favour and podium spacing |
| survival score | 0.05 to 0.95, hunting aptitude against terrain difficulty |
| brain | voting, random or neural. A tribute given a NEAT champion shows `neat` in the table, but picking anything in this combo replaces it |
| granted weapon | A weapon they start with, by quality (0 = none) |
| granted food, granted medkits | Items in their pack at the start |
| sponsor favour bonus | Extra favour, so sponsors send gifts sooner |
| start thirst, start hunger, start health | Their own starting bars; 0 means use the Setup minimums |
| Forget trained genome | Drop a champion genome so the brain starts fresh (and lose the star) |

### Brains

| Control | Meaning |
| --- | --- |
| default brain | voting (the video's instinct-voting brain), random (a baseline) or neural (the network below, untrained until you train it). Used by New roster and Add tribute, and by the opponents in training games |
| Give this brain to every tribute | Set everyone to that brain and drop any genomes |
| number of hidden layers | 1 to 6. A hidden layer is a row of neurons between the 50 inputs and the 16 outputs. The default is 2. Changing the number adds or removes the fields below at once |
| nodes in hidden layer 1, 2, ... | One field per layer, 1 to 512 nodes each. The defaults are 64 and 32. New layers start at 16; widths you typed are kept when you change the count |
| activation | tanh, relu, leaky_relu, sigmoid or selu. tanh pairs with Xavier, relu with He, selu with LeCun |
| initializer | How starting weights are drawn; the note under it explains each one |
| init scale, sparsity | Used by the constant, uniform and normal initializers, and by sparse |
| Apply network settings | Read the fields into the settings and show the shape and parameter count, e.g. `Network: 50 -> 64 -> 32 -> 16, tanh, xavier_uniform, 5872 params` for the defaults |

The width fields change nothing until you press Apply network settings; the Network tab's caption shows the architecture in force. The folding section "Inputs (50) and outputs (16)" lists the perception vector in order and the 16 actions (rest, drink, eat, hunt, pick_up, heal, attack, flee and eight moves). The brain takes the highest score, or a softmax sample when chaos is above 0. The NEAT method ignores this architecture: it grows its own.

### Play

| Control | Meaning |
| --- | --- |
| Slow-mo 2/s, Normal 8/s, Fast 40/s, Max 400/s | Playback speed presets |
| start a new game when this one ends (back to back) | Every finished game's behaviour is kept for the Charts tab and the Research exports |
| Save replay, Load replay | The recording as a `.replay` file. Only open replays you made yourself |
| GIF ticks per frame, Export GIF of this game | 1 is every tick (a long file); 2 is the default. Finishes the game first, then writes the file |

### Train

The Train tab trains **one network**. In every training game that network drives a few roster slots (6 by default), marked with gold stars on the arena, and every other tribute uses the voting brain from the video. The tab is laid out like the training dashboard in the PPO zombie-arena video: pick a method, press Start, and read the panels while it runs.

**The five methods.** The combo at the top lists them in the order a learner should try them; the line under the combo is the help text for the chosen one.

| Method | Help text | Genome |
| --- | --- | --- |
| `imitation` | Copies the voting brain's decisions (supervised). Start here: it gives the network instincts. | The neural network from the Brains tab |
| `genetic` | Evolves the weights of a population of networks; each plays as the learner against voting opponents. | The neural network, or the voting brain's 8 genes |
| `neat` | Evolves weights and the shape of the network, in species (the Monopoly video's method). | A NEAT genome that grows its own hidden nodes |
| `reinforce` | Policy gradient with a value baseline: every action is scored by the reward function. | The neural network |
| `ppo` | Clipped policy gradient with several passes per batch (the zombie video's method). The most stable reward method. | The neural network |

**Why imitation first.** A fresh neural network picks actions almost at random. In this arena that means it does not drink, so it dies of thirst on about day three, long before winning or losing games could teach it anything. Imitation fixes that by copying a brain that already works; the other methods then improve on the copy. The dashboard starts on `imitation` with "start from the current champion" ticked, so the natural order is: imitation, then `ppo` (or `reinforce`, `genetic` or `neat`) from the imitation champion.

**Options under the combo.**

| Control | Default | Meaning |
| --- | --- | --- |
| start from the current champion | on | The run begins from the last run's champion, or, with no earlier run, from a learner genome in the roster (for example one loaded with Load champion into all). Genetic and NEAT build their population from it; imitation, reinforce and PPO load it into their network. It only applies when the kinds match: a neural champion seeds the four neural methods, a NEAT champion seeds NEAT, and `genetic` with brain to evolve `voting` always starts fresh. When it applies the status line says "warm start" |
| curriculum: opponents grow 1, 3, 7, 11, 23 | off | Like the zombie video's one-to-sixteen ladder. The learner faces 1 opponent first (plus its own copies) and is promoted to the next stage when the mean score of its last 5 iterations reaches the promotion threshold (3.0), or after 40 iterations in a stage. The status line says "curriculum" and the event monitor reports each promotion |
| Training feed | `off` | What the arena shows while training runs, see below |

**Controls.**

| Button | What it does |
| --- | --- |
| Start | Begins a run with the method, the options and the Advanced settings. Disabled while a run is going |
| Pause / Resume | Holds the run between iterations (the current iteration finishes first). The progress bar says "(paused)" |
| Stop | Ends the run after the current iteration |
| Reset | Stops, forgets the trainer and clears every panel. Champions already handed to tributes stay on them |
| Watch agent | Gives the champion to the starred learner slots, starts a fresh game at 8 ticks per second and opens the Network tab. Nothing happens before the first iteration ("No champion yet: train first") |

Training runs in the background; the window stays usable.

**The panels, top to bottom, and what to read from them.**

| Panel | What it shows |
| --- | --- |
| Progress bar | Games done in the current iteration. The overlay reads "iteration N" while running and "N iterations done" after |
| Summary line | The last iteration: method, count, mean score, validation score, win rate, then up to four method numbers (imitation: train and validation loss and accuracy; reinforce and PPO: policy and value loss; genetic: worst fitness; NEAT: species, hidden nodes, connections, compatibility threshold) |
| Latest scores (one bar per episode) | The score of every learner episode in the newest iteration. Genetic and NEAT: one bar per genome in the population. Imitation: the validation games. Reinforce and PPO: the collected games. A wide spread means luck still dominates; bars that all rise together mean the policy improved |
| Event monitor | The last 14 timestamped events: `rollout` (an iteration's numbers), `evolution` (a generation's species and bests), `record` (a new best), `curriculum` (a promotion), `info` (demonstrations collected). The full log goes to `events.txt` in the run folder |
| Average score | Three lines per iteration: the mean score of the learner's episodes, the validation score (greedy games on fixed seeds, the honest number), and the best episode. Scores are returns under the reward function for every method but imitation's accuracy-based extras; a rising validation line that tracks the mean means real learning |
| Entropy (lower = more confident) | The policy entropy in nats. It should fall slowly. Falling to 0 means the policy collapsed onto one action; for reinforce and PPO raise the entropy bonus. For genetic and NEAT this is the action entropy measured in that generation's games |
| Average game length (learner survival) | Mean ticks the learner survived. A fresh network dies of thirst early; after imitation the line jumps |
| Learning statistics | Two lines: iteration, seed, seconds per iteration (mean of the last 5), max score so far, learning time in seconds; then the curriculum stage and opponent count, the mean score, entropy and mean length of the last iteration |
| rollout bar | The same fraction as the progress bar, with "rollout done/total games (percent)" |
| CPU, memory, GPU | psutil readings of the process (zeros without psutil). GPU always reads "not used (numpy on the CPU)" |
| Learner genes (gold = changed since last step) | The learner after the newest iteration as bars, gold where a value moved since the iteration before. For the voting brain the eight genes are named on the axis; networks show the first 400 weights; NEAT shows its connection weights, all gold after a structural change |
| Seconds per iteration | One bar per iteration, so you can budget a run |

**Advanced settings.** A closed section at the bottom; only the chosen method's group is shown, plus the reward function (inside the reinforce and PPO group) and the curriculum settings.

| Method | Controls (default) |
| --- | --- |
| Imitation | teacher brain (`voting`, the only choice), demonstration games (12; about 40,000 decisions), learn only from the top N placings (0 = all; set it to 3 to learn from winners), epochs (30), batch size (256), learning rate (0.001), validation games (1), CPU workers (1). The first epoch records the demonstrations, so it is slower and its progress bar counts those games |
| Genetic | brain to evolve (`neural` or `voting`), opponents (`voting`: each genome is the learner against the voting brain, scored by return; `self`: the population plays itself, scored by placement, the original tournament), population (48), generations (20), games per genome (2), elite fraction (0.1), mutation rate (0.1), mutation scale (0.1; about 0.02 after imitation so the copied instincts are not erased), crossover rate (0.5), validation games (2), CPU workers (1). The curriculum applies only with `voting` opponents |
| NEAT | population (48), generations (30), target species (8; the compatibility threshold adjusts to reach it), add node rate (0.03), add connection rate (0.08), validation games (2), CPU workers (1). NEAT starts from minimal genomes (inputs wired straight to the outputs) unless warm-started from a NEAT champion |
| Reinforce and PPO | epochs (30), games per epoch (4), learner copies per game (6), learning rate (0.001), entropy bonus (0.01), PPO clip ratio (0.2) and PPO passes per batch (4) (both used by `ppo` only), validation games (2), CPU workers (1) |
| Reward function | per tick alive (0.01; the zombie video's lesson: reward survival too much and the learner just runs away), win (5), death (-3), kill (1), per health lost (-2), per need restored (0.5, only while the bar was below half), approach water/food (0, a dense shaping reward, off by default because imitation is the preferred way to give instincts), placement (2, scaled by placing), discount (0.98). These edit the config itself and are saved by Save config |
| Curriculum settings | opponents per stage (`1,3,7,11,23`), promotion threshold (3.0), max iterations per stage (40) |

**Training feed.** The radio button decides what the arena shows while training runs.

| Feed | What the arena shows |
| --- | --- |
| `off` | Nothing; training only fills the panels (the default) |
| `replay` | After every iteration, a recording of one real game from that iteration, on the painted map exactly as the trainer saw it: imitation's greedy validation game, the first evaluation game of a genetic or NEAT generation, the first collected game of a reinforce or PPO epoch. The learner's slots carry stars |
| `live` | After every iteration, a fresh game in which the newest champion drives the learner slots (starred) while the other tributes keep their brains. Because it is a live game, the Network tab shows real activations |

The next iteration is shown only when the arena is free: nothing is loaded, or the replay has reached its last frame, or the live game is over and you have watched to its end. Iterations that finish while you are still watching are skipped, and the newest one is shown next. The headline under the arena starts with "training feed: replaying a real generation 3 game" (genetic) or "training feed: epoch 3 champion playing live" (every other method) while the feed is on. The feed plays at the current ticks per second, so pick Fast or Max if training is quicker than playback. `replay` replaces the roster and the settings with the training game's, so save your scenario first; `live` writes the champion into the learner slots of your roster.

**Champions and files.** Champion to all and Champion to selected hand the champion out (with the brain kind it was trained as: `neural`, `voting` or `neat`), and the tributes get stars. Save run folder writes `results/<name>_<timestamp>/` with `config.json`, `history.json`, `learning.json` (the unified curves), `events.txt`, `champion.json` and a `plots/` folder of PNGs. Save champion and Load champion into all use a JSON champion file; every method writes the same shape (a neural file carries the architecture, a NEAT file the genome dictionary), and a loaded file counts as a champion for the next warm start when no trainer exists.

### Research

**Parameter sweep.** Pick a config field (nested ones use a dot, like `terrain.water_threshold`), type comma-separated values (booleans as `true`/`false`), set games per value, CPU workers and whether to collect behaviour telemetry, then Start sweep. Each value plays the same seeded games on the painted map. The results line shows, per finished value, the victor rate, mean days, and the share of player-versus-player and natural eliminations. The run folder `results/<parameter>_<timestamp>/` holds `config.json`, `results.csv`, `summary.json` and one plot per metric against the swept value.

**Charts of the games you have watched.** Every finished game watched this session (including back-to-back ones) is measured. Type a folder (default `output/watched`) and press Export behaviour charts to write twelve PNGs: action distribution, actions by thirst, hunger and health, instinct curves, consumption timing, fight or flight, proximity versus tributes remaining, actions by remaining, the position heatmap, armed versus unarmed heatmaps, and deaths by cause. Forget watched games clears the tally.

The last section, "Answers a reviewer will ask for", is a short fixed text: the method (imitation, genetic algorithm, NEAT, REINFORCE with a value baseline or PPO, chosen on the Train tab, with warm starts between them and an optional opponent curriculum; `experiments/run_comparison.py` trains them all under one budget and runs a 75-game tournament), the reward function (the Reward function section, with the dense approach reward off by default), the observation (a 50-value vector, not a grid), and the tooling (a custom Dear PyGui dashboard, matplotlib charts). The full written answers are in the [research guide](../research/README.md).

## The transport bar

| Control | What it does |
| --- | --- |
| New game | Start a fresh game from the current settings, map and roster, paused on frame 0 |
| Play / Pause | Run or pause. Starts a game if there is none |
| Step | One tick (starts a game if there is none) |
| To end | Simulate the rest of the game instantly, then scrub back to watch any part |
| Rewind | Back to frame 0 |
| ticks / second | Playback speed, 0.5 to 400 |
| frame | Scrub anywhere in the recording |

The headline shows the day, tick, how many are alive and the frame number, and on the last frame of a finished game the victor or "no victor (draw)". While the training feed is on, the headline starts with what the feed is showing. Every game is recorded as it plays, so you can scrub back, click a tribute to see their bars at that moment, then play on. Playing past the end of what has been simulated simulates more.

## The right panel

**Inspector.** Click a tribute (or a row in the Tributes tab). Before a game it shows district and industry, sex, scores, brain (with "(trained)" when it carries a genome), the granted items and the podium cell. During a game it shows the thirst, hunger and health bars at the current frame, whether they are alive or when and how they were eliminated, their weapon and reach, food, medkits, kills, sponsor favour and last action. The event log lists the most recent eliminations and parachutes up to the current frame.

**Network.** Select a neural tribute during a live game and its network is drawn as columns of nodes: the 50 named inputs on the left, the hidden layers, and the 16 named outputs with their probabilities on the right. Red nodes are positive activations, blue nodes negative, dark grey idle. Warm edges are positive weights, cool edges negative, and brighter means larger; only the six strongest edges into each node are drawn. The yellow output label is the action taken. Without a live neural tribute the tab shows the bare architecture from the Brains tab. The picture is redrawn every frame, so use Slow-mo or Step to follow one decision at a time.

Select a NEAT tribute (a starred tribute after a NEAT run, or one given a NEAT champion) and the same tab draws the NEAT genome as a graph instead: inputs on the left, hidden nodes in columns by their depth, outputs on the right, and every enabled connection as an edge coloured by its sign and brightened by its size. The caption over it counts the inputs, hidden nodes, outputs and connections, so you can watch the network grow between generations.

Above the drawing, the folding section "How the champion network changed over training" holds two plots that grow with every iteration, whether or not the tab is open:

| Plot | What it shows |
| --- | --- |
| Genome change per step (L2) and mean \|weight\| | Two lines by iteration: how far the learner moved from the previous iteration (the length of the difference vector; 0 at step 0), and the mean absolute value of its genes. A change line that settles toward 0 means the learner has stopped moving; a mean that keeps growing means the weights are getting larger |
| Champion genome by step | A heat map: one row per iteration, one column per gene (the first 200), red for positive and blue for negative on a scale that is symmetric about zero. Vertical streaks are genes that stayed put; a row that differs from the one above is an iteration that changed the learner |

For genetic and NEAT the rows are each generation's best genome (NEAT: its connection weights, and the plots restart whenever the champion gains a node or connection); for imitation, reinforce and PPO they are the network after each epoch.

**Charts.** Three live charts of every game watched this session, refreshed every 30 frames: the action distribution in percent; the instinct curves (how often the tribute drank, ate or fled at each level of the matching bar, which is how chapter 4's voting rules were tuned); and a heatmap of where tributes spend their time. Games shown by the `replay` feed are not counted; games from the `live` feed and Watch agent are, once they finish.

## Recipes

### Train a neural brain: imitation first, then PPO from it

This is the recommended way to get a neural tribute that survives and then improves.

1. Brains tab: keep the default network (number of hidden layers 2, nodes 64 and 32, tanh, xavier_uniform) or set your own, then Apply network settings. The summary reads `Network: 50 -> 64 -> 32 -> 16, tanh, xavier_uniform, 5872 params`.
2. Train tab: the method is already `imitation`. Open Advanced settings, leave the defaults (12 demonstration games, 30 epochs) and set CPU workers to 4 if you have the cores. Press Start. The event monitor reports the demonstrations collected, then one `rollout` line per epoch with the accuracy and loss; the summary line shows train and validation accuracy climbing. If validation loss turns upward while training loss keeps falling, press Stop.
3. Press Watch agent. The starred tributes now walk to water and drink. Click one and read the Network tab.
4. Keep "start from the current champion" ticked. Pick `ppo`, tick the curriculum, and press Start. The status line reads "Training (ppo, warm start, curriculum)...". The learner faces 1 opponent first; the event monitor prints "promoted to stage 1: 3 opponents" when its recent mean score clears 3.0, and so on up to 23. Read the Average score plot (validation is the honest line), the Entropy plot (falling slowly, not to 0) and the Average game length.
5. Save champion, or Save run folder for the curves. A saved champion loaded later with Load champion into all is picked up by the next warm start.

`reinforce` works the same way with one pass per batch and no clipping; it is noisier. `genetic` from the imitation champion needs mutation scale lowered to about 0.02. `neat` cannot start from a neural champion; see the NEAT recipe.

### Evolve a NEAT brain and watch it grow

1. Train tab: pick `neat`. Untick "start from the current champion" (a neural champion cannot seed NEAT; the box is ignored anyway, but unticking avoids confusion), tick the curriculum, set the training feed to `live`, Play tab: Fast 40/s. Press Start.
2. The event monitor prints "new species N founded", "generation N: S species, best B, mean M", "new champion: fitness F, H hidden nodes, C connections" and, over time, "stagnant species removed". The summary line's extras give the species count, hidden nodes and connections.
3. When the live feed shows a game, click a starred tribute and open the Network tab. The graph starts as 50 inputs wired to 16 outputs and gains hidden nodes as generations pass. The Learner genes bars turn all gold whenever the champion's shape changes.
4. Save champion writes a file with `brain_name: neat`; Load champion into all later gives it to everyone and a NEAT run with the champion box ticked continues from it.

### Compare genetic against the voting brain

1. Train tab: `genetic`, Advanced settings: brain to evolve `voting`, opponents `self`, population 24, generations 10, games per genome 1. Untick the champion box. Training feed `replay`. Start.
2. The Learner genes plot names the eight voting genes on its axis and turns gold where a generation changed them. As soon as generation 0 finishes, the arena replays one of its real games and the headline reads "training feed: replaying a real generation 0 game".
3. Champion to all gives every tribute the evolved voting genes; New game and Play to watch them.

### Paint an island and save it

1. Map tab: preset `flat_field`, Load preset, then Fill water.
2. terrain `grass`, stamp radius 35, Circle at centre. Then `sand`, radius 38, Circle at centre, then `grass` at 35 again to leave a sandy shore.
3. Choose Paint terrain, terrain `rock`, brush radius 3, and drag a few hills onto the island. The white ring shows the brush.
4. Tributes tab: Arrange podiums `edge ring` so the podiums sit on the shore, or Setup: New roster.
5. Map tab: Save scenario, name it `island.json`.

### Favour a tribute with a bow and a sponsor bonus, and drag them

1. Tributes tab: click a name. Type a new name, set granted weapon to 0.9 (the Loot tab's preview shows 0.9 is a bow) and sponsor favour bonus to 0.5.
2. Choose Move tribute, press on their marker and drag it to the podium you want.
3. Press Play and watch the Inspector while they play.

### Set starting bars

Setup tab: Random above 0.5 starts everyone somewhere between half and full; drag min thirst, min hunger and min health separately for finer control; Everyone starts full resets. For one tribute only, use start thirst, start hunger and start health on the Tributes tab.

### Watch training as it happens

1. Map tab: Save scenario first if you edited the roster, because the replay feed replaces it.
2. Train tab: any method, training feed `replay`. Play tab: Fast 40/s. Start.
3. After each iteration the arena replays one real training game; the starred tributes are the learner's slots. When it reaches its last frame the newest finished iteration replaces it.
4. Switch the feed to `live`. When the current replay ends, the newest champion is written into the learner slots and a fresh game starts. Click a starred tribute and open the Network tab to watch it think.
5. Open "How the champion network changed over training" on the Network tab to see how far each iteration moved the learner and which genes changed.

### Pause, inspect, resume

1. Start any run with the feed `off`. When the Average score plot has a few points, press Pause. The progress bar says "(paused)" after the current iteration finishes.
2. Press Watch agent to play the champion so far, or Champion to selected to give it to one tribute and play a normal game against the rest.
3. Press Resume. The run continues from where it stopped, with the same history and plots. Stop ends it; Reset clears the panels for a fresh run.

### Run a chaos sweep and export charts

1. Research tab: parameter `chaos`, values `0,0.25,0.5,0.75,1`, games per value 20, CPU workers 4, telemetry on, Start sweep.
2. The results lines fill in one value at a time. When the status line names the folder, open `results/chaos_<timestamp>/plots/` for one PNG per metric plus behaviour charts.

### Watch games back to back and export behaviour charts

1. Play tab: tick start a new game when this one ends, press Fast 40/s, then Play. Untick to stop.
2. Watch the Charts tab fill in as games finish.
3. Research tab: set a folder and press Export behaviour charts. The status line reports how many PNGs were written.

### Load a replay and export a GIF

1. Play tab: Load replay and pick a `.replay` file you saved earlier. The map it was played on appears, with the roster from the recording.
2. Scrub or play to check it. The Inspector works on replays; the Network tab does not, because a replay has no live brains.
3. Set GIF ticks per frame (2 is a good size), press Export GIF of this game and choose a name. Wait for the status line to say it was saved.

## Files the dashboard reads and writes

| File | Tab | Contents |
| --- | --- | --- |
| `config.json` | Setup | Every setting, including the network architecture and the reward function |
| `scenario.json` | Map | The painted terrain, the hand-placed loot and the roster with podiums, genomes and granted items |
| `game.replay` | Play | Every tick of a game, as a pickle |
| `game.gif` | Play | An animation of the recording, one frame per `GIF ticks per frame` ticks |
| `champion.json` | Train | The champion genome with its brain name. Neural champions carry the architecture; reinforce and PPO add the value network; imitation adds the teacher; NEAT stores the genome as a dictionary with `brain_name: neat` |
| `results/<name>_<timestamp>/` | Train | `config.json`, `history.json`, `learning.json`, `events.txt`, `champion.json`, `plots/` |
| `results/<parameter>_<timestamp>/` | Research | `config.json`, `results.csv`, `summary.json`, `plots/`, plus one `batches/<value>/` folder per value |
| `output/watched/*.png` | Research | The twelve behaviour charts of the games watched this session |
| `docs/tutorial/images/*.png` | none | The tutorial's pictures, written by `python -m hunger_games.ui.screenshots` (see [screenshots.md](screenshots.md)) |

Default names are offered by the file dialogs; the extension is added if you leave it off.

## Known limits

- The game makers' circle is drawn as a solid red ring, not the dashed circle the matplotlib renderer uses.
- Painting, placing loot and dragging podiums only work while no game is loaded. Load replay clears the live game (and shows the replay's map); otherwise restart the dashboard to edit again.
- Export GIF finishes the current game first, then writes the file, and the window is busy until it is done. A loaded replay is exported as it is.
- Replay files are Python pickles. Only open `.replay` files you made yourself, because loading a pickle can run code.
- Training and sweeps with more than one CPU worker use separate processes. This works from the dashboard on macOS, Windows and Linux; the dashboard's entry point is guarded so the workers do not open extra windows.
- Trainers and sweeps use the painted map but not the hand-placed loot or the edited roster. Opponents in training games are always the voting brain, whatever the Brains tab's default brain says, except for genetic with opponents `self`.
- Pause and Stop act between iterations. One iteration is all of its games plus the update, so with many games per iteration and one worker the button seems slow.
- Reset clears the Train tab but not the Network tab's evolution plots; they refresh when the next run adds an iteration.
- Load replay and the `replay` training feed do not refresh the Setup widgets, although the session adopts the recording's settings. Load config does refresh them, but not the reward sliders.
- The `replay` feed replaces the roster with the training game's tributes and the `live` feed writes the champion into some of yours. Save a scenario before turning the feed on if you want your roster back.
- With a curriculum, training games have fewer tributes than your roster (the learner copies plus that stage's opponents). The `replay` feed shows those smaller games; Watch agent and the `live` feed use your full roster.
- "start from the current champion" is ticked by default, so a second run of any method continues from the first run's champion unless you untick it. A genetic run of the voting brain ignores it, NEAT ignores a neural champion, and the neural methods ignore a NEAT champion. After Load champion into all, the file is only used when no run has happened yet or after Reset.
- Changing the network on the Brains tab between runs makes the old champion the wrong size for a warm start. Untick the box, press Reset, or load a champion file that matches before pressing Start.
- The hidden-layer width fields do nothing until Apply network settings is pressed. NEAT ignores the Brains tab's architecture.
- The Tributes tab's brain combo has no `neat` entry; a NEAT tribute keeps its brain only until you pick something in that combo.
- The imitation settings have no field for the teacher's chaos, the validation split or the number of student slots; they stay at 0, one fifth and 6. The curriculum window (5 iterations) has no field either.
- The reinforce and PPO settings have no fields for the value learning rate, the value network's width, gradient clipping, PPO's minibatch size or GAE lambda; they stay at 0.003, 32, 5.0, 256 and 0.95.
- The headline calls an iteration a "generation" only for the genetic method; NEAT iterations are labelled "epoch" there.
- Editing a `.py` file while the dashboard is open changes nothing on screen. Restart the dashboard to pick up source changes.
