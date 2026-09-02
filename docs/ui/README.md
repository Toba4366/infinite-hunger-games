# Using the game makers' dashboard

The dashboard is a desktop window for designing an arena, picking the tributes, watching a game play out tick by tick, training brains, and measuring what the tributes do. It is built with Dear PyGui. This page is for using it; the code is explained in [init.md](init.md), [main.md](main.md), [painter.md](painter.md), [session.md](session.md), [canvas.md](canvas.md), [visualizer.md](visualizer.md) and [app.md](app.md).

## Launching

From the project root (the folder that contains `hunger_games/`):

```text
python -m hunger_games ui
```

`python -m hunger_games.ui` does the same thing. The window opens at 1500 by 920 pixels (it can be resized down to 1100 by 700) with a generated arena and a roster of 24 tributes on their podiums. A readable system font is used when one is found; the theme is dark with crimson buttons and gold sliders.

## The three panels

| Panel | Where | What it holds |
| --- | --- | --- |
| Controls | Left | The four mouse tools and eight tabs: Setup, Map, Loot, Tributes, Brains, Play, Train, Research |
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

Female tributes are circles and male tributes are squares, filled with their district's colour and labelled `D4F`, `D4M` and so on (turn labels off on the Map tab). Weapons are red triangles, food a white dot, medicine a magenta plus. A yellow ring marks the selected tribute. While a game plays, a white parachute appears over anyone receiving a sponsor gift, a red X flashes where someone was just eliminated, and a red ring shows the game makers' safe circle when it is closing. While editing, only the stacks you placed by hand are drawn; the layout's own supplies appear once a game starts.

## Tab by tab

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
| | Save config, Load config | The settings as JSON. Loading refreshes every widget and regenerates the map |

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
| brain | voting, random or neural |
| granted weapon | A weapon they start with, by quality (0 = none) |
| granted food, granted medkits | Items in their pack at the start |
| sponsor favour bonus | Extra favour, so sponsors send gifts sooner |
| start thirst, start hunger, start health | Their own starting bars; 0 means use the Setup minimums |
| Forget trained genome | Drop a champion genome so the brain starts fresh |

### Brains

| Control | Meaning |
| --- | --- |
| default brain | voting (the video's instinct-voting brain), random (a baseline) or neural (the network below, untrained until you train it). Used by New roster and Add tribute |
| Give this brain to every tribute | Set everyone to that brain and drop any genomes |
| hidden layers | Widths separated by commas: `16` for one layer, `32,16` for two |
| activation | tanh, relu, leaky_relu, sigmoid or selu. tanh pairs with Xavier, relu with He, selu with LeCun |
| initializer | How starting weights are drawn; the note under it explains each one |
| init scale, sparsity | Used by the constant, uniform and normal initializers, and by sparse |
| Apply network settings | Read the fields into the settings and show the shape and parameter count, e.g. `50 -> 16 -> 16, tanh, xavier_uniform, 1088 params` |

The folding section "Inputs (50) and outputs (16)" lists the perception vector in order: the three bars, survival and training scores, weapon quality and reach, food and medkits carried, in water, hunt difficulty, downhill direction, direction and distance to water, grass and the centre, the loot here and nearby, the nearest threat's direction, distance, level and health, players in sight, the danger zone and hazard, the safe direction, day fraction, alive fraction, what the cannon and sky told them (field known, field strength, strongest remaining, my rank), and which terrain they stand on. The 16 outputs are rest, drink, eat, hunt, pick_up, heal, attack, flee and eight moves. The brain takes the highest score, or a softmax sample when chaos is above 0.

### Play

| Control | Meaning |
| --- | --- |
| Slow-mo 2/s, Normal 8/s, Fast 40/s, Max 400/s | Playback speed presets |
| start a new game when this one ends (back to back) | Every finished game's behaviour is kept for the Charts tab and the Research exports |
| Save replay, Load replay | The recording as a `.replay` file. Only open replays you made yourself |
| GIF ticks per frame, Export GIF of this game | 1 is every tick (a long file); 2 is the default. Finishes the game first, then writes the file |

### Train

Pick `genetic` or `reinforce` at the top; the settings below switch with it.

**Genetic** evolves a population of genomes by playing them against each other on the painted map, for the neural or the voting brain. Controls: brain to evolve, population, generations, games per genome, elite fraction, mutation rate and scale, crossover rate, validation games (the champion against the default brain on fixed seeds each generation), CPU workers.

**Reinforce** is policy gradient with a value baseline; it rewards every action and trains the neural brain only. Controls: epochs, games per epoch, learners per game (tributes driven by the learning policy; the rest use the default brain as opponents), learning rate, value learning rate, entropy bonus (keeps the policy varied), validation games, CPU workers, and the folding **Reward function**: per tick alive, win, death, kill, per health lost, per need restored (only while the bar was below half), placement (scaled by placing), and discount (how much a reward one tick later is worth).

Start training runs in the background; the window stays usable. Stop after this step finishes the current generation or epoch. The progress bar counts games in the current step.

| Plot | What it shows |
| --- | --- |
| Performance | Genetic: best fitness, validation fitness and population mean per generation. Fitness is 1.0 for winning, 0.0 for first out, plus small bonuses for kills and days. Reinforce: training return, validation return and win rate per epoch. A rising validation line that tracks the training line means real learning; training rising while validation stays flat means the policy is fitting its own games |
| Stability | Reinforce: policy loss, value loss and policy entropy. Entropy falling toward 0 means the policy is collapsing onto one action; raise the entropy bonus. Value loss should fall as the baseline learns. Genetic: the action entropy of that generation's games |
| Time per step | Seconds per generation or epoch, so you can budget a run |
| Champion genes | The latest step's genome as bars, gold where a gene changed since the step before. The voting brain's eight genes are named on the axis; big networks show the first 400 weights |

Buttons: Champion to all, Champion to selected, Watch champion (everyone gets the champion brain and a new game starts at normal speed). Save run folder writes `results/<name>_<timestamp>/` with `config.json`, `history.json`, `champion.json` and a `plots/` folder of PNGs plus a growing-curve GIF. Save champion and Load champion into all use a JSON champion file that carries the network architecture.

### Research

**Parameter sweep.** Pick a config field (nested ones use a dot, like `terrain.water_threshold`), type comma-separated values (booleans as `true`/`false`), set games per value, CPU workers and whether to collect behaviour telemetry, then Start sweep. Each value plays the same seeded games on the painted map. The results line shows, per finished value, the victor rate, mean days, and the share of player-versus-player and natural eliminations. The run folder `results/<parameter>_<timestamp>/` holds `config.json`, `results.csv`, `summary.json` and one plot per metric against the swept value.

**Charts of the games you have watched.** Every finished game watched this session (including back-to-back ones) is measured. Type a folder (default `output/watched`) and press Export behaviour charts to write twelve PNGs: action distribution, actions by thirst, hunger and health, instinct curves, consumption timing, fight or flight, proximity versus tributes remaining, actions by remaining, the position heatmap, armed versus unarmed heatmaps, and deaths by cause. Forget watched games clears the tally.

The last section answers the questions a reviewer asks: the method (genetic algorithm or REINFORCE), the reward function, the observation (a 50-value vector, not a grid), and the tooling (Dear PyGui dashboard, matplotlib charts).

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

The headline shows the day, tick, how many are alive and the frame number, and on the last frame of a finished game the victor or "no victor (draw)". Every game is recorded as it plays, so you can scrub back, click a tribute to see their bars at that moment, then play on. Playing past the end of what has been simulated simulates more.

## The right panel

**Inspector.** Click a tribute (or a row in the Tributes tab). Before a game it shows district and industry, sex, scores, brain, the granted items and the podium cell. During a game it shows the thirst, hunger and health bars at the current frame, whether they are alive or when and how they were eliminated, their weapon and reach, food, medkits, kills, sponsor favour and last action. The event log lists the most recent eliminations and parachutes up to the current frame.

**Network.** Select a neural tribute during a live game and its network is drawn as columns of nodes: the 50 named inputs on the left, the hidden layers, and the 16 named outputs with their probabilities on the right. Red nodes are positive activations, blue nodes negative, dark grey idle. Warm edges are positive weights, cool edges negative, and brighter means larger; only the six strongest edges into each node are drawn. The yellow output label is the action taken. Without a live neural tribute the tab shows the bare architecture from the Brains tab. The picture is redrawn every frame, so use Slow-mo or Step to follow one decision at a time.

**Charts.** Three live charts of every game watched this session, refreshed every 30 frames: the action distribution in percent; the instinct curves (how often the tribute drank, ate or fled at each level of the matching bar, which is how chapter 4's voting rules were tuned); and a heatmap of where tributes spend their time.

## Recipes

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

### Two hidden layers with He and relu, GA training, watch the champion, save the run

1. Brains tab: hidden layers `32,16`, activation `relu`, initializer `he_normal`, Apply network settings. The summary reads `50 -> 32 -> 16 -> 16, relu, he_normal, 2432 params`.
2. Train tab: `genetic`, brain to evolve `neural`, generations 20, CPU workers 4, Start training. The progress bar counts games; the Performance plot grows one point per generation and the gene bars turn gold where the champion changed.
3. When it finishes (or after Stop after this step), press Watch champion. Click a tribute and open the Network tab to watch its hidden layers.
4. Type a name in the field next to Save run folder and press it; the status line names the folder under `results/`. Save champion keeps just the genome.

### Train with reinforce and read the curves

1. Brains tab: default brain `voting` (the opponents), a small network such as `16`.
2. Train tab: `reinforce`, epochs 30, games per epoch 4, learners per game 6, CPU workers 4. Open Reward function if you want a different objective (for example a bigger kill reward and a smaller death penalty for aggressive tributes). Start training.
3. Performance: training return should rise; validation return (greedy policy on fixed seeds) is the honest number. Stability: policy loss is noisy by nature, value loss should fall, entropy should fall slowly and not hit zero. If it collapses, raise the entropy bonus. Watch champion loads the best-validated policy.

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
| `champion.json` | Train | The champion genome with its brain name and architecture; the reinforce version adds the value network |
| `results/<name>_<timestamp>/` | Train | `config.json`, `history.json`, `champion.json`, `plots/` |
| `results/<parameter>_<timestamp>/` | Research | `config.json`, `results.csv`, `summary.json`, `plots/`, plus one `batches/<value>/` folder per value |
| `output/watched/*.png` | Research | The twelve behaviour charts of the games watched this session |

Default names are offered by the file dialogs; the extension is added if you leave it off.

## Known limits

- The game makers' circle is drawn as a solid red ring, not the dashed circle the matplotlib renderer uses.
- Painting, placing loot and dragging podiums only work while no game is loaded. Load replay clears the live game (and shows the replay's map); otherwise restart the dashboard to edit again.
- Export GIF finishes the current game first, then writes the file, and the window is busy until it is done. A loaded replay is exported as it is.
- Replay files are Python pickles. Only open `.replay` files you made yourself, because loading a pickle can run code.
- Training and sweeps with more than one CPU worker use separate processes. This works from the dashboard on macOS, Windows and Linux; the dashboard's entry point is guarded so the workers do not open extra windows.
- Trainers and sweeps use the painted map but not the hand-placed loot or the edited roster.
- Load replay does not refresh the Setup widgets, although the session adopts the replay's settings. Load config does refresh them.
- Editing a `.py` file while the dashboard is open changes nothing on screen. Restart the dashboard to pick up source changes.
