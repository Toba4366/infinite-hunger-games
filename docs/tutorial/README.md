# Tutorial: from first launch to a trained brain

This is the written version of the dashboard's Tutorial tab. Every picture
was taken from the real dashboard by `python -m hunger_games.ui.screenshots`,
so what you see here is what you get. Each step names the tab it lives in
and the "Show me" button that performs it for you.

## 0. Install and launch

```bash
pip install -r requirements.txt
python -m hunger_games ui
```

The window has three panels. Left: the control tabs, starting with this
tutorial. Centre: the arena and the transport bar. Right: the inspector, the
neural network visualiser and live behaviour charts. Hover any control for a
tooltip.

![The dashboard on first launch](images/01_overview.png)

## 1. Build an arena

The **Setup** tab chooses the shape (the 74th games' open field or the 75th
games' circle), the loot layout (the video's ring redesign or the original
Cornucopia pile), the size, the seed and the chaos dial. Changing any of
those regenerates the arena at once and moves tributes off any void. The
**Map** tab loads presets: Perlin hills, flat field, flat round, the 75th
games' island in a sea (`quarter_quell`) and a lake with an island.

![The lake_island preset loaded from the Map tab](images/02_arena.png)

## 2. Paint terrain

Pick **Paint terrain** at the top, choose a terrain and a brush radius on the
Map tab, and drag on the arena. A ring follows the mouse showing the brush.
Stamps put a circle or square of the chosen terrain at the centre, and
"Carve round" turns the map into a circle. Save the result as a scenario.

![A rock stroke painted with the brush ring visible](images/03_paint.png)

## 3. Edit the tributes

The **Tributes** tab lists the roster: click a name, or a dot on the arena
with the Select tool, to edit that tribute. Rename them, change district,
sex, training and survival scores, brain, grant a weapon, food, medkits or
sponsor favour, and set starting bars. Podium presets place everyone along
the edge, around the cornucopia, at random, or in two camps; the **Move
tribute** tool drags one tribute at a time. Female tributes are circles,
males squares, coloured by district.

![A tribute selected in the roster and the editor filled in](images/04_tributes.png)

## 4. Place loot

The **Loot** tab sets a kind, quantity and quality. With the **Place loot**
tool, left-click places a stack and right-click removes it. Weapons are red
triangles, food white dots, medicine magenta crosses. A weapon's quality
decides its name, reach and damage.

![The Loot tab with the Place loot tool selected](images/05_loot.png)

## 5. Play a game

**New game** builds a game from your settings, map and roster and records
every tick. **Play** and **Pause**, **Step**, **To end** and **Rewind** sit
under the arena with the speed slider (from slow motion to 400 ticks per
second) and the frame slider for scrubbing. Parachutes appear above tributes
receiving sponsor gifts; a red cross marks an elimination. The **Play** tab
has speed presets, back-to-back games, replay files and GIF export.

![A game in progress](images/06_play.png)

## 6. Inspect a tribute and watch its network think

Click a tribute for its bars, weapon, kills, sponsor favour and last action
in the **Inspector**. Give tributes the neural brain on the **Brains** tab,
where you set the number of hidden layers and the nodes in each, the
activation and the initializer, and where the 50 inputs and 16 outputs are
listed. During a live game the **Network** tab draws the selected tribute's
network as a node graph: red nodes are positive activations, blue negative,
warm edges positive weights, cool edges negative, and the chosen action is
highlighted in gold. The picture updates every tick.

![A neural tribute's network drawn live](images/07_network.png)

## 7. Train one network, and watch it learn

The **Train** tab trains one network. It plays the tributes marked with a
gold star; every other tribute uses the video's voting brain, so the
learner always faces the same opponent it is being measured against. Pick
a method from the combo, and its one-line explanation appears underneath:

- **imitation** copies the voting brain's decisions (supervised). Start
  here: a fresh network picks "drink" one time in sixteen even while
  standing in water and dies of thirst on day three, so it needs instincts
  before any reward can teach it. "Learn only from the top N placings"
  shows it winning games only.
- **genetic** evolves the weights of a population; **neat** evolves the
  shape of the network too, in species, the way the Monopoly video does.
- **reinforce** and **ppo** learn from the reward function; PPO is the
  zombie video's method and the most stable of the two.

Keep **start from the current champion** ticked so each method continues
from the last champion, and tick the **curriculum** to face 1, 3, 7, 11
and then 23 opponents, promoted as the score clears each stage, the way the
zombie count grew from one to sixteen.

The panels follow the zombie video's dashboard: **Latest scores** shows one
bar per episode of the newest iteration; the **event monitor** logs
rollouts, records, evolution and curriculum promotions; the three graphs
show average score (mean, validation and best), entropy (lower means more
confident) and average game length; **learning statistics** show the
iteration, seed, seconds per iteration, maximum score, learning time, stage
and the rollout bar; and the line below shows CPU and memory. **Pause**
holds the run between iterations, **Reset** forgets it, **Watch agent**
gives the champion to the starred tributes and plays a game live so the
Network tab shows real activations (NEAT genomes are drawn as graphs).

![Training running with the feed on and the plots filling](images/08_train.png)

**Save run folder** writes `results/<name>_<timestamp>/` with the config,
history, the shared learning curves, the event log, the champion and one PNG
per chart plus GIFs. **Save champion** writes a JSON file you can load into
any roster.

## 8. Research

The **Research** tab sweeps any setting over a list of values, playing the
same seeded games on the painted map for each, and writes a run folder with
one chart per metric. It also exports one PNG per behaviour chart for every
game you have watched in this session. The **Charts** tab on the right shows
the same behaviour live. The [research guide](../research/README.md) explains
which chart answers which question.

![The Research tab and the live Charts tab](images/09_research.png)

## 9. Save and share

Configs (Setup), scenarios (Map), replays (Play), champions and run folders
(Train) all save to files. The command line and the `experiments/` scripts
reproduce anything the dashboard does; see the [README](../../README.md).

## Regenerating these pictures

```bash
python -m hunger_games.ui.screenshots docs/tutorial/images
```

On macOS the tool captures only the dashboard window, by its window id, and
needs Screen Recording permission for your terminal or editor.
