# `voting.py`

**Source:** [hunger_games/brain/voting.py](../../hunger_games/brain/voting.py)
**Depends on:** `numpy`; [hunger_games/actions.py](../actions.md) (`DIRECTIONS`, `Action`, `ActionType`); [brain/base.py](base.md) (`Brain`); [hunger_games/perception.py](../perception.md) (`Perception`); [hunger_games/resources.py](../resources.md) (`ResourceKind`); [hunger_games/terrain.py](../terrain.md) (`TerrainType`)
**Used by:** [brain/__init__.py](init.md) (registered as `"voting"`, the default brain; `create_brain` passes `chaos` and `endgame`); [training/genetic.py](../training/genetic.md) (evolves the eight genes through `create_brain`); [hunger_games/ui/app.py](../ui/app.md) (`GENE_NAMES` label the gene chart); `tests/test_brains.py` (`DEFAULT_GENES`, `VotingBrain`)

## Purpose

This is the brain from chapter 4 of the video. Instead of one big rule, the tribute has several *instincts*: thirst, hunger, survival, danger, greed, and optionally an endgame instinct. Each instinct looks at the `Perception` and casts votes for the action it wants. The action with the most votes wins. The lower a need bar gets, the louder that instinct shouts, so a tribute dying of thirst insists on drinking no matter what else is happening.

Every number that shapes the votes lives in an eight-value genome, so the genetic algorithm can evolve better voters without touching this file.

The endgame instinct is new. It is off by default and switched on by `SimulationConfig.endgame_instinct`, which `create_brain` passes as the `endgame` argument. When on, and once fewer than half the field is alive, bold tributes head for the centre to find the last survivors. The nightly sky (`my_rank`) tells them how bold to be.

## Concepts you need

**Voting as a decision rule.** Each instinct adds a positive number of votes to one or more candidate actions. Votes for the same action pile up. The winner is the action with the largest total. This combines several competing goals without writing every combination by hand.

**Urgency curve.** A need bar runs from 1.0 (full) to 0.0 (dead). `urgency()` turns it into votes with `((1 - score) ** urgency_power) * 10`, plus an emergency bonus of 20 below 0.2. With the default power of 2 the curve stays low while the bar is mostly full and shoots up near empty.

**Genome.** A flat numpy array of the eight genes, in `GENE_NAMES` order. `genome()` returns a copy, `set_genome()` loads one. See [base.md](base.md).

**Chaos blending.** With `chaos = 0` the top action always wins. With `chaos = 1` each action's chance is proportional to its votes. In between the two probability vectors are mixed: `(1 - chaos) * certain + chaos * proportional`.

**Chebyshev distance.** Distances in the perception are king-move distances (the larger of `|dx|` and `|dy|`). A threat at `distance <= reach` can be hit.

**Frozen dataclass as a dictionary key.** `Action` is `frozen=True`, so two actions with the same kind, step and target are equal and hash the same. That is what lets the ballot add up votes for "move up" from several instincts.

## Walkthrough

### `GENE_NAMES`

The eight gene names in genome order.

| Index | Gene | Default | Meaning |
| --- | --- | --- | --- |
| 0 | `thirst_weight` | 1.0 | Multiplies the thirst instinct's votes |
| 1 | `hunger_weight` | 1.0 | Multiplies the hunger instinct's votes |
| 2 | `survival_weight` | 1.0 | Multiplies long-term planning votes (hunt early, heal, top up water) |
| 3 | `danger_weight` | 1.5 | Multiplies fight-or-flight votes, and the endgame push |
| 4 | `greed_weight` | 0.6 | Multiplies loot votes (pick up, walk to supplies, drift to centre) |
| 5 | `aggression` | 0.5 | 0.0 never picks fights, 1.0 always attacks; also fuels the centre drift and the endgame push |
| 6 | `caution` | 0.5 | 0.0 never runs, 1.0 always runs |
| 7 | `urgency_power` | 2.0 | Exponent of the urgency curve; higher waits longer before panicking |

### `DEFAULT_GENES`

`np.array([1.0, 1.0, 1.0, 1.5, 0.6, 0.5, 0.5, 2.0])`. Hand-tuned so danger outshouts greed and the two need instincts are equal. `VotingBrain.__init__` copies this array, so editing a brain's genes never changes the module-level default.

### `class Ballot`

A ballot box: a dictionary from `Action` to a running vote total.

#### `__init__(self) -> None`

Starts with `self.votes = {}`.

#### `cast(self, action: Action, votes: float) -> None`

Adds `votes` to the action's total. Zero or negative votes are ignored, so an instinct's formula can dip below zero without subtracting.

#### `winner(self, rng: np.random.Generator, chaos: float) -> Action`

1. An empty ballot returns `Action(REST)`.
2. List the candidate actions and their totals as an array; `best` is the argmax.
3. `chaos <= 0`: return `actions[best]`.
4. Otherwise `proportional = counts / counts.sum()`, `certain` is a one-hot on `best`, and `probabilities = (1 - chaos) * certain + chaos * proportional`. Draw one index with `rng.choice`.

With totals `{A: 6, B: 3, C: 1}` and chaos 0.4: `proportional = [0.6, 0.3, 0.1]`, `certain = [1, 0, 0]`, blend `= [0.84, 0.12, 0.04]`.

### `class VotingBrain(Brain)`

#### Class constants

| Name | Value | Meaning |
| --- | --- | --- |
| `name` | `"voting"` | Registry key and CSV label |
| `PANIC_DISTANCE` | `4` | A stronger player closer than this triggers all-out flight |
| `CRITICAL_LEVEL` | `0.2` | A need bar below this is life-threatening |
| `CRITICAL_BONUS` | `20.0` | Extra votes a life-threatening need casts, enough to beat fear (about 15) and greed (about 10) |

#### `__init__(self, chaos: float = 0.0, genome: np.ndarray | None = None, endgame: bool = False) -> None`

Stores `chaos` via the base class, stores `self.endgame = endgame`, copies `DEFAULT_GENES` into `self.genes`, and loads `genome` if one was given. `create_brain("voting", chaos, rng, endgame=...)` is the usual way to build one.

#### `genome(self) -> np.ndarray`

A copy of `self.genes`.

#### `set_genome(self, genome: np.ndarray) -> None`

Converts to float, checks the shape is `(8,)` (else `ValueError("VotingBrain genome must have 8 values")`), and stores a copy.

#### `gene(self, gene_name: str) -> float`

`float(self.genes[GENE_NAMES.index(gene_name)])`. Every instinct reads its weights through this.

#### `urgency(self, score: float) -> float`

Clamp `score` to 0..1, then `votes = ((1 - score) ** urgency_power) * 10`. If `score < CRITICAL_LEVEL`, add `CRITICAL_BONUS`. With the default power of 2:

| Bar | Votes |
| --- | --- |
| 1.0 | 0.0 |
| 0.8 | 0.4 |
| 0.6 | 1.6 |
| 0.5 | 2.5 |
| 0.4 | 3.6 |
| 0.3 | 4.9 |
| 0.2 | 6.4 |
| 0.19 | 26.6 (bonus kicks in) |
| 0.1 | 28.1 |
| 0.0 | 30.0 |

The jump at 0.2 is the video's "insist on drinking": below it, the need beats any other instinct.

#### `random_step(rng: np.random.Generator) -> tuple[int, int]` (static)

One of the eight compass directions, chosen at random. For wandering.

#### `decide(self, perception: Perception, rng: np.random.Generator) -> Action`

Makes a fresh `Ballot`, calls the eight instinct methods in this order, then returns `ballot.winner(rng, self.chaos)`:

1. `_vote_hazard` (escaping the game makers overrides everything)
2. `_vote_thirst`
3. `_vote_hunger`
4. `_vote_survival`
5. `_vote_danger`
6. `_vote_greed`
7. `_vote_endgame`
8. `_vote_idle`

Order does not change the totals; every instinct only adds.

#### `_vote_hazard(self, p: Perception, ballot: Ballot) -> None`

Nothing if `p.safe_direction == (0, 0)`. Already in the danger zone: 50 votes to move to safety, which beats any other instinct's ten. Fog closing and the edge within sight: `20 * (1 - hazard_distance / (vision_radius + 1)) + 5` votes, so 5 at the edge of sight and about 25 when adjacent.

#### `_vote_thirst(self, p: Perception, ballot: Ballot, rng: np.random.Generator) -> None`

Nothing if `thirst > 0.85`. Otherwise `votes = thirst_weight * urgency(thirst)` go to, in priority order: `DRINK` if standing in water; a move toward visible water; a move downhill (the video's rule); or half the votes to a random step.

#### `_vote_hunger(self, p: Perception, ballot: Ballot, rng: np.random.Generator) -> None`

Nothing if `hunger > 0.8`. `votes = hunger_weight * urgency(hunger)`. Food in the pack: all votes to `EAT` and stop. Otherwise `expected = survival_score - hunt_difficulty`:

- `expected >= -0.05`: `HUNT` with `votes * max(0.2, 0.5 + expected)`.
- Not on grass but grass in sight: move toward it with `votes * 1.0` if `expected < 0`, else `votes * 0.6`.
- Not on grass and none in sight: random step with `votes * 0.4`.

#### `_vote_survival(self, p: Perception, ballot: Ballot) -> None`

`votes = survival_weight * (0.5 + survival_score) * 2`. Fewer than 2 rations on grass with hunger below 0.9: `HUNT`. Health below 0.6 with a medkit: `HEAL` with `votes + urgency(health)`. Else health below 0.8 and nobody in sight: `REST` with half. In water and thirst below 0.95: `DRINK` with `votes * 0.3`.

#### `_vote_danger(self, p: Perception, ballot: Ballot) -> None`

Nothing if nobody is in sight. Otherwise:

```
my_strength = 0.4 * survival_score + 0.4 * weapon_quality + 0.2 * health
advantage   = my_strength - threat.threat
closeness   = 1 - threat.distance / (vision_radius + 1)
votes       = danger_weight * closeness * 10
confidence  = advantage + (aggression - 0.5)
```

- `confidence > 0`: attack if `threat.distance <= reach`, else move toward them, both with `votes * (0.5 + aggression)`.
- Not confident and `distance <= PANIC_DISTANCE`: flee with `votes * (0.5 + caution)`.
- Not confident and far: flee with `votes * 0.3 * (0.5 + caution)`.

#### `_vote_greed(self, p: Perception, ballot: Ballot) -> None`

`votes = greed_weight * 3`. Supplies underfoot: `PICK_UP` with `votes * 3 + 5`. Supplies in sight: move toward them with `votes * (1 - distance / (vision_radius + 1))`. `weapon_quality < 0.6` and a centre direction: move to the centre with `votes * (0.6 - weapon_quality) * (0.5 + aggression)`.

#### `_vote_endgame(self, p: Perception, ballot: Ballot) -> None`

Gated three ways: returns at once if `self.endgame` is `False`, if `alive_fraction > 0.5`, or if `center_direction == (0, 0)` (already at the centre). Otherwise:

```
thinning = (0.5 - alive_fraction) / 0.5          # 0 at half alive, 1 with nobody else left
boldness = 0.3 + aggression * 0.7 + 0.3 * weapon_quality
if field_known: boldness *= 0.5 + my_rank        # 0.5x for the weakest, 1.5x for the strongest
votes    = danger_weight * thinning ** 2 * boldness * 12
```

cast for a move toward the centre. `my_rank` is the fraction of the other living tributes with a lower training score than mine, learnt from the nightly sky (see [../game.md](../game.md)). When the sky is off, `field_known` is `False` and the rank multiplier is skipped. `thinning ** 2` keeps the push tiny until the field is really small: at a quarter alive it is 0.25, with two of 24 left it is about 0.69.

Without this instinct the last two tributes can wander opposite corners for days, the stalled endgame chapter 2 warns about.

#### `_vote_idle(self, p: Perception, ballot: Ballot, rng: np.random.Generator) -> None`

`REST` gets 0.5 and a random step gets 0.3, so a tribute with no opinions still does something.

### A worked ballot

A tribute on sand with default genes. Thirst 0.4, hunger 0.6, health 0.9, survival score 0.6, weapon 0.3 with reach 1, no food or medkits. Water is 3 cells away in direction `(0, 1)`, grass in direction `(-1, 0)`, the centre in direction `(1, -1)`, a weapon 4 cells away in direction `(1, 0)`, and hunt difficulty 0.8. One other player is in sight at distance 3, direction `(1, -1)`, threat 0.35. A quarter of the field is alive, the sky is known and `my_rank` is 0.8. Vision radius 8.

| Instinct | Working | Action | Votes |
| --- | --- | --- | --- |
| hazard | `safe_direction` is `(0, 0)` | none | |
| thirst | `1.0 * ((1 - 0.4)^2 * 10) = 3.6`; not in water, water in sight | move `(0, 1)` | 3.6 |
| hunger | `1.0 * ((1 - 0.6)^2 * 10) = 1.6`; no food; `expected = 0.6 - 0.8 = -0.2`, no hunt; grass in sight, full force | move `(-1, 0)` | 1.6 |
| survival | `votes = 2.2` but not on grass, healthy, not in water | none | |
| danger | strength `0.24 + 0.12 + 0.18 = 0.54`, advantage `0.19`, closeness `1 - 3/9 = 0.667`, votes `1.5 * 0.667 * 10 = 10.0`; confident, out of reach | move `(1, -1)` | 10.0 |
| greed | `votes = 1.8`; weapon at 4: `1.8 * (1 - 4/9)` | move `(1, 0)` | 1.0 |
| greed | poorly armed: `1.8 * 0.3 * 1.0` | move `(1, -1)` | 0.54 |
| endgame (if on) | thinning `0.5`, boldness `0.3 + 0.35 + 0.09 = 0.74`, times `1.3` for rank `= 0.962`; `1.5 * 0.25 * 0.962 * 12` | move `(1, -1)` | 4.33 |
| idle | | rest | 0.5 |
| idle | random step (seed 0 gives `(0, 1)`) | move `(0, 1)` | 0.3 |

Totals with the endgame instinct off: move `(1, -1)` 10.54, move `(0, 1)` 3.9, move `(-1, 0)` 1.6, move `(1, 0)` 1.0, rest 0.5. With it on, move `(1, -1)` rises to 14.87. Either way the tribute closes in on the weaker player, who happens to be in the direction of the centre. With chaos 0 that is the decision. With chaos 0.5 (endgame off) the probabilities are 0.80 for the winner, 0.11 for the water move, 0.05 for the grass move, 0.03 and 0.01 for the rest.

Now drop thirst to 0.15. Urgency becomes `0.85^2 * 10 + 20 = 27.2`, so the water move collects 27.5 votes and beats the 10.54 for the chase. That is the critical bonus doing its job.

## How to use it / experiment

**Build one directly.**

```python
import numpy as np
from hunger_games.brain.voting import VotingBrain, GENE_NAMES

brain = VotingBrain(chaos=0.2, endgame=True)
print(dict(zip(GENE_NAMES, brain.genome())))
```

**Try a different personality.** Set the genome, then run a game with a factory so every tribute shares it.

```python
from hunger_games.config import SimulationConfig
from hunger_games.game import Game

berserker = np.array([1.0, 1.0, 0.5, 2.0, 0.6, 1.0, 0.0, 2.0])
game = Game(SimulationConfig(seed=3), brain_factory=lambda i, rng: VotingBrain(chaos=0.3, genome=berserker, endgame=True))
print(game.run().winner_name)
```

**See the votes for one decision.** Wrap `Ballot.cast` to print each instinct's vote before it lands.

```python
from hunger_games.brain.voting import Ballot
original = Ballot.cast
def loud(self, action, votes):
    print(f"{votes:6.2f} -> {action.kind.value} {action.dx, action.dy}")
    original(self, action, votes)
Ballot.cast = loud
brain.decide(perception, np.random.default_rng(0))
Ballot.cast = original
```

**Switch the endgame on from the config.** `SimulationConfig(endgame_instinct=True)`; `Game._make_brain` passes it to `create_brain`. Pair it with `cannon_and_sky=True` so `my_rank` is known and the rank multiplier applies.

**Evolve the genes.** `TrainingConfig(brain_name="voting")` with a `GeneticTrainer` (see [../training/genetic.md](../training/genetic.md)). The dashboard's gene chart labels its x axis with `GENE_NAMES`.

**Add an instinct.** Write a `_vote_something(self, p, ballot)` method that reads the perception and calls `ballot.cast`, add a call in `decide`, and, if it needs a weight, append a name to `GENE_NAMES` and a default to `DEFAULT_GENES`. Old eight-value genomes will then fail `set_genome`'s shape check, which is the point.

## Gotchas

- The endgame instinct is off unless `endgame=True` is passed. `create_brain` passes `SimulationConfig.endgame_instinct`; `VotingBrain()` on its own defaults to off. `GeneticTrainer.__init__` and `champion_brain` build the template without it (the evaluation games do pass it).
- `urgency()` jumps by 20 at `CRITICAL_LEVEL`. Right at 0.2 the bar is *not* critical (`<`, not `<=`).
- `_vote_thirst` returns early above 0.85 and `_vote_hunger` above 0.8, so the survival instinct is the only thing topping up a nearly full bar.
- `_vote_hunger` can cast for both `HUNT` and a move in the same tick (hunt here, walk to grass). They compete on the ballot.
- `Ballot.cast` ignores non-positive votes. A gene set to a negative value silences that instinct rather than reversing it.
- `chaos` here is a linear blend, not a temperature. Chaos 1.0 is "proportional to votes", which is still far from uniform when one action has 50 votes.
- Genes are read by name through `GENE_NAMES.index`, which is a list search on every call. Fine at this scale, but do not add hundreds of genes.
- `random_step` draws from the game's `rng`, so a decision that wanders consumes randomness and changes every later draw in a seeded game. Adding or removing an instinct that wanders changes replays.
- `PANIC_DISTANCE`, `CRITICAL_LEVEL` and `CRITICAL_BONUS` are class constants, not genes. The genetic algorithm cannot evolve them.
- `set_genome` and `genome` copy; `self.genes` itself is a plain array, so `brain.genes[5] = 1.0` works and is not copied.
