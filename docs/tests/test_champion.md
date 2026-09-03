# `test_champion.py`

**Source:** [tests/test_champion.py](../../tests/test_champion.py)
**Tests:** [../training/common.md](../training/common.md) (`champion_key`), [../training/genetic.md](../training/genetic.md) (`GeneticTrainer.champion`, `GenerationStats`), with [../config.md](../config.md) (`SimulationConfig`) and [../training/init.md](../training/init.md) (`TrainingConfig`)

## Purpose

The tournament fights with each variant's champion, so how the champion is chosen decides what the whole comparison measures. These two tests pin down the rule: the highest curriculum stage first, then the validation win rate, then the validation score. Without them a trainer could quietly go back to "highest training fitness" and send an easy-rung genome to the tournament, which is what happened in the first full run (see [../results/full_methods/README.md](../results/full_methods/README.md), limitations).

## Concepts you need

**Tuple ordering.** `champion_key` returns `(stage, val_win_rate, val_score)`. Python compares tuples element by element, so any difference in the first element settles the comparison before the others are looked at.

**The GA's history.** `GeneticTrainer.history` is a list of `GenerationStats`, one per generation, each carrying that generation's champion genome. The `champion` property picks one of them.

**Running it.** `python -m pytest tests/test_champion.py -q`. Under a second; no games are played.

## Walkthrough

### `test_champion_key_orders_by_stage_then_wins_then_score()`

Four keys: an easy-stage entry with a perfect win rate and a high score, and three stage-3 entries with no wins, some wins, and the same wins with a better score.

**`assert hard_no_wins > easy_high_score`.** Stage 3 with nothing beats stage 0 with everything. A failure means the stage is no longer the first element.

**`assert hard_some_wins > hard_no_wins`** and **`assert hard_same_wins_better_score > hard_some_wins`.** At equal stage, wins decide; at equal wins, score decides.

### `test_genetic_champion_prefers_the_highest_stage()`

**Setup.** A tiny `GeneticTrainer` (40 by 40 arena, 2 days, population of 4); nothing is trained. A helper builds fake `GenerationStats` whose champion genome is filled with the generation's index, so the property's choice can be read back from the genome's first value. The two new fields, `val_win_rate` and `stage`, are set after construction because they sit after `showcase` in the dataclass.

**Three generations.** Generation 0 has the highest training fitness and a perfect validation win rate but played at stage 0. Generations 1 and 2 played at stage 2; generation 2 won half its validation games with a worse score than generation 1.

**`assert float(trainer.champion[0]) == 2.0`.** Generation 2 wins: stage 2 beats stage 0 regardless of fitness, and 0.5 wins beat 0 wins. A failure that returns 0.0 means the property went back to training fitness; 1.0 means it stopped looking at wins.

## How to use it / experiment

- Add a fourth generation at stage 2 with `val_win_rate=0.5` and `val_fitness=5.0` and check that it wins on score.
- The same rule applies to REINFORCE, PPO and NEAT through their `best_key`; a stub trainer like the one in [test_comparison.md](test_comparison.md) is the cheapest way to test those.

## Gotchas

- **`GenerationStats` is built positionally in `GeneticTrainer.step`**, so new fields must be appended after `showcase`. Putting them earlier shifts every later argument by one, which showed up as a `Recording` being plotted as a number.
- **Imitation is not covered**: it has no curriculum and keeps its lowest-validation-loss epoch.
