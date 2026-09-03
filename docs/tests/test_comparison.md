# `test_comparison.py`

**Source:** [tests/test_comparison.py](../../tests/test_comparison.py)
**Tests:** [../research/comparison.md](../research/comparison.md) (`ComparisonConfig`, `MethodComparison`, `Variant`, `count_lines`), with [../training/init.md](../training/init.md) (`ImitationConfig`, `RLConfig`) and [../config.md](../config.md) (`SimulationConfig`)

## Purpose

`MethodComparison` is the experiment that answers the project's research question, and it runs for hours at full size. This file runs it at the smallest size that still exercises every stage: two variants train for one iteration each, the second warm-starts from the first, both champions fight a two-game tournament, and the run folder gets its table, plots, LaTeX and report. One extra assertion pins down the lines-of-code measure.

Without this test a change to a trainer's `champion_spec`, to `IterationStats.to_row`, or to the plot names would only show up at the end of a long real run.

## Concepts you need

**Test discovery.** One `test_*` function. It takes the `tmp_path` fixture so the run folder lands in a temporary directory.

**Variant and warm start.** A `Variant` names a method and its settings; `warm_from` names an earlier variant whose champion seeds it. See [../research/comparison.md](../research/comparison.md).

**The tournament.** Every champion plays the same seeded games as the learner against voting opponents, and the mean return, win rate, survival and kills are stored per variant.

**Running it.** `python -m pytest tests/test_comparison.py -q`. It plays a handful of 40 by 40, 3-day games and takes well under a minute.

## Walkthrough

### `test_comparison_trains_fights_and_reports(tmp_path)`

**Setup.** `config = SimulationConfig(seed=3, width=40, height=40, max_days=3)`: a small arena, 3 days (72 ticks), 24 tributes, the default 64 by 32 network. Two variants:

| Variant | Method | Settings | Warm start |
| --- | --- | --- | --- |
| `"imitation"` | imitation | `ImitationConfig(demonstration_games=1, epochs=1, validation_games=1, learners_per_game=4)` | none |
| `"reinforce_warm"` | reinforce | `RLConfig(epochs=1, episodes_per_epoch=1, learners_per_game=4, validation_games=1)` | `warm_from="imitation"` |

`ComparisonConfig(name="t", iterations=1, tournament_games=2, tournament_learners=4, results_dir=tmp_path)`. One iteration per variant, two tournament games per champion with four learner copies each, one worker, seed 0.

**`folder = comparison.run()`.** Trains imitation (one teacher game, one epoch, one validation game), then REINFORCE starting from the imitation champion (one training episode, one gradient step, one validation game), then plays two seeded games per champion, then writes everything. No progress callback.

**`table = comparison.table()` then `assert list(table["variant"]) == ["imitation", "reinforce_warm"]`.** One row per variant, in the order given. A failure means a variant was skipped or the table is built from a different list.

**`assert all(comparison.tournament[name]["games"] == 2 for name in comparison.tournament)`.** Both champions played exactly `tournament_games` games. A failure means a champion was missing (for example a broken `champion_spec`) or the job list was the wrong length.

**`assert (folder / "results.csv").exists() and (folder / "report.md").exists() and (folder / "results_table.tex").exists()`.** The three table outputs were written. A failure in the `.tex` file usually means the pandas call changed.

**`assert (folder / "plots" / "tournament_win_rate.png").exists() and (folder / "plots" / "score_by_method.png").exists()`.** One tournament bar chart and one overlay learning curve exist, so both plotting paths ran with the keys they expect.

**`assert (folder / "runs").exists()`.** Each variant's own `save_run` folder was written under `runs/`. A failure means `run_dir` was not set before `train_all`.

**`assert "Ranking by tournament score" in (folder / "report.md").read_text()`.** The report includes the ranking section, which only appears when the tournament dictionary is non-empty.

**`assert count_lines("ppo") > count_lines("reinforce")`.** PPO counts `reinforce.py` plus `ppo.py`, so it must be strictly larger. A failure means the file list per method changed or a file went missing.

## How to use it / experiment

- Add a third variant, for example `Variant("ppo_warm", "ppo", PPOConfig(epochs=1, episodes_per_epoch=1, learners_per_game=4, validation_games=1, update_epochs=1), warm_from="imitation")`, to check that the PPO subclass fits the same contract.
- Open `tmp_path` in a debugger (`print(folder)`) to look at a complete, tiny run folder without waiting for a real run.
- To check the warm start actually happened, compare `comparison.champions["imitation"].genome` with the REINFORCE trainer's starting policy; `_build` can be called directly on the second variant after the first has trained.

## Gotchas

- **A warm start that silently fails still passes this test.** The assertions check outputs, not that the REINFORCE policy began as the imitation champion. See `test_warm_starts_begin_from_the_given_genome` in [test_imitation.md](test_imitation.md) for that.
- **`results_dir` is a `Path` here** although the dataclass types it as `str`; `make_run_dir` accepts both.
- **Four learners on 24 tributes** sit in slots 0, 6, 12 and 18, so the tournament means are over 2 games times 4 learners, 8 learner episodes.
- **The test writes real PNGs**, so a matplotlib backend problem shows up here as a failure rather than a skipped chart.
