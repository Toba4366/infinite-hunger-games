# `test_comparison.py`

**Source:** [tests/test_comparison.py](../../tests/test_comparison.py)
**Tests:** [../research/comparison.md](../research/comparison.md) (`ComparisonConfig`, `MethodComparison`, `Variant`, `count_lines`), with [../training/init.md](../training/init.md) (`ImitationConfig`, `RLConfig`) and [../config.md](../config.md) (`SimulationConfig`)

## Purpose

`MethodComparison` is the experiment that answers the project's research question, and it runs for hours at full size. This file runs it at the smallest size that still exercises every stage: two variants train for one iteration each, the second warm-starts from the first, both champions fight a two-game tournament, and the run folder gets its table, plots, LaTeX and report. One extra assertion pins down the lines-of-code measure. A second test checks the extension phase: a variant that misses the win criterion in its first budget keeps training afterwards. A third, with stub objects instead of real trainers, pins down the rule that the criterion window must be played at the final curriculum stage.

Without this test a change to a trainer's `champion_spec`, to `IterationStats.to_row`, or to the plot names would only show up at the end of a long real run.

## Concepts you need

**Test discovery.** Three `test_*` functions. The first two take the `tmp_path` fixture so the run folder lands in a temporary directory; the third needs no files.

**The extension phase.** `ComparisonConfig.extended_iterations` lets variants that have not met the win criterion after `iterations` keep training, with the same trainer, once every variant has had its first budget. See [../research/comparison.md](../research/comparison.md).

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

### `test_comparison_extends_variants_that_miss_the_criterion(tmp_path)`

**Setup.** The same small config and one cold REINFORCE variant (`RLConfig(epochs=1, episodes_per_epoch=1, learners_per_game=4, validation_games=1)`). `ComparisonConfig(name="t", iterations=1, until_win_rate=0.0, win_window=2, extended_iterations=5, tournament_games=1, tournament_learners=4, results_dir=tmp_path)`. The trick is in the numbers: a window of two entries cannot be filled by a one-iteration first budget, so the criterion cannot be met in phase one, and a threshold of zero is met by any full window, so the variant stops on the first extension iteration.

**`folder = comparison.run(on_progress=lambda name, what: messages.append(what))`.** Collects every progress message. Phase one trains one iteration (no criterion), then the extension phase announces itself and trains one more, at which point the window is full and the criterion is met.

**`assert comparison.criterion["reinforce_cold"][0] == 2`.** The criterion was met at the second iteration overall, so the extension counted on from the first budget rather than restarting at one.

**`assert comparison.extended["reinforce_cold"] == 1`** and **`assert comparison.table()["extended_iterations"].tolist() == [1]`.** Exactly one extension iteration ran, and the table reports it. A failure of the first means the extension ran to its cap (the criterion check broke) or never ran; the second means the column was dropped.

**`assert any(what.startswith("extending") for what in messages)`.** The progress callback saw the "extending: the win criterion was not met within the first budget" line.

**`assert (folder / "runs_first_budget").exists() and (folder / "runs").exists()`.** The first-budget snapshot went to its own folder and the final state to `runs/`.

**`assert "extended" in (folder / "report.md").read_text()`.** The criterion table in the report has its extension column and note.

### `_StubCurriculum`, `_StubTrainer` and `test_win_criterion_needs_a_full_window_at_the_final_stage()`

**Why stubs.** The rule under test is about bookkeeping order, not about learning, so real trainers would only add minutes and randomness. `_StubCurriculum(final_stage=4)` starts at stage 3 and reports `finished` once its stage reaches 4. `_StubTrainer.step()` appends an `IterationStats` that wins every validation game (`val_win_rate=1.0`) and records the stage it was "played" at, then, on the fifth step only, promotes the curriculum to stage 4. That mirrors the real trainers, which record the iteration before calling `Curriculum.observe`.

**`reached = comparison._train_steps(Variant("x", "reinforce"), trainer, 20, time.time(), None, None)`.** A `MethodComparison` with no variants and `ComparisonConfig(until_win_rate=0.5, win_window=5)` runs up to 20 stub steps with no deadline and no progress callback.

**`assert reached[0] == 10`.** Without the fix, the five stage-3 wins that earned the promotion would satisfy the window the moment `finished` became true, and `reached` would be 5. With it, the window must hold five iterations recorded at stage 4, which takes five more steps.

**`assert all(s.stage == 4 for s in trainer.learning_history[-5:])`.** The last five records really were at the final stage, so the criterion was judged against the full field.

## How to use it / experiment

- Add a third variant, for example `Variant("ppo_warm", "ppo", PPOConfig(epochs=1, episodes_per_epoch=1, learners_per_game=4, validation_games=1, update_epochs=1), warm_from="imitation")`, to check that the PPO subclass fits the same contract.
- Open `tmp_path` in a debugger (`print(folder)`) to look at a complete, tiny run folder without waiting for a real run.
- To check the warm start actually happened, compare `comparison.champions["imitation"].genome` with the REINFORCE trainer's starting policy; `_build` can be called directly on the second variant after the first has trained.

## Gotchas

- **A warm start that silently fails still passes this test.** The assertions check outputs, not that the REINFORCE policy began as the imitation champion. See `test_warm_starts_begin_from_the_given_genome` in [test_imitation.md](test_imitation.md) for that.
- **`results_dir` is a `Path` here** although the dataclass types it as `str`; `make_run_dir` accepts both.
- **Four learners on 24 tributes** sit in slots 0, 6, 12 and 18, so the tournament means are over 2 games times 4 learners, 8 learner episodes.
- **The stub test calls the private `_train_steps`.** It is the only way to test the criterion rule without training; if that method's signature changes, update the stub call too.
- **The extension test relies on `until_win_rate=0.0`.** A threshold of zero is always met once the window is full, which is what makes the test deterministic; a positive threshold would depend on whether the tiny policy happened to win.
- **The test writes real PNGs**, so a matplotlib backend problem shows up here as a failure rather than a skipped chart.
