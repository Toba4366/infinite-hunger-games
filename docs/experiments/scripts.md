# `run_full.sh` and `run_sensitivity.sh`

**Source:** [experiments/run_full.sh](../../experiments/run_full.sh), [experiments/run_sensitivity.sh](../../experiments/run_sensitivity.sh)
**Runs:** [run_comparison.md](run_comparison.md) (`experiments/run_comparison.py`)

## Purpose

The two shell scripts are the exact commands behind the pages in [../results/README.md](../results/README.md). They exist so that "the full experiment" is one thing that can be re-run, not a paragraph of flags to retype. Each writes its log to `results/<name>_log.txt` when started as shown below, and prints a final line (`ALL EXPERIMENTS DONE`, `SENSITIVITY DONE`) so a `tail -f` or a polling loop can tell when it has finished.

## `run_full.sh`

Three comparison runs in a row, each ending in a 75-game tournament:

1. **`full_methods`.** Every method (`imitation,genetic,neat,reinforce,ppo`), cold and warm pairs (`--pairs`), the curriculum, up to 150 iterations to the win criterion, then the extension phase (`--extend-iterations 1000 --extend-hours 2`) for anything still short of it. About 11 hours on an 8-core machine with 6 workers. Results: [../results/full_methods/README.md](../results/full_methods/README.md).
2. **`sizes`.** Imitation and warm PPO on three hidden-layer shapes (`--sizes 16,64x32,128x64`), 80 iterations, no extension. About 40 minutes. Results: [../results/sizes/README.md](../results/sizes/README.md).
3. **`initializers`.** Imitation and warm PPO with `--initializers xavier_uniform,he_uniform,zeros`, 80 iterations. About 30 minutes. Results: [../results/initializers/README.md](../results/initializers/README.md).

```bash
nohup bash experiments/run_full.sh > results/full_log.txt 2>&1 &
tail -f results/full_log.txt
```

## `run_sensitivity.sh`

The cold-start question raised by the full run: are REINFORCE and PPO slow to start because of the problem, or because of the learning rate, the entropy bonus and the batch size? One run, cold only, on the curriculum, to the criterion within 150 iterations (no extension, because the question is the speed of the first 150):

```bash
python experiments/run_comparison.py --methods reinforce,ppo --curriculum \
  --set learning_rate=1e-3,3e-3,1e-2 --set entropy_bonus=0.01,0.001 --set episodes_per_epoch=4,16 \
  --iterations 150 --until-win 0.5 --window 5 --games 75 --workers 6 --seed 0 --name sensitivity
```

Seven variants per method, each changing one field from the defaults (the defaults themselves appear as `learning_rate_0.001`, `entropy_bonus_0.01` and `episodes_per_epoch_4`). About two hours.

```bash
nohup bash experiments/run_sensitivity.sh > results/sensitivity_log.txt 2>&1 &
```

## `run_lessons.sh`

The second full experiment, on the lesson curriculum, trained to graduation. Every method, cold and warm pairs, `--curriculum lessons`, 150 iterations first and then up to 16,000 more iterations or 14 hours for anything short of the criterion at the last lesson (8 hours in the sizes block); champions are stage-aware. The cold variants use the settings the sensitivity sweep found for cold starts (`--cold-set episodes_per_epoch=16 --cold-set entropy_bonus=0.001 --cold-set reinforce.learning_rate=3e-3`) and the warm variants keep the defaults, so the cold-against-warm comparison is not confounded by settings tuned for one side. Then imitation and warm REINFORCE (the first experiment's winner) at three network sizes on the same curriculum.

```bash
nohup bash experiments/run_lessons.sh > results/lessons_log.txt 2>&1 &
```

Expect it to run for days rather than hours: a variant that never graduates uses its full 14 hours, and there can be six of them, then the sizes block.

## Gotchas

- **Both scripts `cd` to the repository root** (`"$(dirname "$0")/.."`), so they can be started from anywhere, and they use `python3 -u` so the log is unbuffered.
- **The `results/` folder is ignored by git.** Copy the report, `results.csv`, `summary.json`, `config.json` and the charts you cite into `docs/results/<name>/` and write the page, as the existing pages do.
- **The machine matters.** The timings above were measured with the machine otherwise idle; swapping or a busy browser stretched single iterations from seconds to minutes during the recorded runs.
