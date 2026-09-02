# `run_rl.py`

**Source:** [experiments/run_rl.py](../../experiments/run_rl.py)
**Depends on:** `argparse`, `sys`, `pathlib` (standard library); [hunger_games/config.py](../config.md) (`NeuralConfig`, `SimulationConfig`); [training/__init__.py](../training/init.md) (`ReinforceTrainer`, `RLConfig`, `save_run`)
**Used by:** nobody imports it. It is run from the command line.

## Purpose

A command-line wrapper around [training/reinforce.md](../training/reinforce.md). It builds a `SimulationConfig` and an `RLConfig` from flags, trains the neural brain by REINFORCE with a value baseline while printing one line per epoch, and saves everything with `save_run`.

```
python experiments/run_rl.py --epochs 30 --episodes 4 --learners 6 --workers 4
```

## Concepts you need

**Epoch.** One round of collect-then-update: play `--episodes` games, take one gradient step, validate on fixed seeds. See [../training/reinforce.md](../training/reinforce.md).

**Learners and opponents.** Only `--learners` tributes per game are driven by the policy being trained. The rest use the brain named by `--opponents`, which becomes `SimulationConfig.brain_name`. Training against the voting brain is the default because it is a fixed, competent opponent.

**`sys.path` and the `__main__` guard.** Same as in [run_ga.md](run_ga.md): the script puts the repo root first on `sys.path` so `import hunger_games` works from anywhere, and `main()` is guarded so that `spawn`-started worker processes on macOS import the file without starting a second training run.

## Walkthrough

### Path setup

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

Then the two package imports with `# noqa: E402`, because they must come after the path change.

### `main()`

```python
def main() -> None:
```

**Flags.**

| Flag | Default | Goes to | Meaning |
| --- | --- | --- | --- |
| `--epochs` | `30` | `RLConfig.epochs` | Collect-update rounds |
| `--episodes` | `4` | `RLConfig.episodes_per_epoch` | Games per epoch |
| `--learners` | `6` | `RLConfig.learners_per_game` | Learner tributes per game |
| `--lr` | `1e-3` | `RLConfig.learning_rate` | Policy Adam step size |
| `--entropy` | `0.01` | `RLConfig.entropy_bonus` | Entropy bonus weight |
| `--workers` | `1` | `RLConfig.workers` | CPU cores for episode collection |
| `--seed` | `0` | `SimulationConfig.seed` and `RLConfig.seed` | Reproducibility |
| `--hidden` | `16` | `NeuralConfig.hidden_layers` | Comma-separated widths |
| `--activation` | `tanh` | `NeuralConfig.activation` | Activation of both networks |
| `--initializer` | `xavier_uniform` | `NeuralConfig.initializer` | Policy initializer |
| `--opponents` | `voting` | `SimulationConfig.brain_name` | Brain of the non-learner tributes |
| `--size` | `120` | `SimulationConfig.width` and `height` | Arena size |
| `--days` | `24` (`SimulationConfig.max_days`) | `SimulationConfig.max_days` | Day cutoff |
| `--name` | `rl` | `save_run(name=...)` | Run folder prefix |
| `--results` | `results` | `save_run(results_dir=...)` | Where the folder goes |

**Configs.** `NeuralConfig(hidden_layers, activation, initializer)`; `SimulationConfig(width, height, max_days, seed, neural, brain_name=opponents)`; `RLConfig(epochs, episodes_per_epoch, learners_per_game, learning_rate, entropy_bonus, workers, seed)`. Not exposed and left at their defaults: `value_learning_rate=3e-3`, `value_hidden=(32,)`, `validation_games=2`, `validation_seed=90000`, `max_grad_norm=5.0`, and every `RewardConfig` weight (`survive_tick=0.01`, `win=5.0`, `death=-3.0`, `kill=1.0`, `damage_taken=-2.0`, `need_gain=0.5`, `placement=2.0`, `discount=0.98`).

**Train.** `ReinforceTrainer(config, rl).run(on_epoch=...)`. The callback prints

```
epoch   4  ploss 0.012  vloss 3.481  H 2.61  train -1.84  val -0.95  surv 210  win 0.04  18.2s
```

per epoch: `epoch`, `policy_loss`, `value_loss`, `entropy`, `train_return`, `val_return`, `train_survival`, `win_rate`, `seconds`. The table in [../training/reinforce.md](../training/reinforce.md) says what trend to look for in each.

**Save.** `save_run(trainer, "reinforce", args.name, args.results)` and a `saved to ...` line.

### The guard

```python
if __name__ == "__main__":
    main()
```

"Needed for multiprocessing on macOS", as the source comment says.

## How to use it / experiment

**Smoke test.**

```
python experiments/run_rl.py --epochs 3 --episodes 1 --learners 4 --size 60 --days 6
```

**A longer run on four cores.**

```
python experiments/run_rl.py --epochs 300 --episodes 8 --learners 6 --workers 4 --name rl_300
```

**Entropy sweep.** Run with `--entropy 0.001`, `0.01` and `0.05` on the same seed and compare `entropy.png` and `reward.png`. Too little bonus and the entropy curve collapses early; too much and it never falls.

**Change the opponents.** `--opponents random` trains against dice rollers (easy), `--opponents neural` against untrained neural brains. There is no flag to load a trained champion as the opponent; for that, write a scenario roster (see [../scenario.md](../scenario.md)) and pass it to `ReinforceTrainer(scenario=...)` in a copy of this script.

**What the results folder contains.** `results/<name>_<timestamp>/` with `config.json`, `history.json` (one row per epoch with all thirteen numeric fields of `EpochStats`), `champion.json` (the policy with the best validation return, plus the value network), and `plots/` with `reward.png`, `losses.png`, `entropy.png`, `survival.png`, `win_kill_rate.png`, `reward.gif`, `timing.png` and the behaviour charts. Full list in [../training/runs.md](../training/runs.md).

**Reward shaping.** Not a flag. Copy the script and pass `reward=RewardConfig(...)` to `SimulationConfig`.

## Gotchas

- With `--workers` above 1, run from a file. The spawn start method re-imports the script in every worker.
- `--seed` seeds the trainer (its episode seeds and both networks' starting weights). The `SimulationConfig.seed` it also sets is overwritten per episode.
- `champion.json` holds the policy with the best `val_return`, not the final policy. With `validation_games` at its default of 2 that is usually what you want. If you edit the script to set `validation_games=0`, the champion freezes at epoch 0 (see [../training/reinforce.md](../training/reinforce.md) Gotchas).
- Thirty epochs is thirty gradient steps. Expect the reward curve to be flat and noisy at the default budget; use hundreds of epochs.
- `--learners` above `num_players` (24 by default) is clamped to the player count.
- The value network's learning rate, hidden size, gradient clipping and the validation seeds are not flags. Edit the `RLConfig(...)` call.
- `--activation` is shared by the policy and the value network; `--initializer` only affects the policy. The value network always uses `xavier_uniform`.
