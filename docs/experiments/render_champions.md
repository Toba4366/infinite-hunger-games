# `render_champions.py`

**Source:** [experiments/render_champions.py](../../experiments/render_champions.py)
**Depends on:** `argparse`, `json`, `sys`, `pathlib` (standard library); `numpy`; [../config.md](../config.md) (`NeuralConfig`, `SimulationConfig.from_dict`); [../renderer.md](../renderer.md) (`export_recording_gif`); [../training/common.md](../training/common.md) (`LearnerSpec`, `learner_ids`); [../training/reinforce.md](../training/reinforce.md) (`play_rl_episode`)
**Used by:** the results pages under [../results/README.md](../results/README.md)

## Purpose

A results table says a champion won 13 of 75 games. A GIF shows what that looks like: where its copies go, whether they drink, whether they fight or hide, how they die. This script takes a finished comparison run, rebuilds every champion from its saved `champion.json`, plays the first tournament game (seed 50000, the same seed every champion faced in the tournament) with the champion's copies in the learner slots, and writes one GIF per champion with a caption line in `gifs/index.md`.

## Concepts you need

**Champion files.** Every run folder under `runs/` has a `champion.json` in the shared shape (`brain_name`, `neural`, `genome`, `fitness`). A neural genome is a flat list of weights; a NEAT genome is a dictionary of nodes and connections.

**Learner slots.** `learner_ids(num_players, learners)` spreads the copies across the roster (slots 0, 4, 8, ... for six copies of 24), exactly as the tournament did.

**Greedy play.** The game is played with `greedy=True`, so the copies take their highest-scoring action every tick, as in the tournament and in validation.

## Walkthrough

### `load_spec(champion_file)`

Reads the JSON. A dictionary genome becomes `LearnerSpec("neat", genome)`; a list becomes `LearnerSpec("neural", array, NeuralConfig(...))` with the saved architecture (hidden layers restored as a tuple), or the default architecture when the file has none.

### `champions_of(run_folder)`

The run's `config.json` lists its variants in order. For each, the last `runs/<name>_*/champion.json` (the extension phase writes a second folder for the variants it extended, and the later one holds the final champion).

### `render(run_folder, seed=50000, step=2, fps=15, max_frames=600)`

1. Rebuilds the base config with `SimulationConfig.from_dict` and sets the comparison's seed on it, as `run_tournament` does; `learner_ids` with the run's `tournament_learners`.
2. For each champion: `play_rl_episode(config, None, spec, learners, seed, True, True)`, the last argument turning on the tick-by-tick recording; then `export_recording_gif` to `gifs/<name>.gif`, drawing every `step`-th tick at `fps` frames per second, at most `max_frames` frames.
3. A caption row per GIF: whether any copy won (`learner_won`), the copies' mean survival in ticks and their total kills.
4. Writes `gifs/index.md` and returns the folder.

### `main()`

Flags: the run folder, `--seed`, `--step`, `--fps`, `--max-frames`.

## How to use it / experiment

```bash
python experiments/render_champions.py results/full_methods_20260903_025758
python experiments/render_champions.py results/sizes_20260903_135744 --seed 50007 --step 3
```

- A GIF of a 24-day game at `step=2` is about 300 frames and 3 MB; `--step 4 --max-frames 300` halves it, which is what the results pages use.
- Change `--seed` to any `50000 + i` to see a different tournament game; every champion faces the same seed, so the games are comparable.
- The learner copies carry the gold star the dashboard uses, so they are easy to follow.

## Gotchas

- **Renders in the main process**, one game at a time; eight champions take a few minutes.
- **Needs the run's `config.json`** and each variant's champion file; a run interrupted before `write()` has no config.
- **The imitation champion plays with the same greedy rule** as the others; it looks more decisive than the sampled policy that trained.
