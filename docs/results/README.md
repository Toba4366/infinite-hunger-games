# Results kept in the repo

`results/` (the run folders every script and the dashboard write) is ignored by git because a single comparison run is hundreds of megabytes of replays, histories and charts. The pages in this folder keep the parts of the important runs that the README's claims rest on: the generated report, the numbers (`results.csv`, `summary.json`, `config.json`) and the charts that are referred to in the text.

| Page | Run | What it settles |
| --- | --- | --- |
| [full_methods/paper.md](full_methods/paper.md) | all three runs of 2026-09-03 | The research paper: abstract, background, methods, results with statistics, discussion, limitations, references, and the champions' games as GIFs |
| [full_methods/README.md](full_methods/README.md) | `results/full_methods_20260903_025758` | Every method, cold and warm, trained to the win criterion with the curriculum and the extension phase, then the 75-game tournament: which method makes sense, which is simplest, how long each needs, and whether warm starts end better |
| [sizes/README.md](sizes/README.md) | `results/sizes_20260903_135744` | Three hidden-layer shapes (16, 64x32, 128x64) through imitation and warm PPO: how wide the network needs to be to copy the teacher, and which shape fine-tunes best |
| [initializers/README.md](initializers/README.md) | `results/initializers_20260903_143756` | Xavier uniform, He uniform and all-zero starting weights through imitation and warm PPO: why zeros never learn, and which initialiser fine-tunes best |
| [sensitivity/README.md](sensitivity/README.md) | `results/sensitivity_20260903_153031` | Cold REINFORCE and PPO over learning rate, entropy bonus and episodes per epoch: which settings a cold start needs, and why the defaults suit warm starts |

Each run page states the exact command, the machine, the tables, the claims the run supports, and its limitations; each has an `analysis/` subfolder written by `experiments/analyze_comparison.py` (Wilson intervals, Fisher tests, trend slopes, smoothed curves) and, for the main run, a `gifs/` subfolder written by `experiments/render_champions.py`. The paper draws on all of them. To reproduce a run, use the command on its page; the run folder it writes has the same file names as the page's copies. See [../research/README.md](../research/README.md) for how the experiments are designed and [../output.md](../output.md) for what every file in a run folder means.
