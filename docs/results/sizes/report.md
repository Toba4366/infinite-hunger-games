# Method comparison: sizes

Every variant trained for up to 80 iterations, then each champion played 75 seeded games as the learner against voting opponents.

## Ranking by tournament score

| rank | variant | method | tournament score | tournament win rate | survival | iterations | train seconds | lines of code |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ppo_16 | ppo | 0.43 | 0.24 | 215 | 80 | 261 | 894 |
| 2 | ppo_64x32 | ppo | 0.05 | 0.17 | 217 | 80 | 339 | 894 |
| 3 | imitation_64x32 | imitation | -0.36 | 0.16 | 174 | 70 | 74 | 525 |
| 4 | ppo_128x64 | ppo | -0.74 | 0.13 | 157 | 80 | 357 | 894 |
| 5 | imitation_128x64 | imitation | -0.64 | 0.12 | 172 | 60 | 76 | 525 |
| 6 | imitation_16 | imitation | -1.35 | 0.00 | 123 | 80 | 1126 | 525 |

**Best in the tournament:** ppo_16 (ppo) with a mean score of 0.43 and a win rate of 0.24.
**Simplest to implement:** imitation (525 lines).
**Fastest to train under this budget:** imitation_64x32 (74 seconds).

## Training to the win criterion (50% of validation games over 5 iterations, at the final curriculum stage)

| variant | reached | iterations to criterion | seconds to criterion | iterations trained | extended | final validation win rate |
| --- | --- | --- | --- | --- | --- | --- |
| imitation_16 | no | nan | nan | 80 | 0 | 0.00 |
| imitation_64x32 | yes | 70.0 | 74 | 70 | 0 | 1.00 |
| imitation_128x64 | yes | 60.0 | 76 | 60 | 0 | 1.00 |
| ppo_16 | no | nan | nan | 80 | 0 | 0.50 |
| ppo_64x32 | no | nan | nan | 80 | 0 | 0.00 |
| ppo_128x64 | no | nan | nan | 80 | 0 | 0.00 |

## Why the methods differ

- **imitation**: Supervised learning: copies the voting brain. Needs a teacher; cannot exceed it.
- **ppo**: Clipped policy gradient with GAE and several passes per batch. Most stable of the reward methods.

## Charts

`plots/score_by_method.png`, `plots/score_by_time.png`, `plots/validation_by_method.png`, `plots/entropy_by_method.png`, `plots/length_by_method.png`, `plots/tournament_*.png`, `plots/lines_of_code.png`, `plots/train_seconds.png`. Each variant's own run folder is under `runs/`.
