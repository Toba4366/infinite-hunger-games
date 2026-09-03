# Method comparison: initializers

Every variant trained for up to 80 iterations, then each champion played 75 seeded games as the learner against voting opponents.

## Ranking by tournament score

| rank | variant | method | tournament score | tournament win rate | survival | iterations | train seconds | lines of code |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ppo_xavier_uniform | ppo | 0.05 | 0.17 | 217 | 80 | 360 | 894 |
| 2 | imitation_xavier_uniform | imitation | -0.36 | 0.16 | 174 | 70 | 90 | 525 |
| 3 | ppo_he_uniform | ppo | -0.22 | 0.05 | 201 | 80 | 355 | 894 |
| 4 | imitation_he_uniform | imitation | -0.80 | 0.01 | 161 | 28 | 39 | 525 |
| 5 | imitation_zeros | imitation | -2.07 | 0.00 | 79 | 80 | 102 | 525 |
| 6 | ppo_zeros | ppo | -2.07 | 0.00 | 79 | 80 | 792 | 894 |

**Best in the tournament:** ppo_xavier_uniform (ppo) with a mean score of 0.05 and a win rate of 0.17.
**Simplest to implement:** imitation (525 lines).
**Fastest to train under this budget:** imitation_he_uniform (39 seconds).

## Training to the win criterion (50% of validation games over 5 iterations, at the final curriculum stage)

| variant | reached | iterations to criterion | seconds to criterion | iterations trained | extended | final validation win rate |
| --- | --- | --- | --- | --- | --- | --- |
| imitation_xavier_uniform | yes | 70.0 | 90 | 70 | 0 | 1.00 |
| imitation_he_uniform | yes | 28.0 | 39 | 28 | 0 | 1.00 |
| imitation_zeros | no | nan | nan | 80 | 0 | 0.00 |
| ppo_xavier_uniform | no | nan | nan | 80 | 0 | 0.00 |
| ppo_he_uniform | no | nan | nan | 80 | 0 | 0.00 |
| ppo_zeros | no | nan | nan | 80 | 0 | 0.00 |

## Why the methods differ

- **imitation**: Supervised learning: copies the voting brain. Needs a teacher; cannot exceed it.
- **ppo**: Clipped policy gradient with GAE and several passes per batch. Most stable of the reward methods.

## Charts

`plots/score_by_method.png`, `plots/score_by_time.png`, `plots/validation_by_method.png`, `plots/entropy_by_method.png`, `plots/length_by_method.png`, `plots/tournament_*.png`, `plots/lines_of_code.png`, `plots/train_seconds.png`. Each variant's own run folder is under `runs/`.
