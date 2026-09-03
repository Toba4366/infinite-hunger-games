# Method comparison: sensitivity

Every variant trained for up to 150 iterations, then each champion played 75 seeded games as the learner against voting opponents.

## Ranking by tournament score

| rank | variant | method | tournament score | tournament win rate | survival | iterations | train seconds | lines of code |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ppo_episodes_per_epoch_16 | ppo | 0.68 | 0.05 | 209 | 150 | 2277 | 900 |
| 2 | ppo_entropy_bonus_0.001 | ppo | -1.27 | 0.00 | 134 | 150 | 428 | 900 |
| 3 | ppo_learning_rate_0.001 | ppo | -1.45 | 0.00 | 121 | 150 | 454 | 900 |
| 4 | ppo_entropy_bonus_0.01 | ppo | -1.45 | 0.00 | 121 | 150 | 492 | 900 |
| 5 | ppo_episodes_per_epoch_4 | ppo | -1.45 | 0.00 | 121 | 150 | 399 | 900 |
| 6 | ppo_learning_rate_0.003 | ppo | -1.59 | 0.00 | 116 | 150 | 501 | 900 |
| 7 | reinforce_learning_rate_0.001 | reinforce | -1.68 | 0.00 | 108 | 150 | 383 | 759 |
| 8 | reinforce_entropy_bonus_0.01 | reinforce | -1.68 | 0.00 | 108 | 150 | 432 | 759 |
| 9 | reinforce_episodes_per_epoch_4 | reinforce | -1.68 | 0.00 | 108 | 150 | 433 | 759 |
| 10 | reinforce_episodes_per_epoch_16 | reinforce | -1.70 | 0.00 | 105 | 150 | 571 | 759 |
| 11 | reinforce_learning_rate_0.003 | reinforce | -1.73 | 0.00 | 96 | 150 | 460 | 759 |
| 12 | reinforce_learning_rate_0.01 | reinforce | -1.76 | 0.00 | 95 | 150 | 413 | 759 |
| 13 | reinforce_entropy_bonus_0.001 | reinforce | -1.77 | 0.00 | 96 | 150 | 421 | 759 |
| 14 | ppo_learning_rate_0.01 | ppo | -1.82 | 0.00 | 102 | 150 | 1565 | 900 |

**Best in the tournament:** ppo_episodes_per_epoch_16 (ppo) with a mean score of 0.68 and a win rate of 0.05.
**Simplest to implement:** reinforce (759 lines).
**Fastest to train under this budget:** reinforce_learning_rate_0.001 (383 seconds).

## Training to the win criterion (50% of validation games over 5 iterations, at the final curriculum stage)

| variant | reached | iterations to criterion | seconds to criterion | iterations trained | extended | final validation win rate |
| --- | --- | --- | --- | --- | --- | --- |
| reinforce_learning_rate_0.001 | no | - | - | 150 | 0 | 0.00 |
| reinforce_learning_rate_0.003 | no | - | - | 150 | 0 | 0.00 |
| reinforce_learning_rate_0.01 | no | - | - | 150 | 0 | 0.00 |
| reinforce_entropy_bonus_0.01 | no | - | - | 150 | 0 | 0.00 |
| reinforce_entropy_bonus_0.001 | no | - | - | 150 | 0 | 0.00 |
| reinforce_episodes_per_epoch_4 | no | - | - | 150 | 0 | 0.00 |
| reinforce_episodes_per_epoch_16 | no | - | - | 150 | 0 | 0.00 |
| ppo_learning_rate_0.001 | no | - | - | 150 | 0 | 0.00 |
| ppo_learning_rate_0.003 | no | - | - | 150 | 0 | 0.00 |
| ppo_learning_rate_0.01 | no | - | - | 150 | 0 | 0.00 |
| ppo_entropy_bonus_0.01 | no | - | - | 150 | 0 | 0.00 |
| ppo_entropy_bonus_0.001 | no | - | - | 150 | 0 | 0.00 |
| ppo_episodes_per_epoch_4 | no | - | - | 150 | 0 | 0.00 |
| ppo_episodes_per_epoch_16 | no | - | - | 150 | 0 | 0.50 |

## Why the methods differ

- **reinforce**: Policy gradient with a value baseline. One pass per batch; high variance.
- **ppo**: Clipped policy gradient with GAE and several passes per batch. Most stable of the reward methods.

## Charts

`plots/score_by_method.png`, `plots/score_by_time.png`, `plots/validation_by_method.png`, `plots/entropy_by_method.png`, `plots/length_by_method.png`, `plots/tournament_*.png`, `plots/lines_of_code.png`, `plots/train_seconds.png`. Each variant's own run folder is under `runs/`.
