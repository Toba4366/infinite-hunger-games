# Statistical analysis

## Tournament win rates with 95% Wilson intervals

| variant | wins / games | win rate | 95% interval |
| --- | --- | --- | --- |
| reinforce_learning_rate_0.001 | 0 / 75 | 0.000 | 0.000 to 0.049 |
| reinforce_learning_rate_0.003 | 0 / 75 | 0.000 | 0.000 to 0.049 |
| reinforce_learning_rate_0.01 | 0 / 75 | 0.000 | 0.000 to 0.049 |
| reinforce_entropy_bonus_0.01 | 0 / 75 | 0.000 | 0.000 to 0.049 |
| reinforce_entropy_bonus_0.001 | 0 / 75 | 0.000 | 0.000 to 0.049 |
| reinforce_episodes_per_epoch_4 | 0 / 75 | 0.000 | 0.000 to 0.049 |
| reinforce_episodes_per_epoch_16 | 0 / 75 | 0.000 | 0.000 to 0.049 |
| ppo_learning_rate_0.001 | 0 / 75 | 0.000 | 0.000 to 0.049 |
| ppo_learning_rate_0.003 | 0 / 75 | 0.000 | 0.000 to 0.049 |
| ppo_learning_rate_0.01 | 0 / 75 | 0.000 | 0.000 to 0.049 |
| ppo_entropy_bonus_0.01 | 0 / 75 | 0.000 | 0.000 to 0.049 |
| ppo_entropy_bonus_0.001 | 0 / 75 | 0.000 | 0.000 to 0.049 |
| ppo_episodes_per_epoch_4 | 0 / 75 | 0.000 | 0.000 to 0.049 |
| ppo_episodes_per_epoch_16 | 4 / 75 | 0.053 | 0.021 to 0.129 |

## Pairwise Fisher exact tests (two-sided)

| comparison | first | second | wins | difference | p | significant at 0.05 |
| --- | --- | --- | --- | --- | --- | --- |

## Learning trends (straight-line slopes per 100 iterations, whole run)

| variant | iterations | survival slope | p | score slope | p | entropy slope | p | validation win slope | p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reinforce_learning_rate_0.001 | 150 | +26.3 | 2.28e-31 | +0.605 | 1.63e-19 | -0.114 | 2.34e-61 | +0.000 | nan |
| reinforce_learning_rate_0.003 | 150 | +37.7 | 6.91e-42 | +0.906 | 5.8e-30 | -0.209 | 9.49e-58 | +0.006 | 0.59 |
| reinforce_learning_rate_0.01 | 150 | +19.7 | 1.54e-08 | +0.311 | 0.00012 | -0.427 | 6.28e-29 | +0.008 | 0.274 |
| reinforce_entropy_bonus_0.01 | 150 | +26.3 | 2.28e-31 | +0.605 | 1.63e-19 | -0.114 | 2.34e-61 | +0.000 | nan |
| reinforce_entropy_bonus_0.001 | 150 | +28.2 | 5.7e-34 | +0.616 | 4.92e-19 | -0.183 | 4.43e-67 | +0.000 | nan |
| reinforce_episodes_per_epoch_4 | 150 | +26.3 | 2.28e-31 | +0.605 | 1.63e-19 | -0.114 | 2.34e-61 | +0.000 | nan |
| reinforce_episodes_per_epoch_16 | 150 | +37.8 | 1.97e-72 | +0.881 | 4.64e-50 | -0.100 | 4.55e-33 | -0.002 | 0.845 |
| ppo_learning_rate_0.001 | 150 | +42.9 | 4.73e-25 | +0.829 | 7.99e-15 | -0.255 | 2.91e-28 | +0.053 | 0.00803 |
| ppo_learning_rate_0.003 | 150 | +35.9 | 3.82e-20 | +0.648 | 4.05e-09 | -0.328 | 9.4e-29 | +0.025 | 0.105 |
| ppo_learning_rate_0.01 | 150 | +16.4 | 1.13e-09 | +0.344 | 2.9e-05 | -0.328 | 1.15e-33 | +0.000 | nan |
| ppo_entropy_bonus_0.01 | 150 | +42.9 | 4.73e-25 | +0.829 | 7.99e-15 | -0.255 | 2.91e-28 | +0.053 | 0.00803 |
| ppo_entropy_bonus_0.001 | 150 | +50.2 | 8.31e-28 | +0.973 | 3.42e-22 | -0.631 | 1.87e-54 | +0.073 | 0.00407 |
| ppo_episodes_per_epoch_4 | 150 | +42.9 | 4.73e-25 | +0.829 | 7.99e-15 | -0.255 | 2.91e-28 | +0.053 | 0.00803 |
| ppo_episodes_per_epoch_16 | 150 | +49.7 | 1.27e-32 | +0.602 | 2.01e-10 | -0.341 | 8.7e-62 | +0.075 | 0.0737 |

Smoothed curves use a rolling mean over 5 iterations. Charts: `tournament_win_rate_ci.png`, `survival_smoothed.png`, `score_smoothed.png`, `entropy_smoothed.png`, `val_win_smoothed.png`.
