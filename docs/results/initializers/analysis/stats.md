# Statistical analysis

## Tournament win rates with 95% Wilson intervals

| variant | wins / games | win rate | 95% interval |
| --- | --- | --- | --- |
| imitation_xavier_uniform | 12 / 75 | 0.160 | 0.094 to 0.259 |
| imitation_he_uniform | 1 / 75 | 0.013 | 0.002 to 0.072 |
| imitation_zeros | 0 / 75 | 0.000 | 0.000 to 0.049 |
| ppo_xavier_uniform | 13 / 75 | 0.173 | 0.104 to 0.274 |
| ppo_he_uniform | 4 / 75 | 0.053 | 0.021 to 0.129 |
| ppo_zeros | 0 / 75 | 0.000 | 0.000 to 0.049 |

## Pairwise Fisher exact tests (two-sided)

| comparison | first | second | wins | difference | p | significant at 0.05 |
| --- | --- | --- | --- | --- | --- | --- |
| against imitation | imitation_he_uniform | imitation_xavier_uniform | 1 against 12 of 75 | -0.147 | 0.0023 | yes |
| against imitation | imitation_zeros | imitation_xavier_uniform | 0 against 12 of 75 | -0.160 | 0.0003 | yes |
| against imitation | ppo_xavier_uniform | imitation_xavier_uniform | 13 against 12 of 75 | +0.013 | 1.0000 | no |
| against imitation | ppo_he_uniform | imitation_xavier_uniform | 4 against 12 of 75 | -0.107 | 0.0614 | no |
| against imitation | ppo_zeros | imitation_xavier_uniform | 0 against 12 of 75 | -0.160 | 0.0003 | yes |

## Learning trends (straight-line slopes per 100 iterations, whole run)

| variant | iterations | survival slope | p | score slope | p | entropy slope | p | validation win slope | p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| imitation_xavier_uniform | 70 | +146.0 | 1.82e-12 | +3.236 | 7.72e-08 | -1.262 | 4.47e-21 | +0.535 | 0.00921 |
| imitation_he_uniform | 28 | +223.0 | 0.000963 | +5.526 | 0.0424 | -3.525 | 2.77e-11 | +1.861 | 0.0225 |
| imitation_zeros | 80 | +0.0 | nan | +0.000 | 1 | -0.180 | 2.99e-16 | +0.000 | nan |
| ppo_xavier_uniform | 80 | +1.3 | 0.885 | -1.337 | 3.68e-05 | +0.842 | 7.96e-47 | -0.667 | 1.74e-05 |
| ppo_he_uniform | 80 | +10.8 | 0.232 | -0.742 | 0.0108 | +0.927 | 1.95e-39 | -0.707 | 1.06e-08 |
| ppo_zeros | 80 | +6.9 | 0.0101 | +0.081 | 0.342 | +0.029 | 7.73e-58 | +0.000 | nan |

Smoothed curves use a rolling mean over 5 iterations. Charts: `tournament_win_rate_ci.png`, `survival_smoothed.png`, `score_smoothed.png`, `entropy_smoothed.png`, `val_win_smoothed.png`.
