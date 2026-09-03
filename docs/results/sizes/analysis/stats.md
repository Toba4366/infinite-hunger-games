# Statistical analysis

## Tournament win rates with 95% Wilson intervals

| variant | wins / games | win rate | 95% interval |
| --- | --- | --- | --- |
| imitation_16 | 0 / 75 | 0.000 | 0.000 to 0.049 |
| imitation_64x32 | 12 / 75 | 0.160 | 0.094 to 0.259 |
| imitation_128x64 | 9 / 75 | 0.120 | 0.064 to 0.213 |
| ppo_16 | 18 / 75 | 0.240 | 0.158 to 0.348 |
| ppo_64x32 | 13 / 75 | 0.173 | 0.104 to 0.274 |
| ppo_128x64 | 10 / 75 | 0.133 | 0.074 to 0.228 |

## Pairwise Fisher exact tests (two-sided)

| comparison | first | second | wins | difference | p | significant at 0.05 |
| --- | --- | --- | --- | --- | --- | --- |
| against imitation | imitation_64x32 | imitation_16 | 12 against 0 of 75 | +0.160 | 0.0003 | yes |
| against imitation | imitation_128x64 | imitation_16 | 9 against 0 of 75 | +0.120 | 0.0030 | yes |
| against imitation | ppo_16 | imitation_16 | 18 against 0 of 75 | +0.240 | 0.0000 | yes |
| against imitation | ppo_64x32 | imitation_16 | 13 against 0 of 75 | +0.173 | 0.0001 | yes |
| against imitation | ppo_128x64 | imitation_16 | 10 against 0 of 75 | +0.133 | 0.0014 | yes |

## Learning trends (straight-line slopes per 100 iterations, whole run)

| variant | iterations | survival slope | p | score slope | p | entropy slope | p | validation win slope | p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| imitation_16 | 80 | +66.9 | 1.41e-12 | +1.181 | 2.86e-06 | -1.030 | 4.36e-22 | +0.000 | nan |
| imitation_64x32 | 70 | +146.0 | 1.82e-12 | +3.236 | 7.72e-08 | -1.262 | 4.47e-21 | +0.535 | 0.00921 |
| imitation_128x64 | 60 | +147.1 | 2.77e-06 | +3.131 | 5.77e-05 | -1.544 | 1.88e-20 | +0.542 | 0.0425 |
| ppo_16 | 80 | +57.1 | 5.91e-08 | +0.710 | 0.024 | +0.269 | 1.87e-18 | -0.252 | 0.091 |
| ppo_64x32 | 80 | +1.3 | 0.885 | -1.337 | 3.68e-05 | +0.842 | 7.96e-47 | -0.667 | 1.74e-05 |
| ppo_128x64 | 80 | -6.1 | 0.569 | -1.549 | 1.3e-06 | +0.849 | 7.59e-36 | -0.800 | 2.97e-07 |

Smoothed curves use a rolling mean over 5 iterations. Charts: `tournament_win_rate_ci.png`, `survival_smoothed.png`, `score_smoothed.png`, `entropy_smoothed.png`, `val_win_smoothed.png`.
