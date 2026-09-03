# Statistical analysis

## Tournament win rates with 95% Wilson intervals

| variant | wins / games | win rate | 95% interval |
| --- | --- | --- | --- |
| imitation | 12 / 75 | 0.160 | 0.094 to 0.259 |
| genetic_cold | 0 / 75 | 0.000 | 0.000 to 0.049 |
| genetic_warm | 2 / 75 | 0.027 | 0.007 to 0.092 |
| neat | 0 / 75 | 0.000 | 0.000 to 0.049 |
| reinforce_cold | 0 / 75 | 0.000 | 0.000 to 0.049 |
| reinforce_warm | 13 / 75 | 0.173 | 0.104 to 0.274 |
| ppo_cold | 5 / 75 | 0.067 | 0.029 to 0.147 |
| ppo_warm | 2 / 75 | 0.027 | 0.007 to 0.092 |

## Pairwise Fisher exact tests (two-sided)

| comparison | first | second | wins | difference | p | significant at 0.05 |
| --- | --- | --- | --- | --- | --- | --- |
| warm against cold | genetic_warm | genetic_cold | 2 against 0 of 75 | +0.027 | 0.4966 | no |
| warm against cold | reinforce_warm | reinforce_cold | 13 against 0 of 75 | +0.173 | 0.0001 | yes |
| warm against cold | ppo_warm | ppo_cold | 2 against 5 of 75 | -0.040 | 0.4419 | no |
| against imitation | genetic_cold | imitation | 0 against 12 of 75 | -0.160 | 0.0003 | yes |
| against imitation | genetic_warm | imitation | 2 against 12 of 75 | -0.133 | 0.0092 | yes |
| against imitation | neat | imitation | 0 against 12 of 75 | -0.160 | 0.0003 | yes |
| against imitation | reinforce_cold | imitation | 0 against 12 of 75 | -0.160 | 0.0003 | yes |
| against imitation | reinforce_warm | imitation | 13 against 12 of 75 | +0.013 | 1.0000 | no |
| against imitation | ppo_cold | imitation | 5 against 12 of 75 | -0.093 | 0.1203 | no |
| against imitation | ppo_warm | imitation | 2 against 12 of 75 | -0.133 | 0.0092 | yes |

## Learning trends (straight-line slopes per 100 iterations, whole run)

| variant | iterations | survival slope | p | score slope | p | entropy slope | p | validation win slope | p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| imitation | 70 | +146.0 | 1.82e-12 | +3.236 | 7.72e-08 | -1.262 | 4.47e-21 | +0.535 | 0.00921 |
| genetic_cold | 1150 | +2.4 | 0 | +0.046 | 1.82e-319 | +0.002 | 0.26 | +0.002 | 0.000608 |
| genetic_warm | 948 | -0.1 | 0.236 | +0.007 | 0.000246 | -0.009 | 3.98e-13 | -0.008 | 1.64e-13 |
| neat | 1023 | +0.5 | 2.72e-110 | +0.036 | 7.91e-229 | -0.047 | 4.06e-95 | -0.000 | 0.234 |
| reinforce_cold | 1150 | +4.3 | 2e-143 | +0.093 | 3.7e-93 | -0.009 | 9.88e-84 | -0.000 | 0.955 |
| reinforce_warm | 62 | +21.4 | 0.228 | +0.308 | 0.51 | +0.369 | 3.53e-26 | -0.787 | 0.00107 |
| ppo_cold | 1150 | +5.1 | 5.97e-150 | +0.074 | 4.81e-50 | -0.024 | 1.11e-188 | +0.002 | 0.214 |
| ppo_warm | 1150 | +2.6 | 5.3e-25 | +0.086 | 1.51e-52 | +0.060 | 2.04e-288 | -0.012 | 7.29e-16 |

Smoothed curves use a rolling mean over 5 iterations. Charts: `tournament_win_rate_ci.png`, `survival_smoothed.png`, `score_smoothed.png`, `entropy_smoothed.png`, `val_win_smoothed.png`.
