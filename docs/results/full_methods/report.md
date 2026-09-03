# Method comparison: full_methods

Every variant trained for up to 150 iterations, then each champion played 75 seeded games as the learner against voting opponents.

## Ranking by tournament score

| rank | variant | method | tournament score | tournament win rate | survival | iterations | train seconds | lines of code |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | reinforce_warm | reinforce | 0.43 | 0.17 | 216 | 62 | 175 | 753 |
| 2 | imitation | imitation | -0.36 | 0.16 | 174 | 70 | 218 | 525 |
| 3 | ppo_cold | ppo | 1.06 | 0.07 | 233 | 1150 | 3941 | 894 |
| 4 | ppo_warm | ppo | 0.66 | 0.03 | 219 | 1150 | 4435 | 894 |
| 5 | genetic_warm | genetic | -0.48 | 0.03 | 166 | 948 | 8555 | 759 |
| 6 | reinforce_cold | reinforce | -1.30 | 0.00 | 130 | 1150 | 5239 | 753 |
| 7 | genetic_cold | genetic | -1.81 | 0.00 | 90 | 1150 | 7197 | 759 |
| 8 | neat | neat | -2.04 | 0.00 | 78 | 1023 | 7961 | 981 |

**Best in the tournament:** reinforce_warm (reinforce) with a mean score of 0.43 and a win rate of 0.17.
**Simplest to implement:** imitation (525 lines).
**Fastest to train under this budget:** reinforce_warm (175 seconds).

## Training to the win criterion (50% of validation games over 5 iterations, at the final curriculum stage)

Every variant first trained for up to 150 iterations. Those that had not met the criterion by then kept training afterwards, with the same population or weights, for up to 1000 more iterations or 2.0 hours each. The 'extended' column counts those extra iterations; the run folders under `runs_first_budget/` keep the first-budget snapshot.

| variant | reached | iterations to criterion | seconds to criterion | iterations trained | extended | final validation win rate |
| --- | --- | --- | --- | --- | --- | --- |
| imitation | yes | 70.0 | 218 | 70 | 0 | 1.00 |
| genetic_cold | no | nan | nan | 1150 | 1000 | 0.00 |
| genetic_warm | no | nan | nan | 948 | 798 | 0.00 |
| neat | no | nan | nan | 1023 | 873 | 0.00 |
| reinforce_cold | no | nan | nan | 1150 | 1000 | 0.00 |
| reinforce_warm | yes | 62.0 | 175 | 62 | 0 | 1.00 |
| ppo_cold | no | nan | nan | 1150 | 1000 | 0.00 |
| ppo_warm | no | nan | nan | 1150 | 1000 | 0.00 |

## Warm start against cold start

| method | cold: tournament win rate | warm: tournament win rate | cold: iterations to criterion | warm: iterations to criterion | cold: seconds to criterion | warm: seconds to criterion |
| --- | --- | --- | --- | --- | --- | --- |
| genetic | 0.00 | 0.03 | - | - | - | - |
| ppo | 0.07 | 0.03 | - | - | - | - |
| reinforce | 0.00 | 0.17 | - | 62 | - | 175 |

The variant with the higher tournament win rate in each pair: genetic_warm, ppo_cold, reinforce_warm.

## Why the methods differ

- **imitation**: Supervised learning: copies the voting brain. Needs a teacher; cannot exceed it.
- **genetic**: Evolves weights only. No gradients; simple; scales poorly with weight count.
- **neat**: Evolves weights and structure with species. More machinery; small networks; slow per generation.
- **reinforce**: Policy gradient with a value baseline. One pass per batch; high variance.
- **ppo**: Clipped policy gradient with GAE and several passes per batch. Most stable of the reward methods.

## Charts

`plots/score_by_method.png`, `plots/score_by_time.png`, `plots/validation_by_method.png`, `plots/entropy_by_method.png`, `plots/length_by_method.png`, `plots/tournament_*.png`, `plots/lines_of_code.png`, `plots/train_seconds.png`. Each variant's own run folder is under `runs/`.
