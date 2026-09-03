#!/bin/bash
# The full experiment: every method (cold and warm), curriculum on, train to the win criterion, then the tournament.
# Variants that miss the criterion in 150 iterations keep training afterwards (up to 1000 more iterations or 2 hours
# each), so the record shows how long a cold start really needs and whether its final network competes.
cd "$(dirname "$0")/.."
python3 -u experiments/run_comparison.py --methods imitation,genetic,neat,reinforce,ppo --pairs --curriculum \
  --iterations 150 --until-win 0.5 --window 5 --extend-iterations 1000 --extend-hours 2 \
  --games 75 --workers 6 --seed 0 --name full_methods 2>&1
# Network sizes and initialisers, warm-started PPO with the curriculum.
python3 -u experiments/run_comparison.py --methods imitation,ppo --warm --curriculum --sizes 16,64x32,128x64 \
  --iterations 80 --until-win 0.5 --window 5 --games 75 --workers 6 --seed 0 --name sizes 2>&1
python3 -u experiments/run_comparison.py --methods imitation,ppo --warm --curriculum --initializers xavier_uniform,he_uniform,zeros \
  --iterations 80 --until-win 0.5 --window 5 --games 75 --workers 6 --seed 0 --name initializers 2>&1
echo "ALL EXPERIMENTS DONE"
