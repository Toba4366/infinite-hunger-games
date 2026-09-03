#!/bin/bash
# The cold-start sensitivity sweep: which learning rate, entropy bonus and batch size a cold REINFORCE or PPO needs.
# One variant per value, each changing one field from the defaults, all cold, all on the curriculum, to the win
# criterion within 150 iterations (no extension: the point is the speed of the first 150), then the tournament.
cd "$(dirname "$0")/.."
python3 -u experiments/run_comparison.py --methods reinforce,ppo --curriculum \
  --set learning_rate=1e-3,3e-3,1e-2 --set entropy_bonus=0.01,0.001 --set episodes_per_epoch=4,16 \
  --iterations 150 --until-win 0.5 --window 5 --games 75 --workers 6 --seed 0 --name sensitivity 2>&1
echo "SENSITIVITY DONE"
