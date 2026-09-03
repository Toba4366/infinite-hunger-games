#!/bin/bash
# The second full experiment: the lesson curriculum, trained to graduation.
#
# Every reward or evolution method, cold and warm, climbs the lesson curriculum (survive, survive the rules,
# beat 1, 3, 7, 11 and 23, generalise). Each variant gets 150 iterations first; anything still short of the win
# criterion at the last lesson then keeps training, with the same weights or population, for up to 3,000 more
# iterations or 4 hours, so the record shows how long every method needs to graduate the whole curriculum.
# Champions are chosen by stage first (highest lesson reached, then validation wins, then score).
# Then the same for the best method of the first experiment at three network sizes.
cd "$(dirname "$0")/.."
python3 -u experiments/run_comparison.py --methods imitation,genetic,neat,reinforce,ppo --pairs --curriculum lessons \
  --iterations 150 --until-win 0.5 --window 5 --extend-iterations 3000 --extend-hours 4 \
  --games 75 --workers 6 --seed 0 --name lessons_methods 2>&1
python3 -u experiments/run_comparison.py --methods imitation,reinforce --warm --curriculum lessons --sizes 16,64x32,128x64 \
  --iterations 150 --until-win 0.5 --window 5 --extend-iterations 3000 --extend-hours 2 \
  --games 75 --workers 6 --seed 0 --name lessons_sizes 2>&1
echo "LESSONS DONE"
