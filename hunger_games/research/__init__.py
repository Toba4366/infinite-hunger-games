"""research - measuring what tributes do, not just who wins.

    telemetry.py    counts actions against needs, health, danger and position while a game runs
    plots.py        one PNG per chart: the chapter 3 charts, heatmaps, behaviour and training curves
    experiments.py  parameter sweeps that write timestamped run folders

The training packages (training/genetic.py, training/reinforce.py) use the
telemetry to log behaviour every generation or epoch, and the dashboard's
Research tab draws the plots.
"""

# The telemetry collector.
from hunger_games.research.telemetry import BehaviorTelemetry

# What `from hunger_games.research import *` exposes.
__all__ = ["BehaviorTelemetry"]
