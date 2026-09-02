"""hunger_games - a simulator for the Hunger Games, built from chapter 4 of
the "Infinite Hunger Games" video (https://youtu.be/dS3tgfNN1HM?t=1013).

Package map (read them in this order):

    config.py      every setting, in one place
    noise.py       Perlin noise, the height map generator
    terrain.py     heights -> water / sand / grass / rock
    districts.py   district names, industries, colours, sexes
    resources.py   supplies and the two layouts (Cornucopia, ring)
    arena.py       the world: terrain (generated or painted) + supplies + navigation maps
    actions.py     the vocabulary of things a body can do
    perception.py  what a player senses each tick
    brain/         the decision makers (voting, random, neural) and initializers
    player.py      the body
    sponsors.py    parachutes for favoured tributes
    gamemaker.py   the slow safe circle when the games go quiet (on by default, toggleable)
    scenario.py    painted map + loot + roster, saved as JSON
    records.py     the spreadsheet rows
    game.py        the referee for one game
    recorder.py    tick-by-tick recordings for replay and GIF export
    runner.py      play many games, write CSVs
    renderer.py    watch a game on screen, export GIFs
    analysis.py    the chapter 3 charts
    training/      the genetic algorithm and REINFORCE trainers, run folders
    research/      behaviour telemetry, one PNG per chart, parameter sweeps
    ui/            the game makers' dashboard (Dear PyGui)
"""

# Pull the most useful names up to the package level.
from hunger_games.config import ArenaShape, LayoutName, SimulationConfig
from hunger_games.game import Game
from hunger_games.runner import Runner

# What `from hunger_games import *` exposes.
__all__ = ["ArenaShape", "LayoutName", "SimulationConfig", "Game", "Runner"]
