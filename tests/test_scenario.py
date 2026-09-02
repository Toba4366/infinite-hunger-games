"""Tests for scenarios: painted maps, hand-placed loot, edited rosters, sponsors and wounds."""

import numpy as np

from hunger_games.config import SimulationConfig
from hunger_games.game import Game
from hunger_games.resources import ResourceKind
from hunger_games.scenario import LootSpec, Scenario, TributeSpec
from hunger_games.terrain import TerrainType


def flat_map(size: int = 40, kind: TerrainType = TerrainType.GRASS) -> list[list[int]]:
    return np.full((size, size), int(kind), dtype=np.int8).tolist()


def test_scenario_json_round_trip(tmp_path):
    """Save and load must preserve the map, loot and roster, including tuples."""
    scenario = Scenario(
        terrain=flat_map(10),
        use_layout_loot=False,
        loot=[LootSpec(3, 4, int(ResourceKind.WEAPON), 1, 0.9)],
        tributes=[TributeSpec(0, "Katniss", 12, "F", 11, 0.8, "voting", podium=(2, 2), weapon_quality=0.9)],
        title="test",
    )
    path = tmp_path / "s.json"
    scenario.save(path)
    loaded = Scenario.load(path)
    assert loaded.terrain == scenario.terrain
    assert loaded.loot == scenario.loot
    assert loaded.tributes[0].podium == (2, 2)
    assert loaded.tributes[0].name == "Katniss"
    assert loaded.use_layout_loot is False


def test_game_uses_painted_map_loot_and_roster():
    """A game built from a scenario adopts the map, the loot and every roster edit."""
    roster = [
        TributeSpec(
            0,
            "Katniss",
            12,
            "F",
            11,
            0.8,
            "voting",
            podium=(5, 5),
            weapon_quality=0.95,
            food=3,
            medicine=1,
            favor_bonus=0.2,
            start_thirst=0.4,
        ),
        TributeSpec(1, "Peeta", 12, "M", 8, 0.5, "random", podium=(30, 30)),
    ]
    scenario = Scenario(
        terrain=flat_map(),
        use_layout_loot=False,
        loot=[LootSpec(10, 10, int(ResourceKind.MEDICINE), 2, 0.5)],
        tributes=roster,
    )
    game = Game(SimulationConfig(seed=1, width=40, height=40), scenario=scenario)
    assert (game.arena.terrain == int(TerrainType.GRASS)).all()
    assert game.arena.resources.peek(10, 10) == (ResourceKind.MEDICINE, 2, 0.5)
    assert (game.arena.resources.kind != 0).sum() == 1
    katniss, peeta = game.players
    assert (katniss.x, katniss.y) == (5, 5) and (peeta.x, peeta.y) == (30, 30)
    assert katniss.name == "Katniss" and katniss.sex == "F" and katniss.weapon_quality == 0.95
    assert katniss.food == 3 and katniss.medicine == 1 and katniss.favor_bonus == 0.2
    assert katniss.thirst == 0.4 and peeta.thirst == 1.0
    assert peeta.brain.name == "random"


def test_start_bars_follow_config_minimums():
    """Everyone starts full by default; a lower minimum spreads the starting bars."""
    full = Game(SimulationConfig(seed=2, width=40, height=40))
    assert all(p.thirst == 1.0 and p.hunger == 1.0 and p.health == 1.0 for p in full.players)
    spread = Game(SimulationConfig(seed=2, width=40, height=40, start_thirst_min=0.5))
    assert all(0.5 <= p.thirst <= 1.0 for p in spread.players)
    assert any(p.thirst < 0.99 for p in spread.players)


def test_sponsor_favor_and_gifts():
    """Careers with high scores are favoured, and a wounded favourite gets medicine."""
    game = Game(SimulationConfig(seed=3, width=40, height=40, sponsor_gift_chance=1.0))
    pool = game.sponsors
    star = game.players[0]
    star.district, star.training_score, star.kills = 2, 12, 3
    nobody = game.players[1]
    nobody.district, nobody.training_score, nobody.kills = 12, 1, 0
    assert pool.favor(star) > pool.favor(nobody)
    star.health = 0.3
    gifts = pool.daily_gifts([star, nobody], np.random.default_rng(0), 0, 2, 48)
    assert [g.kind for g in gifts] == ["medicine"] and star.medicine == 1


def test_deep_wounds_bleed_and_do_not_heal_by_resting():
    """Below half health a wound bleeds each tick; resting only closes minor wounds."""
    game = Game(SimulationConfig(seed=4, width=40, height=40))
    player = game.players[0]
    player.health = 0.4
    player.rest()
    assert player.health == 0.4
    player.tick_needs(0.0, 0.0)
    assert player.health < 0.4
    player.health = 0.9
    player.rest()
    assert player.health > 0.9


def test_medicine_is_rare_in_layouts():
    """With sponsors as the main source of healing, medkits are a small share of layout loot."""
    game = Game(SimulationConfig(seed=5))
    kinds = game.arena.resources.kind
    total = (kinds != 0).sum()
    assert (kinds == int(ResourceKind.MEDICINE)).sum() / total < 0.06
