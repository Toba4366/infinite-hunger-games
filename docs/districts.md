# `districts.py`

**Source:** [hunger_games/districts.py](../hunger_games/districts.py)
**Depends on:** nothing. The file has no imports at all, which makes it safe for every other module to use.
**Used by:** [game.py](game.md) (`SEXES`, `default_tribute_name`), [renderer.py](renderer.md) (`SEX_MARKERS`, `district_color_rgb`), [sponsors.py](sponsors.md) (`is_career_district`), and the dashboard (`ui/app.py` uses `DISTRICT_INDUSTRIES` and `SEXES`, `ui/session.py` uses `SEXES` and `default_tribute_name`, `ui/canvas.py` uses `district_color_255`).

## Purpose

`districts.py` is a small reference table for the twelve districts of Panem: what each one is known for, what colour it is drawn in, and how the two tributes from one district are told apart. It also holds the one rule that matters for gameplay: which districts are "careers", the districts whose tributes train for years and attract sponsors.

In the films every district sends one female and one male tribute, and the audience tracks them by district. The simulator does the same. `Game._generated_spec` cycles through districts `1..12` two slots at a time, alternating `"F"` and `"M"` using `SEXES`, and names each tribute with `default_tribute_name`. The renderer and the dashboard colour every tribute by district and draw females as circles and males as squares.

The career districts are `(1, 2, 4)` by default, set in [config.md](config.md) as `career_districts`. `is_career_district` is the membership test that `SponsorPool.favor` uses to give careers a quarter of their possible sponsor favour (see [sponsors.md](sponsors.md)). It takes the tuple as an argument rather than reading a global, so a config can change the careers without touching this file.

This is plain data with four tiny helpers. It is a good first file to read if you are new to Python: dictionaries, tuples, string formatting and one line of hex-colour parsing.

## Concepts you need

**Dictionaries.** `DISTRICT_INDUSTRIES[4]` looks up the value stored under the key `4`. Keys here are integers, values are strings.

**Tuples.** `SEXES = ("F", "M")` is an ordered pair that cannot be changed. `SEXES[index % 2]` alternates between the two.

**Modulo for wrapping.** `(district - 1) % 12 + 1` maps `13` to `1`, `14` to `2` and so on, so any roster size still gets a colour.

**Hex colours.** `"#A6C13C"` is red `A6`, green `C1`, blue `3C` in base 16. `int("A6", 16)` is `166`. Dividing by `255` gives the `0.0..1.0` floats matplotlib wants.

**Generator expressions and tuple unpacking.** `red, green, blue = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))` builds three values lazily and unpacks them into three names in one line.

**f-strings.** `f"D{district} {word}"` inserts variables into a string.

**Membership test.** `district in career_districts` is `True` if the value appears in the tuple.

## Walkthrough

### `DISTRICT_INDUSTRIES`

```python
DISTRICT_INDUSTRIES = {1: "Luxury", 2: "Masonry", ..., 12: "Mining"}
```

| District | Industry |
| --- | --- |
| 1 | Luxury |
| 2 | Masonry |
| 3 | Technology |
| 4 | Fishing |
| 5 | Power |
| 6 | Transportation |
| 7 | Lumber |
| 8 | Textiles |
| 9 | Grain |
| 10 | Livestock |
| 11 | Agriculture |
| 12 | Mining |

Display text only. The dashboard's district combo box shows `"4 Fishing"` and the inspector panel prints `District 12 (Mining)`. Nothing in the simulation reads the industry.

### `DISTRICT_COLORS`

```python
DISTRICT_COLORS = {1: "#A6C13C", 2: "#7B3FA0", ..., 12: "#111111"}
```

| District | Hex | Colour |
| --- | --- | --- |
| 1 | `#A6C13C` | Peridot green |
| 2 | `#7B3FA0` | Purple |
| 3 | `#1E90FF` | Electric blue |
| 4 | `#0B3D91` | Deep-sea blue |
| 5 | `#FF7F0E` | Orange |
| 6 | `#9A9A9A` | Dove grey |
| 7 | `#80461B` | Russet brown |
| 8 | `#FFB380` | Peach |
| 9 | `#F2D648` | Buttery yellow |
| 10 | `#C41E3A` | Crimson |
| 11 | `#1B5E20` | Dark green |
| 12 | `#111111` | Black |

District 12 is near-black, so the renderer draws it with a light outline to keep it visible on dark ground.

### `SEXES`

```python
SEXES = ("F", "M")
```

The two sexes in the order the roster alternates them within a district. `Game._generated_spec` uses `SEXES[index % 2]`, so even slots are female and odd slots male.

### `SEX_MARKERS`

```python
SEX_MARKERS = {"F": "o", "M": "s"}
```

The matplotlib marker per sex: `"o"` is a circle, `"s"` is a square. The renderer groups players by sex and scatters each group with its marker.

### `district_color_rgb`

```python
def district_color_rgb(district: int) -> tuple[float, float, float]
```

The district colour as three floats from `0.0` to `1.0`. Wraps the district with `(district - 1) % 12 + 1`, reads the hex string, slices out the three two-character pairs at positions `1`, `3` and `5`, parses each in base 16, and divides by `255.0`.

```python
from hunger_games.districts import district_color_rgb
print(district_color_rgb(5))    # (1.0, 0.4980..., 0.0549...)
print(district_color_rgb(17))   # same as district 5
```

### `district_color_255`

```python
def district_color_255(district: int) -> tuple[int, int, int]
```

The same colour as three integers `0..255`, for Dear PyGui, which wants byte channels. It calls `district_color_rgb` and multiplies back up with `round`. Going float and back is a little roundabout but keeps one source of truth.

```python
from hunger_games.districts import district_color_255
print(district_color_255(5))    # (255, 127, 14)
```

### `default_tribute_name`

```python
def default_tribute_name(district: int, sex: str) -> str
```

The name a tribute gets before a game maker renames them. `"F"` becomes `"Female"`; anything else becomes `"Male"`.

```python
from hunger_games.districts import default_tribute_name
print(default_tribute_name(4, "F"))   # D4 Female
print(default_tribute_name(12, "M"))  # D12 Male
```

### `is_career_district`

```python
def is_career_district(district: int, career_districts: tuple[int, ...]) -> bool
```

`district in career_districts`. That is the whole function. It exists as a named function so the rule has one home and reads clearly at the call site in `SponsorPool.favor`.

```python
from hunger_games.districts import is_career_district
print(is_career_district(2, (1, 2, 4)))   # True
print(is_career_district(12, (1, 2, 4)))  # False
```

## How to use it / experiment

Print the full table:

```python
from hunger_games.districts import DISTRICT_COLORS, DISTRICT_INDUSTRIES
for district in range(1, 13):
    print(district, DISTRICT_INDUSTRIES[district], DISTRICT_COLORS[district])
```

Change which districts are careers without editing this file. It is a config setting:

```python
from hunger_games.config import SimulationConfig
from hunger_games.game import Game

config = SimulationConfig(career_districts=(11, 12), seed=1)
game = Game(config)
d12 = [p for p in game.players if p.district == 12][0]
print(game.sponsors.favor(d12))   # now includes the career quarter
```

Recolour a district for your own charts by editing `DISTRICT_COLORS` at runtime (this affects every later call):

```python
from hunger_games import districts
districts.DISTRICT_COLORS[12] = "#FF00FF"
print(districts.district_color_rgb(12))
```

Reuse the colours in your own matplotlib plot:

```python
import matplotlib.pyplot as plt
from hunger_games.districts import SEX_MARKERS, district_color_rgb
plt.scatter([1], [1], color=[district_color_rgb(4)], marker=SEX_MARKERS["F"])
```

## Gotchas

- **Districts wrap, but only for colours.** `district_color_rgb(13)` returns district 1's colour. `DISTRICT_INDUSTRIES[13]` raises `KeyError`. Keep district numbers `1..12` unless you only need a colour.
- **`default_tribute_name` treats anything but `"F"` as male.** It does not validate the string.
- **Careers are not stored here.** The default `(1, 2, 4)` lives in `SimulationConfig.career_districts`. `is_career_district` only checks what it is given.
- **Both colour helpers return tuples, not lists.** matplotlib wants a list of colours for `scatter`, so wrap a single colour in `[...]` as in the example above.
- **Sex strings are single letters.** The roster, the records and the renderer all expect `"F"` or `"M"`. Do not write `"Female"` into a `TributeSpec`.
