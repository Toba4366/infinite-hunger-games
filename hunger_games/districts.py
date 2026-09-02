"""districts.py - the twelve districts of Panem: names, industries and colours.

The renderer and the dashboard colour every tribute by district using this
table, and draw female tributes as circles and male tributes as squares so
the two tributes from one district can be told apart.
"""

# The industry each district is known for.
DISTRICT_INDUSTRIES = {
    1: "Luxury",
    2: "Masonry",
    3: "Technology",
    4: "Fishing",
    5: "Power",
    6: "Transportation",
    7: "Lumber",
    8: "Textiles",
    9: "Grain",
    10: "Livestock",
    11: "Agriculture",
    12: "Mining",
}

# Each district's colour as a hex string.
DISTRICT_COLORS = {
    # Peridot / "snot" green.
    1: "#A6C13C",
    # Purple.
    2: "#7B3FA0",
    # Electric blue.
    3: "#1E90FF",
    # Deep-sea blue.
    4: "#0B3D91",
    # Orange.
    5: "#FF7F0E",
    # Dove grey.
    6: "#9A9A9A",
    # Russet brown.
    7: "#80461B",
    # Peach.
    8: "#FFB380",
    # Buttery yellow.
    9: "#F2D648",
    # Crimson.
    10: "#C41E3A",
    # Dark green.
    11: "#1B5E20",
    # Black (drawn with a light outline so it shows on dark ground).
    12: "#111111",
}

# The two sexes, in the order the roster alternates them within a district.
SEXES = ("F", "M")
# The matplotlib marker for each sex: circle for female, square for male.
SEX_MARKERS = {"F": "o", "M": "s"}


def district_color_rgb(district: int) -> tuple[float, float, float]:
    """The district colour as three floats from 0.0 to 1.0 (matplotlib style)."""
    # Districts above 12 wrap around so any roster size still gets a colour.
    hex_color = DISTRICT_COLORS[(district - 1) % 12 + 1]
    # Strip the '#' and split into red, green and blue pairs.
    red, green, blue = (int(hex_color[index : index + 2], 16) for index in (1, 3, 5))
    # Scale 0..255 down to 0.0..1.0.
    return red / 255.0, green / 255.0, blue / 255.0


def district_color_255(district: int) -> tuple[int, int, int]:
    """The district colour as three integers from 0 to 255 (Dear PyGui style)."""
    # Convert the float version back up to 0..255.
    return tuple(int(round(channel * 255)) for channel in district_color_rgb(district))


def default_tribute_name(district: int, sex: str) -> str:
    """The name a tribute gets before the game maker renames them, e.g. 'D4 Female'."""
    # Spell the sex out in full.
    word = "Female" if sex == "F" else "Male"
    # Short district tag plus the word.
    return f"D{district} {word}"


def is_career_district(district: int, career_districts: tuple[int, ...]) -> bool:
    """Do tributes from this district train for the games and attract sponsors?"""
    # A simple membership check against the configured list.
    return district in career_districts
