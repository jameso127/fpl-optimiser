"""Drawing pieces shared by the pictures (pitch.py, transfers.py): colours, fonts, shirts, chips.

The font is Barlow Condensed (SIL Open Font License, see fonts/OFL.txt), bundled because the
slim container image has no fonts and players' names need accents.
"""

import io
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

FONTS = Path(__file__).parent / "fonts"

INK = "#1b1b2f"
WHITE = "#ffffff"
PURPLE = "#2d0a3a"
DEEP = "#1d0726"  # page background behind cards
MUTED = "#d9c8e0"  # secondary text on purple
ACCENT = "#00e58b"
RED = "#ff2d55"
YELLOW = "#ffd400"

# Shirt colours (body, sleeves) by club short name; anyone else gets a neutral grey.
KITS = {
    "ARS": ("#db0007", "#ffffff"), "AVL": ("#670e36", "#95bfe5"), "BOU": ("#da291c", "#000000"),
    "BRE": ("#e30613", "#ffffff"), "BHA": ("#0057b8", "#ffffff"), "BUR": ("#6c1d45", "#99d6ea"),
    "CHE": ("#034694", "#034694"), "CRY": ("#1b458f", "#c4122e"), "EVE": ("#003399", "#003399"),
    "FUL": ("#ffffff", "#000000"), "IPS": ("#0044a9", "#ffffff"), "LEE": ("#ffffff", "#1d428a"),
    "LEI": ("#003090", "#fdbe11"), "LIV": ("#c8102e", "#c8102e"), "MCI": ("#6cabdd", "#6cabdd"),
    "MUN": ("#da291c", "#000000"), "NEW": ("#241f20", "#ffffff"), "NFO": ("#dd0000", "#ffffff"),
    "SOU": ("#d71920", "#ffffff"), "SUN": ("#eb172b", "#ffffff"), "TOT": ("#ffffff", "#132257"),
    "WHU": ("#7a263a", "#1bb1e7"), "WOL": ("#fdb913", "#231f20"),
}  # fmt: skip
DEFAULT_KIT = ("#9aa0a6", "#6b7075")

# FPL's fixture difficulty colours, 1 (easiest) to 5, with a readable text colour on each.
DIFFICULTY = {
    1: ("#257d5a", WHITE), 2: ("#00ff86", INK), 3: ("#e7e7e7", INK),
    4: ("#ff1751", WHITE), 5: ("#80072d", WHITE),
}  # fmt: skip


@cache
def font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    name = "BarlowCondensed-Bold.ttf" if bold else "BarlowCondensed-Medium.ttf"
    return ImageFont.truetype(FONTS / name, size)


def text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    value: str,
    size: int,
    fill: str = WHITE,
    anchor: str = "mm",
    bold: bool = True,
) -> None:
    draw.text(xy, value, font=font(size, bold), fill=fill, anchor=anchor)


def fit(draw: ImageDraw.ImageDraw, value: str, size: int, width: float) -> str:
    """Shorten with an ellipsis until it fits."""
    while len(value) > 1 and draw.textlength(value, font=font(size)) > width:
        value = value[:-2] + "…"
    return value


def shirt(draw: ImageDraw.ImageDraw, cx: float, top: float, team: str, scale: float) -> None:
    """A football shirt in the club's colours, 108 x 96 at scale 1, centred on cx."""
    body, sleeves = KITS.get(team, DEFAULT_KIT)

    def pts(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
        return [(cx + x * scale, top + y * scale) for x, y in points]

    outline = "#00000055"
    outer = [(-16, 0), (-36, 6), (-54, 26), (-40, 46), (-30, 38), (-30, 96), (30, 96), (30, 38),
             (40, 46), (54, 26), (36, 6), (16, 0), (0, 14)]  # fmt: skip
    draw.polygon(pts(outer), fill=body, outline=outline, width=2)
    for side in (-1, 1):
        sleeve = [(36 * side, 6), (54 * side, 26), (40 * side, 46), (30 * side, 38)]
        draw.polygon(pts(sleeve), fill=sleeves, outline=outline, width=2)
    draw.line(pts([(-16, 0), (0, 14), (16, 0)]), fill=outline, width=3)


def badge(draw: ImageDraw.ImageDraw, xy: tuple[float, float], label: str, fill: str) -> None:
    """A round C / V badge."""
    x, y = xy
    draw.ellipse((x - 19, y - 19, x + 19, y + 19), fill=fill, outline=WHITE, width=3)
    text(draw, (x, y), label, 26, WHITE if fill == INK else INK)


def pill(
    draw: ImageDraw.ImageDraw,
    cx: float,
    y: float,
    label: str,
    fill: str,
    ink: str,
    size: int = 24,
) -> None:
    """A rounded label centred on cx, its top at y."""
    width = draw.textlength(label, font=font(size)) + size
    height = size + 6
    draw.rounded_rectangle((cx - width / 2, y, cx + width / 2, y + height), radius=height / 2,
                           fill=fill)  # fmt: skip
    text(draw, (cx, y + height / 2), label, size, ink)


def fixture_chip(draw: ImageDraw.ImageDraw, cx: float, y: float, player: dict[str, Any]) -> None:
    """The next opponent, coloured by FPL's difficulty rating; nothing if fixtures are unknown."""
    games = player.get("fixtures")
    if games is None:
        return
    if not games:
        pill(draw, cx, y, "No game", "#555555", WHITE)
        return
    first = games[0]
    label = f"{first['opponent']} ({'H' if first['home'] else 'A'})"
    label += f" +{len(games) - 1}" if len(games) > 1 else ""
    pill(draw, cx, y, label, *DIFFICULTY.get(first["difficulty"], DIFFICULTY[3]))


def doubt(player: dict[str, Any]) -> float | None:
    """The chance of playing, if FPL has flagged it below 100%."""
    chance = player.get("chance_of_playing")
    return chance if chance is not None and chance < 100 else None


def png(image: Image.Image) -> bytes:
    out = io.BytesIO()
    image.convert("RGB").save(out, format="PNG", optimize=True)
    return out.getvalue()
