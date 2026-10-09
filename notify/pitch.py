"""Draw the recommended team on a pitch, as a PNG to send with the message.

Portrait (4:5), which fills a phone screen in Telegram. Layout, top to bottom: a header with
the gameweek, deadline and verdict; the starting XI in formation, goalkeeper at the top as in the
FPL app, each player a shirt in club colours with name, expected points and fixture; the bench.
The font is Barlow Condensed (SIL Open Font License, see fonts/OFL.txt), bundled because the
slim container image has no fonts and players' names need accents.
"""

import datetime as dt
import io
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from PIL import Image, ImageDraw, ImageFont

from notify.format import pick

FONTS = Path(__file__).parent / "fonts"
WIDTH, HEIGHT = 1080, 1350
HEADER, BENCH = 200, 250  # heights of the bands above and below the pitch
LONDON = ZoneInfo("Europe/London")

INK = "#1b1b2f"
WHITE = "#ffffff"
PURPLE = "#2d0a3a"
ACCENT = "#00e58b"
GRASS = ("#2f8f4e", "#2a8547")  # alternating stripes
LINES = (255, 255, 255, 90)

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
def _font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    name = "BarlowCondensed-Bold.ttf" if bold else "BarlowCondensed-Medium.ttf"
    return ImageFont.truetype(FONTS / name, size)


def _text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    size: int,
    fill: str = WHITE,
    anchor: str = "mm",
    bold: bool = True,
) -> None:
    draw.text(xy, text, font=_font(size, bold), fill=fill, anchor=anchor)


def _fit(draw: ImageDraw.ImageDraw, text: str, size: int, width: float) -> str:
    """Shorten with an ellipsis until it fits."""
    while len(text) > 1 and draw.textlength(text, font=_font(size)) > width:
        text = text[:-2] + "…"
    return text


def _grass(draw: ImageDraw.ImageDraw, top: int, bottom: int) -> None:
    stripes = 10
    height = (bottom - top) / stripes
    for i in range(stripes):
        y = top + i * height
        draw.rectangle((0, y, WIDTH, y + height), fill=GRASS[i % 2])


def _markings(image: Image.Image, top: int, bottom: int) -> None:
    """Halfway line, centre circle and the penalty area at the top, drawn translucent."""
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    m, w = 40, 4
    d.rectangle((m, top + 20, WIDTH - m, bottom + 40), outline=LINES, width=w)
    d.rectangle((WIDTH / 2 - 230, top + 20, WIDTH / 2 + 230, top + 200), outline=LINES, width=w)
    d.rectangle((WIDTH / 2 - 100, top + 20, WIDTH / 2 + 100, top + 85), outline=LINES, width=w)
    half = bottom - 40
    d.line((m, half, WIDTH - m, half), fill=LINES, width=w)
    d.arc((WIDTH / 2 - 120, half - 120, WIDTH / 2 + 120, half + 120), 180, 360, fill=LINES, width=w)
    image.alpha_composite(layer)


def _shirt(draw: ImageDraw.ImageDraw, cx: float, top: float, team: str, scale: float) -> None:
    body, sleeves = KITS.get(team, DEFAULT_KIT)

    def pts(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
        return [(cx + x * scale, top + y * scale) for x, y in points]

    outline = "#00000055"
    shirt = [(-16, 0), (-36, 6), (-54, 26), (-40, 46), (-30, 38), (-30, 96), (30, 96), (30, 38),
             (40, 46), (54, 26), (36, 6), (16, 0), (0, 14)]  # fmt: skip
    draw.polygon(pts(shirt), fill=body, outline=outline, width=2)
    for side in (-1, 1):
        sleeve = [(36 * side, 6), (54 * side, 26), (40 * side, 46), (30 * side, 38)]
        draw.polygon(pts(sleeve), fill=sleeves, outline=outline, width=2)
    draw.line(pts([(-16, 0), (0, 14), (16, 0)]), fill=outline, width=3)


def _badge(draw: ImageDraw.ImageDraw, xy: tuple[float, float], label: str, fill: str) -> None:
    x, y = xy
    draw.ellipse((x - 19, y - 19, x + 19, y + 19), fill=fill, outline=WHITE, width=3)
    _text(draw, (x, y), label, 26, WHITE if fill == INK else INK)


def _fixture_chip(draw: ImageDraw.ImageDraw, cx: float, y: float, player: dict[str, Any]) -> None:
    games = player.get("fixtures")
    if games is None:
        return
    if not games:
        label, (fill, ink) = "No game", ("#555555", WHITE)
    else:
        first = games[0]
        label = f"{first['opponent']} ({'H' if first['home'] else 'A'})"
        label += f" +{len(games) - 1}" if len(games) > 1 else ""
        fill, ink = DIFFICULTY.get(first["difficulty"], DIFFICULTY[3])
    width = draw.textlength(label, font=_font(24)) + 26
    draw.rounded_rectangle((cx - width / 2, y, cx + width / 2, y + 30), radius=15, fill=fill)
    _text(draw, (cx, y + 15), label, 24, ink)


def _player(
    draw: ImageDraw.ImageDraw,
    cx: float,
    top: float,
    player: dict[str, Any],
    new: bool,
    scale: float = 1.0,
) -> None:
    _shirt(draw, cx, top, player["team"], scale)
    if player["captain"]:
        _badge(draw, (cx + 46 * scale, top + 8 * scale), "C", INK)
    elif player["vice_captain"]:
        _badge(draw, (cx + 46 * scale, top + 8 * scale), "V", WHITE)
    if new:
        tag = (cx - 78 * scale, top - 4, cx - 22 * scale, top + 24)
        draw.rounded_rectangle(tag, radius=8, fill="#ffd400")
        _text(draw, ((tag[0] + tag[2]) / 2, top + 10), "NEW", 20, INK)

    # Name plates keep one width on the pitch and the bench (only the shirt shrinks), so names
    # read the same everywhere; long ones are shortened to fit.
    plate_w, y = 158, top + 104 * scale
    left, right = cx - plate_w / 2, cx + plate_w / 2
    draw.rounded_rectangle((left, y, right, y + 34), radius=8, fill=WHITE)
    _text(draw, (cx, y + 17), _fit(draw, player["name"], 29, plate_w - 12), 29, INK)

    draw.rounded_rectangle((left, y + 34, right, y + 66), radius=8, fill=PURPLE)
    _text(draw, (cx, y + 50), f"{player['xpts']:.1f} xPts", 27, ACCENT)
    _fixture_chip(draw, cx, y + 74, player)


def _header(draw: ImageDraw.ImageDraw, rec: dict[str, Any], best: dict[str, Any],
            deadline: dt.datetime | None) -> None:  # fmt: skip
    draw.rectangle((0, 0, WIDTH, HEADER), fill=PURPLE)
    draw.rectangle((0, HEADER - 8, WIDTH, HEADER), fill=ACCENT)
    _text(draw, (56, 70), f"GAMEWEEK {rec['gameweek']}", 80, WHITE, anchor="lm")
    if deadline is not None:
        uk = deadline.astimezone(LONDON)
        when = f"Deadline {uk:%a} {uk.day} {uk:%b}, {uk:%H:%M}"
        _text(draw, (58, 135), when, 36, "#d9c8e0", anchor="lm", bold=False)

    n = best["transfers"]
    verdict = "HOLD" if n == 0 else f"{n} TRANSFER{'S' if n > 1 else ''}"
    _text(draw, (WIDTH - 56, 62), verdict, 42, ACCENT, anchor="rm")
    gain = f"{best['gain_vs_hold']:+.1f} xPts" if n else f"{best['xpts']:.1f} xPts"
    _text(draw, (WIDTH - 56, 125), gain, 76, WHITE, anchor="rm")


def _bench(draw: ImageDraw.ImageDraw, best: dict[str, Any], new: set[int]) -> None:
    top = HEIGHT - BENCH
    draw.rectangle((0, top, WIDTH, HEIGHT), fill="#1f5c33")
    _text(draw, (40, top + 26), "BENCH", 30, "#cfe8d6", anchor="lm")
    _text(draw, (WIDTH - 40, top + 26), "FPL Optimiser", 26, "#cfe8d6", anchor="rm", bold=False)
    spacing = WIDTH / (len(best["bench"]) + 1)
    for i, player in enumerate(best["bench"], start=1):
        _player(draw, spacing * i, top + 44, player, player["id"] in new, scale=0.7)


def draw_pitch(rec: dict[str, Any], deadline: dt.datetime | None = None) -> bytes:
    """The recommended team as a PNG."""
    best = pick(rec)
    new = {m["in"]["id"] for m in best["moves"]}
    image = Image.new("RGBA", (WIDTH, HEIGHT), PURPLE)
    draw = ImageDraw.Draw(image)

    pitch_top, pitch_bottom = HEADER, HEIGHT - BENCH
    _grass(draw, pitch_top, pitch_bottom)
    _markings(image, pitch_top, pitch_bottom)
    draw = ImageDraw.Draw(image)
    _header(draw, rec, best, deadline)

    rows = [[p for p in best["starting_xi"] if p["position"] == pos]
            for pos in ("GK", "DEF", "MID", "FWD")]  # fmt: skip
    row_height = (pitch_bottom - pitch_top) / len(rows)
    for r, row in enumerate(rows):
        top = pitch_top + r * row_height + 18
        spacing = WIDTH / (len(row) + 1)
        for i, player in enumerate(row, start=1):
            _player(draw, spacing * i, top, player, player["id"] in new, scale=0.85)

    _bench(draw, best, new)
    out = io.BytesIO()
    image.convert("RGB").save(out, format="PNG", optimize=True)
    return out.getvalue()
