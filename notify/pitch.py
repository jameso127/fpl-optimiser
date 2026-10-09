"""Draw the recommended team on a pitch, as a PNG to send with the message.

Portrait (4:5), which fills a phone screen in Telegram. Layout, top to bottom: a header with
the gameweek, deadline and verdict; the starting XI in formation, goalkeeper at the top as in the
FPL app, each player a shirt in club colours with name, expected points and fixture (captain,
vice, new signings and injury doubts marked); the bench.
"""

import datetime as dt
from typing import Any
from zoneinfo import ZoneInfo

from PIL import Image, ImageDraw

from notify.draw import (
    ACCENT,
    INK,
    MUTED,
    PURPLE,
    RED,
    WHITE,
    YELLOW,
    badge,
    doubt,
    fit,
    fixture_chip,
    pill,
    png,
    shirt,
    text,
)  # fmt: skip
from notify.format import pick

WIDTH, HEIGHT = 1080, 1350
HEADER, BENCH = 200, 250  # heights of the bands above and below the pitch
LONDON = ZoneInfo("Europe/London")
GRASS = ("#2f8f4e", "#2a8547")  # alternating stripes
LINES = (255, 255, 255, 90)


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


def _player(
    draw: ImageDraw.ImageDraw,
    cx: float,
    top: float,
    player: dict[str, Any],
    new: bool,
    scale: float = 1.0,
) -> None:
    shirt(draw, cx, top, player["team"], scale)
    if player["captain"]:
        badge(draw, (cx + 46 * scale, top + 8 * scale), "C", INK)
    elif player["vice_captain"]:
        badge(draw, (cx + 46 * scale, top + 8 * scale), "V", WHITE)
    if new:
        pill(draw, cx - 50 * scale, top - 6, "NEW", YELLOW, INK, size=20)
    if (chance := doubt(player)) is not None:
        pill(draw, cx + 50 * scale, top + 52 * scale, f"{chance:.0f}%", RED, WHITE, size=20)

    # Name plates keep one width on the pitch and the bench (only the shirt shrinks), so names
    # read the same everywhere; long ones are shortened to fit.
    plate_w, y = 158, top + 104 * scale
    left, right = cx - plate_w / 2, cx + plate_w / 2
    draw.rounded_rectangle((left, y, right, y + 34), radius=8, fill=WHITE)
    text(draw, (cx, y + 17), fit(draw, player["name"], 29, plate_w - 12), 29, INK)

    draw.rounded_rectangle((left, y + 34, right, y + 66), radius=8, fill=PURPLE)
    text(draw, (cx, y + 50), f"{player['xpts']:.1f} xPts", 27, ACCENT)
    fixture_chip(draw, cx, y + 74, player)


def _header(draw: ImageDraw.ImageDraw, rec: dict[str, Any], best: dict[str, Any],
            deadline: dt.datetime | None) -> None:  # fmt: skip
    draw.rectangle((0, 0, WIDTH, HEADER), fill=PURPLE)
    draw.rectangle((0, HEADER - 8, WIDTH, HEADER), fill=ACCENT)
    text(draw, (56, 70), f"GAMEWEEK {rec['gameweek']}", 80, WHITE, anchor="lm")
    if deadline is not None:
        uk = deadline.astimezone(LONDON)
        when = f"Deadline {uk:%a} {uk.day} {uk:%b}, {uk:%H:%M}"
        text(draw, (58, 135), when, 36, MUTED, anchor="lm", bold=False)

    n = best["transfers"]
    verdict = "HOLD" if n == 0 else f"{n} TRANSFER{'S' if n > 1 else ''}"
    text(draw, (WIDTH - 56, 62), verdict, 42, ACCENT, anchor="rm")
    gain = f"{best['gain_vs_hold']:+.1f} xPts" if n else f"{best['xpts']:.1f} xPts"
    text(draw, (WIDTH - 56, 125), gain, 76, WHITE, anchor="rm")


def _bench(draw: ImageDraw.ImageDraw, best: dict[str, Any], new: set[int]) -> None:
    top = HEIGHT - BENCH
    draw.rectangle((0, top, WIDTH, HEIGHT), fill="#1f5c33")
    text(draw, (40, top + 26), "BENCH", 30, "#cfe8d6", anchor="lm")
    text(draw, (WIDTH - 40, top + 26), "FPL Optimiser", 26, "#cfe8d6", anchor="rm", bold=False)
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
    return png(image)
