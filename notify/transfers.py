"""Draw the recommended transfers, as a PNG to send after the pitch picture.

One card per transfer: the player going out on the left, the one coming in on the right, the
expected-points swing between them, and under each a bar chart of their points over the last
few gameweeks (both charts on the same scale, so they compare at a glance). At the bottom, the
expected gain for every number of transfers, the recommended one highlighted.
"""

from typing import Any

from PIL import Image, ImageDraw

from notify.draw import (
    ACCENT,
    DEEP,
    INK,
    MUTED,
    PURPLE,
    RED,
    WHITE,
    doubt,
    fit,
    fixture_chip,
    pill,
    png,
    shirt,
    text,
)  # fmt: skip
from notify.format import pick

WIDTH = 1080
HEADER, CARD, GAP, OPTIONS = 180, 410, 24, 330
MARGIN = 32
OUT_BAR, IN_BAR, EMPTY_BAR, OTHER_OPTION = "#ff6b81", ACCENT, "#5a4a62", "#6d5a75"


def _side(
    draw: ImageDraw.ImageDraw,
    cx: float,
    top: float,
    player: dict[str, Any],
    label: str,
    price: float,
    scale_max: float,
    bar: str,
) -> None:
    """One player in a card: label, shirt, name, club and price, xPts, fixture, form chart."""
    pill(
        draw,
        cx,
        top + 18,
        label,
        RED if label == "OUT" else ACCENT,
        WHITE if label == "OUT" else INK,
    )

    shirt_x = cx - 150
    shirt(draw, shirt_x, top + 66, player["team"], 0.8)
    fixture_chip(draw, shirt_x, top + 154, player)
    if (chance := doubt(player)) is not None:
        pill(draw, shirt_x, top + 190, f"{chance:.0f}% fit", RED, WHITE, size=20)

    x = cx - 80
    text(draw, (x, top + 90), fit(draw, player["name"], 42, 290), 42, WHITE, anchor="lm")
    text(draw, (x, top + 132), f"{player['team']} · £{price:.1f}m", 28, MUTED, "lm", bold=False)
    text(draw, (x, top + 172), f"{player['xpts']:.1f} xPts", 32, ACCENT, anchor="lm")
    _form_chart(draw, cx, top + 236, player.get("recent", []), scale_max, bar)


def _form_chart(
    draw: ImageDraw.ImageDraw,
    cx: float,
    top: float,
    recent: list[dict[str, Any]],
    scale_max: float,
    colour: str,
) -> None:
    """Points per gameweek as bars, value above each, gameweek below. Weeks without a minute
    played get a flat grey stub."""
    if not recent:
        text(draw, (cx, top + 70), "No recent games", 26, MUTED, bold=False)
        return
    total = sum(g["points"] for g in recent)
    text(draw, (cx, top + 8), f"{total} pts in last {len(recent)}", 24, MUTED, bold=False)
    baseline, tallest, slot = top + 128, 74, 84
    left = cx - slot * len(recent) / 2
    for i, game in enumerate(recent):
        mid = left + slot * (i + 0.5)
        played = game["minutes"] > 0
        height = max(4.0, tallest * max(game["points"], 0) / scale_max) if played else 4.0
        draw.rounded_rectangle((mid - 24, baseline - height, mid + 24, baseline), radius=6,
                               fill=colour if played else EMPTY_BAR)  # fmt: skip
        value = str(game["points"]) if played else "–"
        text(draw, (mid, baseline - height - 16), value, 24, WHITE)
        text(draw, (mid, baseline + 18), f"GW{game['gameweek']}", 20, MUTED, bold=False)


def _arrow(draw: ImageDraw.ImageDraw, cx: float, cy: float, swing: float) -> None:
    draw.ellipse((cx - 38, cy - 38, cx + 38, cy + 38), fill=ACCENT if swing >= 0 else RED)
    draw.polygon([(cx - 18, cy - 8), (cx + 2, cy - 8), (cx + 2, cy - 20), (cx + 22, cy),
                  (cx + 2, cy + 20), (cx + 2, cy + 8), (cx - 18, cy + 8)], fill=INK)  # fmt: skip
    text(draw, (cx, cy + 66), f"{swing:+.1f}", 36, ACCENT if swing >= 0 else RED)
    text(draw, (cx, cy + 98), "xPts", 22, MUTED, bold=False)


def _card(draw: ImageDraw.ImageDraw, top: float, move: dict[str, Any]) -> None:
    out, into = move["out"], move["in"]
    draw.rounded_rectangle((MARGIN, top, WIDTH - MARGIN, top + CARD), radius=28, fill=PURPLE)
    points = [g["points"] for g in [*out.get("recent", []), *into.get("recent", [])]]
    scale_max = max([8, *points])
    _side(draw, 285, top, out, "OUT", out["sell_price"], scale_max, OUT_BAR)
    _side(draw, 795, top, into, "IN", into["price"], scale_max, IN_BAR)
    _arrow(draw, WIDTH / 2, top + 110, into["xpts"] - out["xpts"])


def _header(draw: ImageDraw.ImageDraw, best: dict[str, Any]) -> None:
    draw.rectangle((0, 0, WIDTH, HEADER), fill=PURPLE)
    draw.rectangle((0, HEADER - 8, WIDTH, HEADER), fill=ACCENT)
    text(draw, (56, 72), "TRANSFERS", 76, WHITE, anchor="lm")
    hit = f"-{best['hit_cost']:.0f} hit" if best["hit_cost"] else "no hit"
    sub = f"Bank after £{best['bank_after']:.1f}m · {hit}"
    text(draw, (58, 132), sub, 34, MUTED, anchor="lm", bold=False)
    text(draw, (WIDTH - 56, 72), f"{best['gain_vs_hold']:+.1f} xPts", 76, ACCENT, anchor="rm")
    text(draw, (WIDTH - 56, 132), "vs keeping your team", 30, MUTED, anchor="rm", bold=False)


def _options(draw: ImageDraw.ImageDraw, top: float, rec: dict[str, Any]) -> None:
    """Expected gain against holding for 0..N transfers (hits included)."""
    options = rec["options"]
    draw.rounded_rectangle((MARGIN, top, WIDTH - MARGIN, top + OPTIONS - GAP), radius=28,
                           fill=PURPLE)  # fmt: skip
    text(draw, (WIDTH / 2, top + 36), "EXPECTED GAIN BY NUMBER OF TRANSFERS", 28, MUTED)
    gains = [o["gain_vs_hold"] for o in options]
    high, low = max([*gains, 0.5]), min([*gains, 0.0])
    chart_top, chart_bottom = top + 100, top + OPTIONS - 90
    zero = chart_top + (chart_bottom - chart_top) * high / (high - low)
    per_point = (chart_bottom - chart_top) / (high - low)
    slot = (WIDTH - 2 * MARGIN - 80) / len(options)
    draw.line((MARGIN + 30, zero, WIDTH - MARGIN - 30, zero), fill=MUTED, width=2)
    for i, option in enumerate(options):
        mid = MARGIN + 40 + slot * (i + 0.5)
        gain = option["gain_vs_hold"]
        picked = option["transfers"] == rec["recommended_transfers"]
        end = zero - gain * per_point
        draw.rounded_rectangle((mid - 34, min(zero, end), mid + 34, max(zero, end) + 1), radius=6,
                               fill=ACCENT if picked else OTHER_OPTION)  # fmt: skip
        value_y = end - 18 if gain >= 0 else end + 18
        text(draw, (mid, value_y), f"{gain:+.1f}", 26, WHITE)
        n = option["transfers"]
        label = "Hold" if n == 0 else str(n)
        text(draw, (mid, chart_bottom + 34), label, 28, ACCENT if picked else WHITE)
        if option["hit_cost"]:
            text(draw, (mid, chart_bottom + 62), f"-{option['hit_cost']:.0f} hit", 20, MUTED,
                 bold=False)  # fmt: skip


def draw_transfers(rec: dict[str, Any]) -> bytes | None:
    """The recommended transfers as a PNG, or None when the advice is to hold."""
    best = pick(rec)
    if not best["moves"]:
        return None
    height = HEADER + GAP + len(best["moves"]) * (CARD + GAP) + OPTIONS
    image = Image.new("RGBA", (WIDTH, height), DEEP)
    draw = ImageDraw.Draw(image)
    _header(draw, best)
    for i, move in enumerate(best["moves"]):
        _card(draw, HEADER + GAP + i * (CARD + GAP), move)
    _options(draw, HEADER + GAP + len(best["moves"]) * (CARD + GAP), rec)
    return png(image)
