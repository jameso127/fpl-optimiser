"""Turn a recommendation (the optimise job's JSON) into Telegram messages (HTML).

Written to be read on a phone: the verdict comes first, then the transfers, the captain and the
team. Tables are monospace and under 30 characters wide, so they don't wrap on a small screen.
"""

import datetime as dt
from html import escape as _escape
from typing import Any
from zoneinfo import ZoneInfo

from notify.telegram import MAX_LENGTH

LONDON = ZoneInfo("Europe/London")
NAME_WIDTH = 13  # longer names are shortened with an ellipsis in the team table


def escape(text: str) -> str:
    """Telegram HTML only needs &, < and > escaped."""
    return _escape(text, quote=False)


def _pick(rec: dict[str, Any]) -> dict[str, Any]:
    return next(o for o in rec["options"] if o["transfers"] == rec["recommended_transfers"])


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _short(name: str) -> str:
    return name if len(name) <= NAME_WIDTH else name[: NAME_WIDTH - 1] + "…"


def _header(rec: dict[str, Any], deadline: dt.datetime | None) -> str:
    title = f"⚽ <b>Gameweek {rec['gameweek']}</b>"
    if deadline is None:
        return title
    uk = deadline.astimezone(LONDON)
    return f"{title}\n⏰ Deadline {uk:%a} {uk.day} {uk:%b}, {uk:%H:%M} UK"


def _verdict(rec: dict[str, Any], best: dict[str, Any]) -> str:
    n = best["transfers"]
    if n == 0:
        return (
            "✋ <b>Hold this week.</b>\n"
            f"No transfer adds at least {rec['min_gain_per_transfer']:.1f} pts, "
            "so save it for later."
        )
    hit = f"after a -{best['hit_cost']:.0f} hit" if best["hit_cost"] else "no hit"
    return (
        f"✅ <b>Make {_plural(n, 'transfer')}</b> for "
        f"<b>{best['gain_vs_hold']:+.1f} pts</b> expected ({hit})."
    )


def _move(i: int, move: dict[str, Any]) -> str:
    out, into = move["out"], move["in"]
    swing = into["xpts"] - out["xpts"]
    return (
        f"{i}. {escape(out['name'])} ➜ <b>{escape(into['name'])}</b>  <i>{swing:+.1f} pts</i>\n"
        f"     {escape(out['team'])} £{out['sell_price']:.1f}m ➜ "
        f"{escape(into['team'])} £{into['price']:.1f}m"
    )


def _transfers(best: dict[str, Any]) -> str:
    moves = "\n".join(_move(i, m) for i, m in enumerate(best["moves"], start=1))
    return f"🔁 <b>Transfers</b>\n{moves}\n💰 Bank after: £{best['bank_after']:.1f}m"


def _captaincy(best: dict[str, Any]) -> str:
    captain = next((p for p in best["starting_xi"] if p["captain"]), None)
    xpts = f" · {captain['xpts']:.1f} xPts ×2" if captain else ""
    return (
        f"🎖 Captain <b>{escape(best['captain'])}</b>{xpts}\n"
        f"     Vice: {escape(best['vice_captain'])}"
    )


def _xi_row(player: dict[str, Any], new: set[int]) -> str:
    role = "C" if player["captain"] else "V" if player["vice_captain"] else " "
    mark = "*" if player["id"] in new else " "
    short = _short(player["name"])
    name = escape(short) + " " * (NAME_WIDTH - len(short))  # pad by visible width, not escaped
    team = escape(player["team"])
    return f"{player['position']:<4}{name}{mark}{role} {team:<4}{player['xpts']:>4.1f}"


def _team(best: dict[str, Any]) -> str:
    new = {m["in"]["id"] for m in best["moves"]}
    rows = "\n".join(_xi_row(p, new) for p in best["starting_xi"])
    bench = ", ".join(escape(p["name"]) for p in best["bench"])
    key = "C captain · V vice" + (" · * new signing" if new else "")
    return (
        f"📋 <b>Starting XI</b> · {best['xpts']:.1f} xPts\n<pre>{rows}</pre>\n"
        f"<i>{key}</i>\n🪑 Bench: {bench}"
    )


def _option_row(option: dict[str, Any], picked: int) -> str:
    n = option["transfers"]
    marker = ">" if n == picked else " "
    moves = "hold" if n == 0 else str(n)
    hit = f"-{option['hit_cost']:.0f}" if option["hit_cost"] else ""
    return f"{marker} {moves:<5}{hit:>4}{option['net_xpts']:>7.1f}{option['gain_vs_hold']:>+7.1f}"


def _options(rec: dict[str, Any]) -> str:
    header = f"  {'Moves':<5}{'Hit':>4}{'xPts':>7}{'Gain':>7}"
    rows = "\n".join(_option_row(o, rec["recommended_transfers"]) for o in rec["options"])
    return (
        f"📊 <b>Every option</b>\n<pre>{header}\n{rows}</pre>\n"
        f"<i>&gt; = recommended. Each transfer must add at least "
        f"{rec['min_gain_per_transfer']:.1f} pts to be worth it.</i>"
    )


def _assumptions(rec: dict[str, Any]) -> str:
    lines = "\n".join(f"• {escape(line)}" for line in rec["assumptions"])
    return f"ℹ️ <b>Based on</b>\n{lines}"


def format_recommendation(rec: dict[str, Any], deadline: dt.datetime | None = None) -> list[str]:
    """One or more messages, each within Telegram's length limit."""
    best = _pick(rec)
    blocks = [f"{_header(rec, deadline)}\n\n{_verdict(rec, best)}"]
    if best["moves"]:
        blocks.append(_transfers(best))
    blocks += [
        _captaincy(best),
        _team(best),
        _options(rec),
        _assumptions(rec),
        "<i>Predictions are estimates, not guarantees. Good luck! 🍀</i>",
    ]
    return _split(blocks)


def _lines(block: str) -> list[str]:
    """A block that is too long for one message, cut at line breaks. Only plain-text blocks can
    get this long (the tables are a handful of lines), so no HTML tag is ever left open."""
    chunks: list[str] = []
    current = ""
    for line in block.split("\n"):
        line = line[:MAX_LENGTH]
        if current and len(current) + len(line) + 1 > MAX_LENGTH:
            chunks.append(current)
            current = ""
        current = f"{current}\n{line}" if current else line
    return [*chunks, current]


def _split(blocks: list[str]) -> list[str]:
    """Pack blocks into messages without cutting a block in half where it can be avoided."""
    messages: list[str] = []
    current = ""
    for block in blocks:
        for part in _lines(block) if len(block) > MAX_LENGTH else [block]:
            if current and len(current) + len(part) + 2 > MAX_LENGTH:
                messages.append(current)
                current = ""
            current = f"{current}\n\n{part}" if current else part
    if current:
        messages.append(current)
    return messages
