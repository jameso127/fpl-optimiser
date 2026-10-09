"""Turn a recommendation (the optimise job's JSON) into Telegram messages (HTML).

Written to be read on a phone. The summary (deadline, verdict, captain) is short enough to be the
caption under the pitch picture; the details follow, with the long parts (every player, every
option, the assumptions) folded into expandable quotes so the message stays short until tapped.
"""

import datetime as dt
from html import escape as _escape
from typing import Any
from zoneinfo import ZoneInfo

from notify.telegram import MAX_LENGTH

LONDON = ZoneInfo("Europe/London")
CAPTION_LENGTH = 1024  # Telegram's limit for a photo caption
EXPAND_OPEN, EXPAND_CLOSE = "<blockquote expandable>", "</blockquote>"

# FPL's fixture difficulty rating, 1 (easiest) to 5, as a coloured dot.
DIFFICULTY = {1: "🟢", 2: "🟢", 3: "⚪", 4: "🟠", 5: "🔴"}


def escape(text: str) -> str:
    """Telegram HTML only needs &, < and > escaped."""
    return _escape(text, quote=False)


def pick(rec: dict[str, Any]) -> dict[str, Any]:
    """The recommended option."""
    return next(o for o in rec["options"] if o["transfers"] == rec["recommended_transfers"])


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _expandable(title: str, lines: list[str]) -> str:
    return f"{EXPAND_OPEN}<b>{title}</b>\n" + "\n".join(lines) + EXPAND_CLOSE


# --- player facts -----------------------------------------------------------------------------


def fixtures_text(player: dict[str, Any]) -> str:
    """e.g. '🟠 ARS (A)'; 'no game' in a blank gameweek; '' when fixtures are unknown."""
    games = player.get("fixtures")
    if games is None:
        return ""
    if not games:
        return "no game"
    return ", ".join(
        f"{DIFFICULTY.get(g['difficulty'], '')} {escape(g['opponent'])} "
        f"({'H' if g['home'] else 'A'})"
        for g in games
    )


def _facts(player: dict[str, Any]) -> str:
    parts = [fixtures_text(player)]
    if "form" in player:
        parts.append(f"form {player['form']:.1f}")
    if "ownership" in player:
        parts.append(f"{player['ownership']:.0f}% owned")
    return " · ".join(p for p in parts if p)


def _doubt(player: dict[str, Any]) -> str | None:
    """'75%: Knock' for a player who might not play, else None."""
    chance = player.get("chance_of_playing")
    if chance is None or chance >= 100:
        return None
    news = player.get("news")
    return f"{chance:.0f}%" + (f": {escape(news)}" if news else "")


# --- the summary (the photo caption) ----------------------------------------------------------


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


def _captaincy(best: dict[str, Any]) -> str:
    captain = next((p for p in best["starting_xi"] if p["captain"]), None)
    xpts = f" · {captain['xpts']:.1f} xPts ×2" if captain else ""
    return (
        f"🎖 Captain <b>{escape(best['captain'])}</b>{xpts}\n"
        f"     Vice: {escape(best['vice_captain'])}"
    )


def summary(rec: dict[str, Any], deadline: dt.datetime | None = None) -> str:
    """Deadline, verdict and captain: short enough for a photo caption (CAPTION_LENGTH)."""
    best = pick(rec)
    return f"{_header(rec, deadline)}\n\n{_verdict(rec, best)}\n\n{_captaincy(best)}"


# --- the details ------------------------------------------------------------------------------


def _move(i: int, move: dict[str, Any]) -> str:
    out, into = move["out"], move["in"]
    swing = into["xpts"] - out["xpts"]
    lines = [
        f"{i}. {escape(out['name'])} ➜ <b>{escape(into['name'])}</b>  <i>{swing:+.1f} pts</i>",
        f"     {escape(out['team'])} £{out['sell_price']:.1f}m ➜ "
        f"{escape(into['team'])} £{into['price']:.1f}m",
    ]
    if facts := _facts(into):
        lines.append(f"     {facts}")
    if doubt := _doubt(out):
        lines.append(f"     🚑 {escape(out['name'])} {doubt}")
    return "\n".join(lines)


def _transfers(best: dict[str, Any]) -> str:
    moves = "\n".join(_move(i, m) for i, m in enumerate(best["moves"], start=1))
    return f"🔁 <b>Transfers</b>\n{moves}\n💰 Bank after: £{best['bank_after']:.1f}m"


def _squad_notes(best: dict[str, Any]) -> str:
    bench = ", ".join(escape(p["name"]) for p in best["bench"])
    lines = [f"🪑 Bench: {bench}"]
    doubts = [
        f"{escape(p['name'])} {d}"
        for p in [*best["starting_xi"], *best["bench"]]
        if (d := _doubt(p))
    ]
    if doubts:
        lines.append("⚠️ <b>Fitness doubts</b>\n" + "\n".join(f"• {d}" for d in doubts))
    return "\n".join(lines)


def _player_line(player: dict[str, Any], new: set[int]) -> str:
    role = " (C)" if player["captain"] else " (V)" if player["vice_captain"] else ""
    badge = "🆕 " if player["id"] in new else ""
    facts = _facts(player)
    return (
        f"{badge}<b>{escape(player['name'])}</b>{role} {escape(player['team'])} · "
        f"{player['xpts']:.1f} xPts" + (f" · {facts}" if facts else "")
    )


def _players(best: dict[str, Any]) -> str:
    new = {m["in"]["id"] for m in best["moves"]}
    lines = [_player_line(p, new) for p in best["starting_xi"]]
    return _expandable(f"🔎 Starting XI · {best['xpts']:.1f} xPts", lines)


def _option_line(option: dict[str, Any], picked: int) -> str:
    n = option["transfers"]
    label = "Hold" if n == 0 else _plural(n, "transfer")
    hit = f", -{option['hit_cost']:.0f} hit" if option["hit_cost"] else ""
    text = f"{label}{hit}: {option['net_xpts']:.1f} pts ({option['gain_vs_hold']:+.1f})"
    return f"▶️ <b>{text}</b>" if n == picked else f"▫️ {text}"


def _options(rec: dict[str, Any]) -> str:
    lines = [_option_line(o, rec["recommended_transfers"]) for o in rec["options"]]
    lines.append(
        f"<i>Each transfer must add at least {rec['min_gain_per_transfer']:.1f} pts "
        "to be worth it.</i>"
    )
    return _expandable("📊 Every option", lines)


def _assumptions(rec: dict[str, Any]) -> str:
    return _expandable("ℹ️ Based on", [f"• {escape(line)}" for line in rec["assumptions"]])


def details(rec: dict[str, Any]) -> list[str]:
    """Everything after the summary, as messages within Telegram's length limit."""
    best = pick(rec)
    blocks = [_transfers(best)] if best["moves"] else []
    blocks += [
        _squad_notes(best),
        _players(best),
        _options(rec),
        _assumptions(rec),
        "<i>Predictions are estimates, not guarantees. Good luck! 🍀</i>",
    ]
    return _split(blocks)


def format_recommendation(rec: dict[str, Any], deadline: dt.datetime | None = None) -> list[str]:
    """The whole recommendation as text: used when there is no picture to put the summary under."""
    return _split([summary(rec, deadline), *details(rec)])


# --- fitting Telegram's length limit ----------------------------------------------------------


def _lines(block: str, limit: int) -> list[str]:
    """A block that is too long for one message, cut at line breaks. Tags never span lines
    (except the expandable quote, which `_parts` re-opens per chunk), so none is left open."""
    chunks: list[str] = []
    current = ""
    for line in block.split("\n"):
        line = line[:limit]
        if current and len(current) + len(line) + 1 > limit:
            chunks.append(current)
            current = ""
        current = f"{current}\n{line}" if current else line
    return [*chunks, current]


def _parts(block: str) -> list[str]:
    if len(block) <= MAX_LENGTH:
        return [block]
    if block.startswith(EXPAND_OPEN) and block.endswith(EXPAND_CLOSE):
        inner = block[len(EXPAND_OPEN) : -len(EXPAND_CLOSE)]
        limit = MAX_LENGTH - len(EXPAND_OPEN) - len(EXPAND_CLOSE)
        return [f"{EXPAND_OPEN}{chunk}{EXPAND_CLOSE}" for chunk in _lines(inner, limit)]
    return _lines(block, MAX_LENGTH)


def _split(blocks: list[str]) -> list[str]:
    """Pack blocks into messages without cutting a block in half where it can be avoided."""
    messages: list[str] = []
    current = ""
    for block in blocks:
        for part in _parts(block):
            if current and len(current) + len(part) + 2 > MAX_LENGTH:
                messages.append(current)
                current = ""
            current = f"{current}\n\n{part}" if current else part
    if current:
        messages.append(current)
    return messages
