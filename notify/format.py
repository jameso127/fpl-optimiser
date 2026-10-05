"""Turn a recommendation (the optimise job's JSON) into Telegram messages (HTML)."""

from html import escape as _escape
from typing import Any

from notify.telegram import MAX_LENGTH


def escape(text: str) -> str:
    """Telegram HTML only needs &, < and > escaped."""
    return _escape(text, quote=False)


def _pick(rec: dict[str, Any]) -> dict[str, Any]:
    return next(o for o in rec["options"] if o["transfers"] == rec["recommended_transfers"])


def _row(player: dict[str, Any]) -> str:
    captain = " (C)" if player["captain"] else " (V)" if player["vice_captain"] else ""
    name = f"{player['name']}{captain}"
    return f"{player['position']:<4}{name:<18}{player['team']:<4}{player['xpts']:>5.1f}"


def _move(move: dict[str, Any]) -> str:
    out, into = move["out"], move["in"]
    return (
        f"OUT {escape(out['name'])} ({out['team']}, £{out['sell_price']:.1f}m)\n"
        f"IN  {escape(into['name'])} ({into['team']}, £{into['price']:.1f}m)"
    )


def _summary(option: dict[str, Any]) -> str:
    n = option["transfers"]
    label = "hold" if n == 0 else f"{n} transfer{'s' if n > 1 else ''}"
    hit = f", -{option['hit_cost']:.0f} hit" if option["hit_cost"] else ""
    return f"{label}: {option['net_xpts']:.1f} pts{hit} ({option['gain_vs_hold']:+.1f})"


def format_recommendation(rec: dict[str, Any]) -> list[str]:
    """One or more messages, each within Telegram's length limit."""
    best = _pick(rec)
    n = best["transfers"]
    verdict = (
        "Hold: no transfer is worth it."
        if n == 0
        else f"Make {n} transfer{'s' if n > 1 else ''}: {best['gain_vs_hold']:+.1f} points "
        f"({'no hit' if not best['hit_cost'] else f'-{best["hit_cost"]:.0f} hit included'})."
    )
    blocks = [f"<b>Gameweek {rec['gameweek']}</b>\n{verdict}"]
    if best["moves"]:
        blocks.append(
            "<b>Transfers</b>\n<pre>" + "\n\n".join(_move(m) for m in best["moves"]) + "</pre>"
        )

    xi = "\n".join(_row(p) for p in best["starting_xi"])
    bench = ", ".join(escape(p["name"]) for p in best["bench"])
    blocks.append(f"<b>Starting XI</b> (xPts)\n<pre>{escape(xi)}</pre>Bench: {bench}")

    blocks.append(
        "<b>All options</b>\n<pre>" + "\n".join(_summary(o) for o in rec["options"]) + "</pre>"
    )
    blocks.append(
        "<b>Based on</b>\n" + "\n".join(f"• {escape(line)}" for line in rec["assumptions"])
    )
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
