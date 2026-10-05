"""Rebuild a manager's current squad, selling prices, bank and free transfers from the FPL API.

FPL does not expose selling prices or free transfers directly, so they are derived:

- Squad: the picks of the last finished gameweek, plus any transfers already made for the next
  one. If that last gameweek was a Free Hit, the team reverts afterwards, so the squad is
  taken from the gameweek before it.
- Selling price: purchase price plus half the profit, rounded down to 0.1m, or the market price
  if it has fallen. The purchase price is the cost of the most recent transfer in of that
  player (Free Hit and ignored weeks excluded); for players held since the start, the price in
  gameweek 1, read from the player's history.
- Transfers the manager has declared: FPL does not show a manager's changes for an upcoming
  gameweek until after its deadline (the picks endpoint returns 404), so the user tells the bot
  about transfers they have already made and they are applied on top, with the prices recorded
  when they were declared. A declared transfer that does not fit the squad (already visible in
  FPL, or declared wrongly) is ignored with a note rather than corrupting the squad.
- Free transfers: simulated week by week from the transfer history and chips (one new free
  transfer a week, banked up to a cap). Special top-ups FPL sometimes grants (e.g. around
  international tournaments) are not visible in the API, so this is an estimate: pass an
  override when it is wrong.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from common.users.models import DeclaredTransfer


class EntrySource(Protocol):
    """The per-manager FPL endpoints the optimiser needs."""

    def entry_history(self, team_id: int) -> dict[str, Any]: ...
    def entry_picks(self, team_id: int, gameweek: int) -> dict[str, Any]: ...
    def entry_transfers(self, team_id: int) -> list[dict[str, Any]]: ...
    def element_summary(self, player_id: int) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Squad:
    team_id: int
    base_gameweek: int  # the gameweek whose picks the squad was built from
    player_ids: tuple[int, ...]
    selling_prices: dict[int, int]  # tenths of a million
    bank: int  # tenths of a million, after any transfers already made for the next gameweek
    free_transfers: int  # still available for the next gameweek
    planned_transfers: int  # already made for the next gameweek (visible in FPL, or declared)
    declared_applied: tuple[tuple[int, int], ...] = ()  # (out, in) the user declared and we applied
    free_transfers_estimated: bool = True  # False when the user's override was used
    notes: tuple[str, ...] = field(default_factory=tuple)  # anything we ignored or assumed


def selling_price(purchase: int, now: int) -> int:
    """FPL's rule: half of any profit (rounded down to 0.1m) is yours; losses are in full."""
    if now <= purchase:
        return now
    return purchase + (now - purchase) // 2


def chip_weeks(chips: list[dict[str, Any]], name: str) -> set[int]:
    return {int(c["event"]) for c in chips if c.get("name") == name}


def base_gameweek(last_finished: int, chips: list[dict[str, Any]]) -> int:
    """The gameweek whose squad carries into the next one (the one before a Free Hit)."""
    base = last_finished
    freehit = chip_weeks(chips, "freehit")
    while base in freehit and base > 1:
        base -= 1
    return base


def free_transfers_available(history: dict[str, Any], next_gameweek: int, max_free: int = 5) -> int:
    """Free transfers at the start of `next_gameweek`, before any planned transfers.

    Each gameweek brings one, up to `max_free`; unused ones roll over. Using a Wildcard or Free
    Hit makes that week's transfers free without spending any.
    """
    made = {int(g["event"]): int(g["event_transfers"]) for g in history.get("current", [])}
    chips = history.get("chips", [])
    free_weeks = chip_weeks(chips, "wildcard") | chip_weeks(chips, "freehit")
    available = 1  # gameweek 2: one free transfer (gameweek 1 is the unlimited squad build)
    for gw in range(2, next_gameweek):
        used = made.get(gw, 0)
        left = available if gw in free_weeks else max(0, available - used)
        available = min(max_free, left + 1)
    return available


def current_squad(
    picks: dict[str, Any], transfers: list[dict[str, Any]], next_gameweek: int
) -> tuple[list[int], int, int]:
    """(player ids, bank, number of planned transfers) after the next gameweek's transfers."""
    ids = [int(p["element"]) for p in picks["picks"]]
    bank = int(picks["entry_history"]["bank"])
    planned = sorted(
        (t for t in transfers if int(t["event"]) == next_gameweek), key=lambda t: t["time"]
    )
    for t in planned:
        ids.remove(int(t["element_out"]))
        ids.append(int(t["element_in"]))
        bank += int(t["element_out_cost"]) - int(t["element_in_cost"])
    return ids, bank, len(planned)


def apply_declared(
    ids: list[int],
    bank: int,
    declared: Sequence[DeclaredTransfer],
    names: Mapping[int, str] | None = None,
) -> tuple[list[int], int, list[DeclaredTransfer], list[str]]:
    """Apply transfers the user says they have made, in the order they were declared.

    Returns the new squad, the new bank, the transfers applied, and notes about any that were
    skipped. A transfer is skipped, never forced, when it cannot be true of this squad:
    already reflected in what FPL shows, selling someone not owned, or unaffordable.
    """
    label = (lambda i: names.get(i, str(i))) if names else str
    ids = list(ids)
    applied: list[DeclaredTransfer] = []
    notes: list[str] = []
    for t in declared:
        out_here, in_here = t.out_id in ids, t.in_id in ids
        move = f"{label(t.out_id)} to {label(t.in_id)}"
        if out_here and not in_here and bank + t.out_price - t.in_price < 0:
            notes.append(f"Ignored your transfer {move}: you could not afford it.")
        elif out_here and not in_here:
            ids.remove(t.out_id)
            ids.append(t.in_id)
            bank += t.out_price - t.in_price
            applied.append(t)
        elif in_here and not out_here:
            notes.append(f"Your transfer {move} is already visible in your FPL team.")
        elif not out_here:
            notes.append(f"Ignored your transfer {move}: {label(t.out_id)} is not in your squad.")
        else:
            notes.append(
                f"Ignored your transfer {move}: {label(t.in_id)} is already in your squad."
            )
    return ids, bank, applied, notes


def purchase_prices(
    player_ids: list[int],
    transfers: list[dict[str, Any]],
    freehit_weeks: set[int],
    source: EntrySource,
    declared_costs: Mapping[int, int] | None = None,
) -> dict[int, int]:
    """Each player's purchase price: his latest transfer in, else his gameweek-1 price.

    `declared_costs` (players the user says they just bought, with what they paid) take
    precedence: they are the most recent purchases.
    """
    latest: dict[int, int] = {}
    for t in sorted(transfers, key=lambda t: t["time"]):
        if int(t["event"]) in freehit_weeks:
            continue  # a Free Hit squad is temporary; it does not change what you paid
        latest[int(t["element_in"])] = int(t["element_in_cost"])
    latest.update(declared_costs or {})
    prices: dict[int, int] = {}
    for pid in player_ids:
        if pid in latest:
            prices[pid] = latest[pid]
            continue
        rows = source.element_summary(pid)["history"]
        first = next((r for r in rows if int(r["round"]) == 1), rows[0])
        prices[pid] = int(first["value"])
    return prices


def build_squad(
    source: EntrySource,
    team_id: int,
    market_prices: dict[int, int],
    last_finished: int,
    next_gameweek: int,
    max_free_transfers: int = 5,
    free_transfers_override: int | None = None,
    declared: Sequence[DeclaredTransfer] = (),
    names: Mapping[int, str] | None = None,
) -> Squad:
    history = source.entry_history(team_id)
    chips = history.get("chips", [])
    base = base_gameweek(last_finished, chips)
    transfers = source.entry_transfers(team_id)
    picks = source.entry_picks(team_id, base)
    ids, bank, planned = current_squad(picks, transfers, next_gameweek)
    ids, bank, applied, notes = apply_declared(ids, bank, declared, names)
    planned += len(applied)
    bought_at = purchase_prices(
        ids,
        transfers,
        chip_weeks(chips, "freehit"),
        source,
        {t.in_id: t.in_price for t in applied},
    )
    selling = {pid: selling_price(bought_at[pid], market_prices[pid]) for pid in ids}

    if free_transfers_override is not None:
        free = free_transfers_override
    else:
        available = free_transfers_available(history, next_gameweek, max_free_transfers)
        free = max(0, available - planned)
    return Squad(
        team_id=team_id,
        base_gameweek=base,
        player_ids=tuple(ids),
        selling_prices=selling,
        bank=bank,
        free_transfers=free,
        planned_transfers=planned,
        declared_applied=tuple((t.out_id, t.in_id) for t in applied),
        free_transfers_estimated=free_transfers_override is None,
        notes=tuple(notes),
    )
