from typing import Any

import pytest

from optimise import squad

SQUAD_IDS = list(range(1, 16))


def test_selling_price_is_half_the_profit_rounded_down_and_losses_in_full() -> None:
    assert squad.selling_price(50, 50) == 50
    assert squad.selling_price(50, 48) == 48  # a loss is taken in full
    assert squad.selling_price(50, 51) == 50  # +0.1m: half rounds down to nothing
    assert squad.selling_price(50, 52) == 51
    assert squad.selling_price(50, 55) == 52  # +0.5m: 0.2m of it is yours
    assert squad.selling_price(41, 60) == 50


def test_the_squad_reverts_to_before_a_free_hit() -> None:
    chips = [{"name": "freehit", "event": 5}]
    assert squad.base_gameweek(5, chips) == 4
    assert squad.base_gameweek(4, chips) == 4
    assert squad.base_gameweek(6, chips) == 6
    assert squad.base_gameweek(5, [{"name": "wildcard", "event": 5}]) == 5


def _history(
    transfers_by_gw: dict[int, int], chips: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    return {
        "current": [{"event": gw, "event_transfers": n} for gw, n in transfers_by_gw.items()],
        "chips": chips or [],
    }


def test_unused_free_transfers_bank_up_to_the_cap() -> None:
    quiet = _history({gw: 0 for gw in range(1, 6)})

    assert squad.free_transfers_available(quiet, 2) == 1
    assert squad.free_transfers_available(quiet, 4) == 3
    assert squad.free_transfers_available(quiet, 6) == 5  # gameweeks 2-5 banked 4, plus one
    assert squad.free_transfers_available(quiet, 9, max_free=5) == 5  # capped
    assert squad.free_transfers_available(quiet, 9, max_free=2) == 2  # an older cap


def test_using_free_transfers_spends_them_and_hits_leave_nothing() -> None:
    history = _history({1: 0, 2: 1, 3: 0, 4: 3})
    # GW2: 1 available, 1 used -> 0, +1 = 1.  GW3: unused -> 1, +1 = 2.
    # GW4: 2 available, 3 used (one hit) -> 0, +1 = 1 for GW5.
    assert squad.free_transfers_available(history, 5) == 1


def test_a_wildcard_or_free_hit_week_costs_no_free_transfers() -> None:
    wildcard = _history({1: 0, 2: 0, 3: 8}, [{"name": "wildcard", "event": 3}])
    freehit = _history({1: 0, 2: 0, 3: 9}, [{"name": "freehit", "event": 3}])
    normal = _history({1: 0, 2: 0, 3: 8})

    assert squad.free_transfers_available(wildcard, 4) == 3  # 2 banked, chip week unspent, +1
    assert squad.free_transfers_available(freehit, 4) == 3
    assert squad.free_transfers_available(normal, 4) == 1


def _picks(bank: int = 5) -> dict[str, Any]:
    return {"entry_history": {"bank": bank}, "picks": [{"element": i} for i in SQUAD_IDS]}


def _transfer(
    event: int, time: str, out: int, into: int, cost_out: int, cost_in: int
) -> dict[str, Any]:
    return {
        "event": event, "time": time, "element_out": out, "element_out_cost": cost_out,
        "element_in": into, "element_in_cost": cost_in,
    }  # fmt: skip


def test_transfers_already_made_for_the_next_gameweek_are_applied_with_their_cash() -> None:
    transfers = [
        _transfer(4, "2025-09-01T10:00:00Z", 1, 99, 50, 50),  # an old transfer: ignored here
        _transfer(5, "2025-09-10T10:00:00Z", 2, 20, 55, 60),
        _transfer(5, "2025-09-10T11:00:00Z", 3, 21, 45, 40),
    ]
    ids, bank, planned = squad.current_squad(_picks(bank=5), transfers, next_gameweek=5)

    assert sorted(ids) == sorted([*(set(SQUAD_IDS) - {2, 3}), 20, 21])
    assert bank == 5 + (55 - 60) + (45 - 40)
    assert planned == 2


class FakeSource:
    def __init__(
        self,
        history: dict[str, Any],
        picks: dict[str, Any],
        transfers: list[dict[str, Any]],
        first_prices: dict[int, int],
    ) -> None:
        self._history, self._picks, self._transfers = history, picks, transfers
        self._first = first_prices
        self.picks_requested: list[int] = []
        self.summaries_requested: list[int] = []

    def entry_history(self, team_id: int) -> dict[str, Any]:
        return self._history

    def entry_picks(self, team_id: int, gameweek: int) -> dict[str, Any]:
        self.picks_requested.append(gameweek)
        return self._picks

    def entry_transfers(self, team_id: int) -> list[dict[str, Any]]:
        return self._transfers

    def element_summary(self, player_id: int) -> dict[str, Any]:
        self.summaries_requested.append(player_id)
        return {
            "history": [{"round": 3, "value": 999}, {"round": 1, "value": self._first[player_id]}]
        }


def test_purchase_price_is_the_latest_transfer_in_else_the_gameweek_one_price() -> None:
    transfers = [
        _transfer(2, "2025-08-20T10:00:00Z", 9, 5, 50, 60),
        _transfer(6, "2025-10-01T10:00:00Z", 5, 9, 70, 52),  # sold 5, bought 9 back
        _transfer(8, "2025-10-20T10:00:00Z", 9, 5, 53, 66),  # and 5 again, dearer
    ]
    source = FakeSource({}, _picks(), transfers, {i: 40 + i for i in SQUAD_IDS})
    prices = squad.purchase_prices([5, 9, 4], transfers, set(), source)

    assert prices == {5: 66, 9: 52, 4: 44}
    assert source.summaries_requested == [4]  # only the never-transferred player is looked up


def test_free_hit_week_transfers_do_not_change_what_you_paid() -> None:
    transfers = [_transfer(7, "2025-10-05T10:00:00Z", 1, 5, 50, 80)]  # a Free Hit punt
    source = FakeSource({}, _picks(), transfers, {i: 45 for i in SQUAD_IDS})

    assert squad.purchase_prices([5], transfers, {7}, source) == {5: 45}


def test_build_squad_combines_everything() -> None:
    history = _history({1: 0, 2: 0, 3: 1, 4: 0}, [])
    transfers = [_transfer(3, "2025-09-01T10:00:00Z", 1, 15, 50, 58)]
    source = FakeSource(history, _picks(bank=7), transfers, {i: 50 for i in SQUAD_IDS})
    market = {i: 50 for i in SQUAD_IDS}
    market[15] = 64  # +0.6m since he was bought at 5.8m
    built = squad.build_squad(source, 123, market, last_finished=4, next_gameweek=5)

    assert built.base_gameweek == 4 and source.picks_requested == [4]
    assert built.selling_prices[15] == 58 + (64 - 58) // 2 == 61
    assert built.selling_prices[3] == 50  # unchanged price: sells for what was paid
    assert built.bank == 7
    # GW2: 1 unused -> 2 at GW3. GW3: 1 used -> 1 left -> 2 at GW4. GW4: unused -> 3 for GW5.
    assert built.free_transfers == 3 and built.planned_transfers == 0


def test_build_squad_after_a_free_hit_uses_the_squad_from_before_it() -> None:
    history = _history({1: 0, 2: 0, 3: 0, 4: 11}, [{"name": "freehit", "event": 4}])
    source = FakeSource(history, _picks(), [], {i: 50 for i in SQUAD_IDS})
    built = squad.build_squad(
        source, 1, {i: 50 for i in SQUAD_IDS}, last_finished=4, next_gameweek=5
    )

    assert source.picks_requested == [3]
    assert built.base_gameweek == 3
    assert built.free_transfers == 4  # the Free Hit week spent none: 1+1+1, then +1


def test_planned_transfers_use_up_free_transfers_and_an_override_wins() -> None:
    history = _history({1: 0, 2: 0, 3: 0})
    transfers = [_transfer(4, "2025-09-30T10:00:00Z", 2, 20, 50, 50)]
    source = FakeSource(history, _picks(), transfers, {i: 50 for i in SQUAD_IDS} | {20: 50})
    market = {i: 50 for i in [*SQUAD_IDS, 20]}

    planned = squad.build_squad(source, 1, market, last_finished=3, next_gameweek=4)
    assert planned.planned_transfers == 1 and planned.free_transfers == 3 - 1  # 3 available, 1 used
    forced = squad.build_squad(source, 1, market, 3, 4, free_transfers_override=1)
    assert forced.free_transfers == 1


def test_selling_a_player_whose_price_fell_gives_back_the_market_price() -> None:
    source = FakeSource(_history({1: 0}), _picks(), [], {i: 50 for i in SQUAD_IDS})
    market = {i: 50 for i in SQUAD_IDS} | {7: 46}
    built = squad.build_squad(source, 1, market, last_finished=1, next_gameweek=2)

    assert built.selling_prices[7] == 46


@pytest.mark.parametrize("last_finished", [1, 2, 3])
def test_free_transfers_before_any_history_is_one(last_finished: int) -> None:
    assert squad.free_transfers_available({"current": [], "chips": []}, 2) == 1
