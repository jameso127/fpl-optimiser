import itertools
from collections import Counter

import pytest

from optimise import model
from optimise.model import Player, Problem, Solution
from tests.fakes import synthetic_world


def _problem(seed: int = 7, free_transfers: int = 1, bank: int | None = None, w: float = 0.1):  # type: ignore[no-untyped-def]
    world = synthetic_world(seed)
    pool = tuple(
        Player(
            int(r["id"]),
            str(r["web_name"]),
            int(r["element_type"]),
            int(r["team"]),
            int(r["now_cost"]),
            float(r["xpts"]),
        )  # fmt: skip
        for r in world.players.to_dict("records")
    )
    selling = {i: world.purchase[i] for i in world.squad}
    return Problem(pool, selling, world.bank if bank is None else bank, free_transfers, w), world


def _by_id(problem: Problem) -> dict[int, Player]:
    return {p.id: p for p in problem.pool}


def _assert_legal(problem: Problem, sol: Solution) -> None:
    by_id = _by_id(problem)
    squad = [by_id[i] for i in (*sol.xi, *sol.bench)]
    xi = [by_id[i] for i in sol.xi]

    assert len(squad) == 15 and len({p.id for p in squad}) == 15
    assert Counter(p.position for p in squad) == {1: 2, 2: 5, 3: 5, 4: 3}
    assert max(Counter(p.team for p in squad).values()) <= 3
    assert len(xi) == 11
    counts = Counter(p.position for p in xi)
    assert counts[1] == 1 and 3 <= counts[2] <= 5 and 2 <= counts[3] <= 5 and 1 <= counts[4] <= 3
    assert sol.captain in sol.xi and sol.vice_captain in sol.xi and sol.captain != sol.vice_captain
    assert by_id[sol.captain].xpts == max(p.xpts for p in xi)
    # Bench: the spare goalkeeper first, then outfielders by expected points.
    bench = [by_id[i] for i in sol.bench]
    assert bench[0].position == 1 and all(p.position != 1 for p in bench[1:])
    assert [p.xpts for p in bench[1:]] == sorted((p.xpts for p in bench[1:]), reverse=True)
    # Exactly k players changed, and the money adds up without going overdrawn.
    assert len(sol.transfers_in) == len(sol.transfers_out) == sol.transfers
    spend = sum(by_id[i].price for i in sol.transfers_in)
    refund = sum(problem.selling_prices[i] for i in sol.transfers_out)
    assert sol.bank_after == problem.bank + refund - spend >= 0


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_every_option_is_a_legal_squad_xi_and_budget(seed: int) -> None:
    problem, _ = _problem(seed)
    options = model.solve_options(problem, 4)

    assert [o.transfers for o in options] == [0, 1, 2, 3, 4]
    for option in options:
        _assert_legal(problem, option)
        held = set(problem.selling_prices) - set(option.transfers_out)
        assert len(held) == 15 - option.transfers


def test_with_no_transfers_the_xi_and_captain_are_provably_the_best_available() -> None:
    problem, world = _problem(w=0.0)
    by_id = _by_id(problem)
    squad = [by_id[i] for i in world.squad]
    by_pos = {
        pos: sorted((p.xpts for p in squad if p.position == pos), reverse=True)
        for pos in (1, 2, 3, 4)
    }

    best = 0.0
    for d, m, f in itertools.product(range(3, 6), range(2, 6), range(1, 4)):
        if d + m + f != 10:
            continue
        xi = by_pos[1][:1] + by_pos[2][:d] + by_pos[3][:m] + by_pos[4][:f]
        best = max(best, sum(xi) + max(xi))  # the captain scores twice
    solution = model.solve(problem, 0)

    assert solution is not None
    assert solution.xi_xpts == pytest.approx(best)


def test_a_hit_is_charged_only_beyond_the_free_transfers() -> None:
    one, _ = _problem(free_transfers=1)
    two, _ = _problem(free_transfers=2)
    by_k_one = {o.transfers: o for o in model.solve_options(one, 3)}
    by_k_two = {o.transfers: o for o in model.solve_options(two, 3)}

    assert [by_k_one[k].hit_cost for k in range(4)] == [0, 0, 4, 8]
    assert [by_k_two[k].hit_cost for k in range(4)] == [0, 0, 0, 4]
    for option in by_k_one.values():
        assert option.net_xpts == pytest.approx(option.xi_xpts - option.hit_cost)


def test_a_clearly_better_affordable_player_is_bought() -> None:
    problem, world = _problem(bank=0)
    owned = set(world.squad)
    # Make every outside defender worthless except one cheap star from a club with room.
    clubs = Counter(p.team for p in problem.pool if p.id in owned)
    star = next(
        p for p in problem.pool
        if p.position == 2 and p.id not in owned and p.price <= 45 and clubs[p.team] < 3
    )  # fmt: skip

    def rated(p: Player) -> float:
        if p.id == star.id:
            return 40.0
        return 0.0 if p.position == 2 and p.id not in owned else p.xpts

    pool = tuple(Player(p.id, p.name, p.position, p.team, p.price, rated(p)) for p in problem.pool)
    boosted = Problem(pool, problem.selling_prices, problem.bank, problem.free_transfers)
    solution = model.solve(boosted, 1)

    assert solution is not None and star.id in solution.transfers_in
    hold = model.solve(boosted, 0)
    assert hold is not None and solution.net_xpts > hold.net_xpts + 10


def test_budget_uses_the_selling_price_of_players_already_owned() -> None:
    problem, world = _problem(bank=0)
    owned = set(world.squad)
    by_id = _by_id(problem)
    clubs = Counter(by_id[i].team for i in owned)
    position = 3
    top_sell = max(problem.selling_prices[i] for i in owned if by_id[i].position == position)
    target = next(
        p
        for p in problem.pool
        if p.position == position and p.id not in owned and clubs[p.team] < 3
    )
    star = Player(target.id, target.name, position, target.team, top_sell + 5, 60.0)
    pool = tuple(star if p.id == star.id else p for p in problem.pool)

    broke = Problem(pool, problem.selling_prices, 0, 1)
    rich = Problem(pool, problem.selling_prices, top_sell + 5, 1)
    a, b = model.solve(broke, 1), model.solve(rich, 1)

    assert a is not None and star.id not in a.transfers_in  # not affordable from any sale
    assert b is not None and star.id in b.transfers_in


def test_an_impossible_number_of_transfers_has_no_solution() -> None:
    problem, _ = _problem()
    assert model.solve(problem, 16) is None
    assert 16 not in [o.transfers for o in model.solve_options(problem, 16)]


def test_solving_is_deterministic() -> None:
    problem, _ = _problem()
    assert model.solve(problem, 2) == model.solve(problem, 2)


def test_an_illegal_current_squad_is_reported_not_papered_over() -> None:
    problem, world = _problem()
    by_id = _by_id(problem)
    bad = dict(problem.selling_prices)
    # Swap a defender for a sixth midfielder: 2 GK / 4 DEF / 6 MID / 3 FWD is not a legal squad.
    bad.pop(next(i for i in world.squad if by_id[i].position == 2))
    extra = next(p for p in problem.pool if p.position == 3 and p.id not in world.squad)
    bad[extra.id] = extra.price
    options = model.solve_options(Problem(problem.pool, bad, problem.bank, 1), 2)

    assert options and all(o.transfers != 0 for o in options)  # only repairs are possible
    with pytest.raises(ValueError, match="not a legal squad"):
        model.recommend(options)


def _sol(k: int, net: float) -> Solution:
    return Solution(k, 0.0, (), (), 0, 0, tuple(range(k)), tuple(range(k)), net, 0.0, net, 0)


def test_recommend_holds_unless_each_transfer_earns_its_place() -> None:
    options = [_sol(0, 40.0), _sol(1, 40.4), _sol(2, 40.8)]  # gains too small per transfer
    assert model.recommend(options, min_gain_per_transfer=0.5).transfers == 0


def test_recommend_takes_the_best_net_option_when_gains_are_clear() -> None:
    options = [_sol(0, 40.0), _sol(1, 42.0), _sol(2, 45.0), _sol(3, 44.0)]
    assert model.recommend(options, min_gain_per_transfer=0.5).transfers == 2


def test_recommend_prefers_fewer_transfers_when_the_gain_is_marginal() -> None:
    options = [_sol(0, 40.0), _sol(1, 43.0), _sol(2, 43.1)]  # the second transfer adds 0.1
    assert model.recommend(options).transfers == 1


def test_recommend_requires_the_hold_option() -> None:
    with pytest.raises(ValueError, match="not a legal squad"):
        model.recommend([_sol(1, 40.0)])
