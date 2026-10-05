"""Squad and transfer optimiser: a mixed-integer program solved with HiGHS (via SciPy).

For each number of transfers k it finds the best squad reachable from the current one with
exactly k transfers, together with the starting XI, captain and bench order.

Variables, for every player i in the pool (all binary):
    s_i  in the 15-man squad
    x_i  in the starting XI
    c_i  is the captain (scores double)

Objective (maximise):
    sum_i xpts_i * (x_i + c_i)  +  w * sum_i xpts_i * (s_i - x_i)
i.e. the starters once, the captain a second time, and a small weight w on the bench (it only
scores when a starter does not play, which the expected points already reflect, so this is
mostly a tie-break and a mild preference for a useful bench). A transfer hit of 4 points per
transfer beyond the free ones is a constant for a fixed k, subtracted afterwards.

Constraints:
    squad size 15; 2 GK, 5 DEF, 5 MID, 3 FWD; at most 3 players per club
    budget: sum of buy prices <= selling prices of the current squad + bank, where a player
        already owned costs his *selling* price (not the market price) and everyone else his
        market price
    XI: exactly 11, 1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD, every starter is in the squad
    exactly one captain, who starts
    transfers: exactly k current players are not in the new squad (so k are brought in)
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

SQUAD_SIZE = 15
XI_SIZE = 11
MAX_PER_CLUB = 3
SQUAD_BY_POSITION = {1: 2, 2: 5, 3: 5, 4: 3}  # GK, DEF, MID, FWD
XI_BOUNDS = {1: (1, 1), 2: (3, 5), 3: (2, 5), 4: (1, 3)}
HIT_COST = 4.0
POSITION_NAMES = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


@dataclass(frozen=True)
class Player:
    id: int
    name: str
    position: int  # 1 GK, 2 DEF, 3 MID, 4 FWD
    team: int
    price: int  # market price in tenths of a million
    xpts: float


@dataclass(frozen=True)
class Problem:
    pool: tuple[Player, ...]
    selling_prices: Mapping[int, int]  # current squad: player id -> selling price (tenths)
    bank: int  # tenths of a million
    free_transfers: int
    bench_weight: float = 0.1

    @property
    def budget(self) -> int:
        return sum(self.selling_prices.values()) + self.bank

    def buy_price(self, player: Player) -> int:
        """What the player costs to hold: the selling price if already owned."""
        return self.selling_prices.get(player.id, player.price)


@dataclass(frozen=True)
class Solution:
    transfers: int
    hit_cost: float
    xi: tuple[int, ...]  # by position, then expected points
    bench: tuple[int, ...]  # the goalkeeper first, then outfielders by expected points
    captain: int
    vice_captain: int
    transfers_in: tuple[int, ...]
    transfers_out: tuple[int, ...]
    xi_xpts: float  # starters plus the captain's extra, before any hit
    bench_xpts: float
    net_xpts: float  # xi_xpts minus the hit
    bank_after: int


class _Rows:
    """Sparse constraint rows: sum(coef * var) between lb and ub."""

    def __init__(self, n_vars: int) -> None:
        self._n = n_vars
        self._r: list[int] = []
        self._c: list[int] = []
        self._v: list[float] = []
        self.lb: list[float] = []
        self.ub: list[float] = []

    def add(self, cols: list[int], coefs: list[float], lb: float, ub: float) -> None:
        row = len(self.lb)
        self._r += [row] * len(cols)
        self._c += cols
        self._v += coefs
        self.lb.append(lb)
        self.ub.append(ub)

    def matrix(self) -> coo_matrix:
        return coo_matrix((self._v, (self._r, self._c)), shape=(len(self.lb), self._n))


def hit_cost(transfers: int, free_transfers: int) -> float:
    return HIT_COST * max(0, transfers - free_transfers)


def solve(problem: Problem, transfers: int, time_limit: float = 30.0) -> Solution | None:
    """Best squad with exactly `transfers` transfers, or None if there is none."""
    pool = problem.pool
    n = len(pool)
    s0, x0, c0 = 0, n, 2 * n  # column offsets of the three variable blocks
    xp = np.array([p.xpts for p in pool])
    w = problem.bench_weight

    cost = np.zeros(3 * n)
    cost[s0 : s0 + n] = -w * xp
    cost[x0 : x0 + n] = -(1 - w) * xp
    cost[c0 : c0 + n] = -xp

    rows = _Rows(3 * n)
    squad_cols = [s0 + i for i in range(n)]
    rows.add(squad_cols, [1.0] * n, SQUAD_SIZE, SQUAD_SIZE)
    rows.add([x0 + i for i in range(n)], [1.0] * n, XI_SIZE, XI_SIZE)
    rows.add([c0 + i for i in range(n)], [1.0] * n, 1, 1)

    for pos, count in SQUAD_BY_POSITION.items():
        idx = [i for i, p in enumerate(pool) if p.position == pos]
        rows.add([s0 + i for i in idx], [1.0] * len(idx), count, count)
        lo, hi = XI_BOUNDS[pos]
        rows.add([x0 + i for i in idx], [1.0] * len(idx), lo, hi)
    for team in {p.team for p in pool}:
        idx = [i for i, p in enumerate(pool) if p.team == team]
        rows.add([s0 + i for i in idx], [1.0] * len(idx), 0, MAX_PER_CLUB)

    prices = [float(problem.buy_price(p)) for p in pool]
    rows.add(squad_cols, prices, 0, float(problem.budget))

    for i in range(n):
        rows.add([x0 + i, s0 + i], [1.0, -1.0], -np.inf, 0)  # a starter is in the squad
        rows.add([c0 + i, x0 + i], [1.0, -1.0], -np.inf, 0)  # the captain starts

    kept = [s0 + i for i, p in enumerate(pool) if p.id in problem.selling_prices]
    rows.add(kept, [1.0] * len(kept), SQUAD_SIZE - transfers, SQUAD_SIZE - transfers)

    result = milp(
        c=cost,
        constraints=LinearConstraint(rows.matrix().tocsr(), rows.lb, rows.ub),
        integrality=np.ones(3 * n),
        bounds=Bounds(0, 1),
        options={"time_limit": time_limit},
    )
    if not result.success or result.x is None:
        return None

    chosen = np.round(result.x).astype(int)
    squad = [pool[i] for i in range(n) if chosen[s0 + i]]
    xi = [pool[i] for i in range(n) if chosen[x0 + i]]
    captain = next(pool[i] for i in range(n) if chosen[c0 + i])
    return _describe(problem, transfers, squad, xi, captain)


def _describe(
    problem: Problem, transfers: int, squad: list[Player], xi: list[Player], captain: Player
) -> Solution:
    xi_ids = {p.id for p in xi}
    xi_sorted = sorted(xi, key=lambda p: (p.position, -p.xpts, p.id))
    bench = [p for p in squad if p.id not in xi_ids]
    # The substitute goalkeeper is always first; outfield subs follow by expected points.
    bench_sorted = sorted(bench, key=lambda p: (p.position != 1, -p.xpts, p.id))
    vice = max((p for p in xi if p.id != captain.id), key=lambda p: (p.xpts, -p.id))
    new_ids = {p.id for p in squad}
    old_ids = set(problem.selling_prices)
    by_id = {p.id: p for p in problem.pool}
    bought = sorted(new_ids - old_ids)
    sold = sorted(old_ids - new_ids)
    spend = sum(by_id[i].price for i in bought)
    refund = sum(problem.selling_prices[i] for i in sold)
    xi_xpts = sum(p.xpts for p in xi) + captain.xpts
    hit = hit_cost(transfers, problem.free_transfers)
    return Solution(
        transfers=transfers,
        hit_cost=hit,
        xi=tuple(p.id for p in xi_sorted),
        bench=tuple(p.id for p in bench_sorted),
        captain=captain.id,
        vice_captain=vice.id,
        transfers_in=tuple(bought),
        transfers_out=tuple(sold),
        xi_xpts=xi_xpts,
        bench_xpts=sum(p.xpts for p in bench),
        net_xpts=xi_xpts - hit,
        bank_after=problem.bank + refund - spend,
    )


def solve_options(problem: Problem, max_transfers: int) -> list[Solution]:
    """The best squad for each of 0..max_transfers transfers (infeasible counts are omitted)."""
    options = (solve(problem, k) for k in range(max_transfers + 1))
    return [o for o in options if o is not None]


def recommend(
    options: list[Solution], min_gain_per_transfer: float = 0.5, tie_tolerance: float = 0.25
) -> Solution:
    """The option to act on.

    An option with transfers is eligible only if it earns at least `min_gain_per_transfer`
    per transfer over holding: predictions are noisy, so small projected gains are not worth
    the churn (or a hit). Among eligible options the best net expected points wins, but the
    fewest transfers within `tie_tolerance` points of it are preferred, since a marginal
    extra transfer is more likely noise than value.
    """
    hold = next((o for o in options if o.transfers == 0), None)
    if hold is None:
        raise ValueError("no feasible 0-transfer option: the current squad is not a legal squad")
    eligible = [hold] + [
        o
        for o in options
        if o.transfers > 0 and o.net_xpts - hold.net_xpts >= min_gain_per_transfer * o.transfers
    ]
    best = max(o.net_xpts for o in eligible)
    near_best = [o for o in eligible if o.net_xpts >= best - tie_tolerance]
    return min(near_best, key=lambda o: (o.transfers, -o.net_xpts))
