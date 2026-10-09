"""Optimise job: predictions + a manager's squad -> the best team for 0..N transfers.

Reads the latest gameweek's `predictions` and `players` snapshot, rebuilds the manager's squad
from the FPL API (cached, polite), and for each number of transfers finds the best squad,
starting XI, captain and bench. Writes `season=<s>/gw=<n>/recommendations/<team_id>.json`,
which the notify job turns into a message. Re-running overwrites.

    FPL_TEAM_ID=123456 uv run python -m optimise.main
"""

import datetime as dt
import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from common.config import Settings, get_settings
from common.fpl_client import FplClient
from common.logging import configure_logging
from common.storage import (
    exists,
    gw_path,
    latest_gameweek,
    read_parquet,
    recommendation_path,
    write_json,
)
from common.users import get_user_repository
from common.users.models import DeclaredTransfer
from optimise import model, squad
from optimise.model import POSITION_NAMES, Player, Problem, Solution
from optimise.squad import EntrySource, Squad

log = logging.getLogger(__name__)


def build_pool(players: pd.DataFrame, xpts: pd.Series, owned: set[int]) -> list[Player]:
    """Everyone with a positive expected score, plus everyone already in the squad."""
    pool = []
    for row in players.to_dict("records"):
        pid = int(row["id"])
        points = float(xpts.get(pid, 0.0))
        if points > 0 or pid in owned:
            pool.append(
                Player(
                    id=pid,
                    name=str(row["web_name"]),
                    position=int(row["element_type"]),
                    team=int(row["team"]),
                    price=int(row["now_cost"]),
                    xpts=points,
                )
            )
    return pool


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def player_stats(
    players: pd.DataFrame, fixtures: pd.DataFrame | None, teams: Mapping[int, str], gameweek: int
) -> dict[int, dict[str, Any]]:
    """Context for each player card: this gameweek's fixtures and FPL's own numbers (form,
    ownership, injury news). Every field is optional: a missing column, or no fixtures file,
    just leaves it out, and a blank gameweek is an empty fixture list."""
    games: dict[int, list[dict[str, Any]]] = {}
    if fixtures is not None:
        for f in fixtures[fixtures["event"] == gameweek].to_dict("records"):
            home, away = int(f["team_h"]), int(f["team_a"])
            for team, opponent, at_home, fdr in (
                (home, away, True, f["team_h_difficulty"]),
                (away, home, False, f["team_a_difficulty"]),
            ):
                games.setdefault(team, []).append(
                    {
                        "opponent": teams.get(opponent, str(opponent)),
                        "home": at_home,
                        "difficulty": int(fdr),
                    }
                )
    stats: dict[int, dict[str, Any]] = {}
    for row in players.to_dict("records"):
        extra: dict[str, Any] = {}
        if fixtures is not None:
            extra["fixtures"] = games.get(int(row["team"]), [])
        for key, column in (
            ("form", "form"),
            ("ownership", "selected_by_percent"),
            ("total_points", "total_points"),
            ("chance_of_playing", "chance_of_playing_next_round"),
        ):
            value = _number(row.get(column))
            if value is not None:
                extra[key] = value
        news = row.get("news")
        if isinstance(news, str) and news.strip():
            extra["news"] = news.strip()
        stats[int(row["id"])] = extra
    return stats


def _player_json(
    p: Player,
    teams: dict[int, str],
    captain: int,
    vice: int,
    stats: Mapping[int, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": p.id,
        "name": p.name,
        "position": POSITION_NAMES[p.position],
        "team": teams.get(p.team, str(p.team)),
        "xpts": round(p.xpts, 2),
        "price": p.price / 10,
        "captain": p.id == captain,
        "vice_captain": p.id == vice,
        **stats.get(p.id, {}),
    }


def _option_json(
    option: Solution,
    by_id: dict[int, Player],
    teams: dict[int, str],
    problem: Problem,
    hold: float,
    stats: Mapping[int, dict[str, Any]],
) -> dict[str, Any]:
    def card(i: int, captain: int = -1, vice: int = -1) -> dict[str, Any]:
        return _player_json(by_id[i], teams, captain, vice, stats)

    def cards(ids: tuple[int, ...]) -> list[dict[str, Any]]:
        return [card(i, option.captain, option.vice_captain) for i in ids]

    moves = []
    for out_id, in_id in zip(option.transfers_out, option.transfers_in, strict=False):
        moves.append(
            {
                "out": {**card(out_id), "sell_price": problem.selling_prices[out_id] / 10},
                "in": card(in_id),
            }
        )
    return {
        "transfers": option.transfers,
        "hit_cost": option.hit_cost,
        "xpts": round(option.xi_xpts, 2),
        "net_xpts": round(option.net_xpts, 2),
        "gain_vs_hold": round(option.net_xpts - hold, 2),
        "bank_after": option.bank_after / 10,
        "captain": by_id[option.captain].name,
        "vice_captain": by_id[option.vice_captain].name,
        "starting_xi": cards(option.xi),
        "bench": cards(option.bench),
        "moves": moves,
    }


def assumptions(my_squad: Squad, names: Mapping[int, str]) -> list[str]:
    """What the advice is based on, in plain words, so a reader can spot a wrong assumption.

    FPL hides a manager's changes for an upcoming gameweek until its deadline, so these lines
    matter: they tell the user what we know and what we were told.
    """

    def label(i: int) -> str:
        return names.get(i, str(i))

    lines = [f"Your team as FPL shows it after the gameweek {my_squad.base_gameweek} deadline."]
    if my_squad.declared_applied:
        moves = ", ".join(f"{label(o)} to {label(i)}" for o, i in my_squad.declared_applied)
        lines.append(f"Plus the transfers you told me about: {moves}.")
    visible = my_squad.planned_transfers - len(my_squad.declared_applied)
    if visible > 0:
        lines.append(f"{visible} transfer(s) for this gameweek are already visible in FPL.")
    source = "estimated" if my_squad.free_transfers_estimated else "your setting"
    lines.append(f"Free transfers left: {my_squad.free_transfers} ({source}).")
    lines.extend(my_squad.notes)
    return lines


def build_recommendation(
    *,
    season: str,
    gameweek: int,
    team_id: int,
    model_version: str,
    my_squad: Squad,
    problem: Problem,
    options: list[Solution],
    chosen: Solution,
    teams: dict[int, str],
    names: Mapping[int, str],
    settings: Settings,
    stats: Mapping[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    by_id = {p.id: p for p in problem.pool}
    stats = stats or {}
    hold = next(o for o in options if o.transfers == 0).net_xpts
    return {
        "season": season,
        "gameweek": gameweek,
        "team_id": team_id,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "model_version": model_version,
        "free_transfers": problem.free_transfers,
        "planned_transfers": my_squad.planned_transfers,
        "bank": problem.bank / 10,
        "squad_value": sum(problem.selling_prices.values()) / 10,
        "max_transfers_considered": settings.max_transfers,
        "min_gain_per_transfer": settings.min_gain_per_transfer,
        "recommended_transfers": chosen.transfers,
        "assumptions": assumptions(my_squad, names),
        "options": [_option_json(o, by_id, teams, problem, hold, stats) for o in options],
    }


def run(
    settings: Settings,
    source: EntrySource,
    team_id: int | None = None,
    declared: Sequence[DeclaredTransfer] = (),
) -> dict[str, Any]:
    team_id = team_id or settings.fpl_team_id
    if team_id is None:
        raise RuntimeError("set FPL_TEAM_ID to the manager whose team should be optimised")
    season, gameweek = latest_gameweek(settings, "predictions", "run ingest and predict first")
    log.info("optimise start", extra={"season": season, "gameweek": gameweek, "team": team_id})

    players = read_parquet(settings, gw_path(season, gameweek, "players"))
    teams_df = read_parquet(settings, gw_path(season, gameweek, "teams"))
    events = read_parquet(settings, gw_path(season, gameweek, "events"))
    preds = read_parquet(settings, gw_path(season, gameweek, "predictions"))
    finished = events.loc[events["finished"] & (events["id"] < gameweek), "id"]
    if finished.empty:
        raise RuntimeError("no finished gameweek yet: there is no squad to optimise")

    market = {int(i): int(c) for i, c in zip(players["id"], players["now_cost"], strict=True)}
    names = {int(i): str(n) for i, n in zip(players["id"], players["web_name"], strict=True)}
    my_squad = squad.build_squad(
        source,
        team_id,
        market,
        last_finished=int(finished.max()),
        next_gameweek=gameweek,
        max_free_transfers=settings.max_free_transfers,
        free_transfers_override=settings.free_transfers_override,
        declared=declared,
        names=names,
    )
    xpts = preds.set_index("id")["xpts"]
    pool = build_pool(players, xpts, set(my_squad.player_ids))
    problem = Problem(
        pool=tuple(pool),
        selling_prices=my_squad.selling_prices,
        bank=my_squad.bank,
        free_transfers=my_squad.free_transfers,
        bench_weight=settings.bench_weight,
    )
    options = model.solve_options(problem, settings.max_transfers)
    chosen = model.recommend(options, settings.min_gain_per_transfer)
    versions: list[str] = (
        [str(v) for v in preds["model_version"].dropna().unique()]
        if "model_version" in preds
        else []
    )
    teams = {int(i): str(n) for i, n in zip(teams_df["id"], teams_df["short_name"], strict=True)}
    fixtures_rel = gw_path(season, gameweek, "fixtures")
    fixtures = read_parquet(settings, fixtures_rel) if exists(settings, fixtures_rel) else None
    rec = build_recommendation(
        season=season,
        gameweek=gameweek,
        team_id=team_id,
        model_version=versions[0] if versions else "unknown",
        my_squad=my_squad,
        problem=problem,
        options=options,
        chosen=chosen,
        teams=teams,
        names=names,
        settings=settings,
        stats=player_stats(players, fixtures, teams, gameweek),
    )
    uri = write_json(settings, rec, recommendation_path(season, gameweek, team_id))
    log.info(
        "optimise done",
        extra={"uri": uri, "recommended_transfers": chosen.transfers, "options": len(options)},
    )
    return rec


def main() -> None:
    """Optimise for every active user. One user's failure does not stop the others, but it does
    make the job fail so that it shows up."""
    settings = get_settings()
    configure_logging(settings.log_level)
    repo = get_user_repository(settings)
    client = FplClient(settings)
    season, gameweek = latest_gameweek(settings, "predictions", "run ingest and predict first")

    failed = 0
    for user in repo.list_active():
        try:
            personal = settings.model_copy(
                update={
                    "max_transfers": user.max_transfers or settings.max_transfers,
                    "free_transfers_override": user.free_transfers_override,
                }
            )
            declared = repo.declared(user.chat_id, season, gameweek)
            run(personal, client, user.fpl_team_id, declared)
        except Exception:
            failed += 1
            log.exception("optimise failed", extra={"team": user.fpl_team_id})
    if failed:
        raise SystemExit(f"{failed} user(s) failed")


if __name__ == "__main__":
    main()
