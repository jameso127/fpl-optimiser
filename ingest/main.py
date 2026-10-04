"""Ingest job: FPL API -> Parquet in gs://$DATA_BUCKET/season=<yyyy-yy>/gw=<n>/.

For target gameweek n it writes a snapshot (players, teams, events, fixtures) to gw=n/, and
live per-player stats for every finished gameweek, backfilling any that are missing and
always refreshing the most recent finished one. Re-running overwrites; it never appends.

The players snapshot matters: it holds FPL's `ep_next` as of the deadline, which cannot be
recovered later and is the baseline the predictions are backtested against.
"""

import logging

from common.config import Settings, get_settings
from common.fpl_client import FplClient, FplSource
from common.logging import configure_logging
from common.storage import exists, gw_path, write_parquet
from ingest import transform
from ingest.dry_run import DryRunSource

log = logging.getLogger(__name__)


def run(settings: Settings, source: FplSource) -> int:
    bootstrap = source.bootstrap()
    season = settings.season or transform.season_label(bootstrap)
    gameweek = settings.gameweek or transform.target_gameweek(bootstrap)
    log.info(
        "ingest start",
        extra={"season": season, "gameweek": gameweek, "dry_run": settings.dry_run},
    )

    snapshot = {
        "players": transform.players_frame(bootstrap),
        "teams": transform.teams_frame(bootstrap),
        "events": transform.events_frame(bootstrap),
        "fixtures": transform.fixtures_frame(source.fixtures()),
    }
    for name, df in snapshot.items():
        uri = write_parquet(settings, df, gw_path(season, gameweek, name))
        log.info("wrote snapshot", extra={"uri": uri, "rows": len(df)})

    finished = [gw for gw in transform.finished_gameweeks(bootstrap) if gw < gameweek]
    for gw in finished:
        rel = gw_path(season, gw, "live")
        if gw != max(finished) and exists(settings, rel):
            continue
        df = transform.live_frame(source.live(gw), gw)
        uri = write_parquet(settings, df, rel)
        log.info("wrote live stats", extra={"uri": uri, "rows": len(df), "live_gameweek": gw})

    log.info("ingest done", extra={"season": season, "gameweek": gameweek})
    return gameweek


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    source: FplSource = DryRunSource() if settings.dry_run else FplClient(settings)
    run(settings, source)


if __name__ == "__main__":
    main()
