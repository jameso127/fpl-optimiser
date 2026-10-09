"""Redraw the README pictures (docs/images/) from the latest local recommendation.

Uses the same code as the notify job, so the README shows exactly what the bot sends. Run the
pipeline locally first (ingest, predict, optimise), then:

    uv run python -m scripts.readme_images
"""

import datetime as dt
from pathlib import Path

from common.config import get_settings
from common.storage import gw_path, latest_gameweek, read_json, read_parquet, recommendation_path
from notify.pitch import draw_pitch
from notify.transfers import draw_transfers

OUT = Path(__file__).resolve().parent.parent / "docs" / "images"


def main() -> None:
    settings = get_settings()
    if settings.fpl_team_id is None:
        raise SystemExit("set FPL_TEAM_ID to the team whose recommendation should be drawn")
    season, gameweek = latest_gameweek(settings, "predictions", "run the pipeline first")
    rec = read_json(settings, recommendation_path(season, gameweek, settings.fpl_team_id))
    events = read_parquet(settings, gw_path(season, gameweek, "events"))
    deadline = dt.datetime.fromisoformat(
        str(events.loc[events["id"] == gameweek, "deadline_time"].iloc[0])
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pitch.png").write_bytes(draw_pitch(rec, deadline))
    transfers = draw_transfers(rec)
    if transfers is not None:
        (OUT / "transfers.png").write_bytes(transfers)
    print(f"wrote {OUT} for gameweek {gameweek}")


if __name__ == "__main__":
    main()
