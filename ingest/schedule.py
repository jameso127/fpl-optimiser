"""Is the next gameweek's deadline today?

The pipeline starts every morning at 10:00 UK. Ingest, run in `SCHEDULE_ONLY` mode, writes this
answer to `schedule.json`, and the workflow reads it to decide whether to carry on. Keeping the
rule here means it is plain Python that can be tested, not an expression in the workflow.
"""

import datetime as dt
from dataclasses import dataclass
from typing import Any
from zoneinfo import ZoneInfo

LONDON = ZoneInfo("Europe/London")


@dataclass(frozen=True)
class Schedule:
    gameweek: int
    deadline: dt.datetime
    deadline_day: bool  # is the deadline later today, UK time?

    def to_json(self) -> dict[str, Any]:
        return {
            "gameweek": self.gameweek,
            "deadline": self.deadline.isoformat(),
            "deadline_day": self.deadline_day,
        }


def plan(events: list[dict[str, Any]], now: dt.datetime) -> Schedule | None:
    """The plan for the next gameweek, or None when there is no upcoming deadline."""
    upcoming = next((e for e in events if e.get("is_next")), None)
    if upcoming is None:
        return None
    deadline = dt.datetime.fromisoformat(upcoming["deadline_time"]).astimezone(dt.UTC)
    return Schedule(
        gameweek=int(upcoming["id"]),
        deadline=deadline,
        deadline_day=deadline.astimezone(LONDON).date() == now.astimezone(LONDON).date(),
    )
