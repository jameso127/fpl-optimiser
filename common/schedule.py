"""Is the next gameweek's deadline today?

The pipeline runs every morning at 10:00 UK. Every job runs each day, but notify only sends on
a deadline day: it asks this module, so the rule is plain Python that can be tested.
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
