"""When to send the recommendations for the next gameweek.

On a deadline day (in UK time) the message goes out at 10:00, or 2.5 hours before the deadline
if that is earlier. The pipeline is started a few minutes ahead so its data is fresh. Ingest
writes the plan to `schedule.json`, and the workflow reads it, so this logic is plain Python
that can be tested rather than expressions in the workflow definition.
"""

import datetime as dt
from dataclasses import dataclass
from typing import Any
from zoneinfo import ZoneInfo

LONDON = ZoneInfo("Europe/London")
SEND_HOUR = 10
BEFORE_DEADLINE = dt.timedelta(hours=2, minutes=30)
LEAD_TIME = dt.timedelta(minutes=10)  # start the pipeline this long before the send time


@dataclass(frozen=True)
class Schedule:
    gameweek: int
    deadline: dt.datetime
    send_at: dt.datetime
    run_at: dt.datetime
    deadline_day: bool  # is the deadline today, UK time?

    def to_json(self) -> dict[str, Any]:
        return {
            "gameweek": self.gameweek,
            "deadline": self.deadline.isoformat(),
            "send_at": self.send_at.isoformat(),
            "run_at": self.run_at.isoformat(),
            "deadline_day": self.deadline_day,
        }


def plan(events: list[dict[str, Any]], now: dt.datetime) -> Schedule | None:
    """The plan for the next gameweek, or None when there is no upcoming deadline."""
    upcoming = next((e for e in events if e.get("is_next")), None)
    if upcoming is None:
        return None
    deadline = dt.datetime.fromisoformat(upcoming["deadline_time"]).astimezone(dt.UTC)
    morning = dt.datetime.combine(
        deadline.astimezone(LONDON).date(), dt.time(SEND_HOUR), tzinfo=LONDON
    )
    send_at = min(morning, deadline - BEFORE_DEADLINE).astimezone(dt.UTC)
    return Schedule(
        gameweek=int(upcoming["id"]),
        deadline=deadline,
        send_at=send_at,
        run_at=send_at - LEAD_TIME,
        deadline_day=deadline.astimezone(LONDON).date() == now.astimezone(LONDON).date(),
    )
