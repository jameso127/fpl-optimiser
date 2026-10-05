import datetime as dt

from ingest.schedule import plan


def _events(deadline: str, gameweek: int = 6) -> list[dict[str, object]]:
    return [
        {"id": gameweek - 1, "deadline_time": "2026-09-26T10:00:00Z", "is_next": False},
        {"id": gameweek, "deadline_time": deadline, "is_next": True},
    ]


def _at(text: str) -> dt.datetime:
    return dt.datetime.fromisoformat(text)


def test_an_evening_deadline_is_sent_at_ten_in_the_morning_uk_time() -> None:
    # 18:30 BST on a Saturday in October: 10:00 BST is 09:00 UTC.
    result = plan(_events("2026-10-10T17:30:00Z"), _at("2026-10-10T06:00:00+00:00"))

    assert result is not None and result.gameweek == 6 and result.deadline_day
    assert result.send_at == _at("2026-10-10T09:00:00+00:00")
    assert result.run_at == _at("2026-10-10T08:50:00+00:00")


def test_the_morning_time_follows_uk_clock_changes() -> None:
    # In winter (GMT) 10:00 is 10:00 UTC.
    result = plan(_events("2026-12-05T17:30:00Z"), _at("2026-12-05T07:00:00+00:00"))

    assert result is not None and result.send_at == _at("2026-12-05T10:00:00+00:00")


def test_an_early_deadline_is_sent_two_and_a_half_hours_before() -> None:
    # A 11:00 UTC deadline: 10:00 would be too close, so 08:30 instead.
    result = plan(_events("2026-12-05T11:00:00Z"), _at("2026-12-05T07:00:00+00:00"))

    assert result is not None and result.send_at == _at("2026-12-05T08:30:00+00:00")


def _is_deadline_day(events: list[dict[str, object]], now: str) -> bool:
    result = plan(events, _at(now))
    assert result is not None
    return result.deadline_day


def test_it_is_only_a_deadline_day_on_the_deadlines_uk_date() -> None:
    events = _events("2026-10-10T17:30:00Z")

    assert not _is_deadline_day(events, "2026-10-09T06:00:00+00:00")
    assert _is_deadline_day(events, "2026-10-10T06:00:00+00:00")


def test_the_uk_date_is_used_not_the_utc_date() -> None:
    # 23:30 UTC on 9 October is already 00:30 on 10 October in the UK (BST).
    events = _events("2026-10-10T17:30:00Z")

    assert _is_deadline_day(events, "2026-10-09T23:30:00+00:00")


def test_no_upcoming_gameweek_means_no_plan() -> None:
    assert (
        plan(
            [{"id": 38, "deadline_time": "2027-05-23T13:30:00Z", "is_next": False}],
            _at("2027-06-01T00:00:00+00:00"),
        )
        is None
    )
