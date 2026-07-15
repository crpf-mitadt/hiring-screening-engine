"""Experience calculations."""

from datetime import date

from chandra_eval.schemas.resume import ExperienceEntry


def _year_month(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    parts = value.split("-")
    try:
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 1
    except ValueError:
        return None
    return year, month


def months_between(start: str | None, end: str | None, active: bool = False) -> int:
    start_ym = _year_month(start)
    if start_ym is None:
        return 0
    if active:
        today = date.today()
        end_ym = (today.year, today.month)
    else:
        end_ym = _year_month(end)
    if end_ym is None:
        return 0
    return max(0, (end_ym[0] - start_ym[0]) * 12 + end_ym[1] - start_ym[1])


def total_experience_months(entries: list[ExperienceEntry]) -> int:
    return sum(
        months_between(entry.start_date, entry.end_date, entry.currently_active)
        for entry in entries
    )
