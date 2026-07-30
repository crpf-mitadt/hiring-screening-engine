"""Experience calculations with overlap and ambiguity tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import re

from chandra_eval.schemas.resume import ExperienceEntry


_YEAR_MONTH_RE = re.compile(r"^(\d{4})(?:-(\d{1,2}))?$")


@dataclass(frozen=True)
class ExperienceRange:
    index: int
    title: str | None
    organization: str | None
    start_date: str | None
    end_date: str | None
    status: str
    confirmed_months: int = 0
    possible_months: int = 0
    warning: str | None = None

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "title": self.title,
            "organization": self.organization,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "status": self.status,
            "confirmed_months": self.confirmed_months,
            "possible_months": self.possible_months,
            "warning": self.warning,
        }


@dataclass(frozen=True)
class ExperienceDurationResult:
    confirmed_months: int
    possible_additional_months: int = 0
    uncertain_ranges: list[ExperienceRange] = field(default_factory=list)
    invalid_ranges: list[ExperienceRange] = field(default_factory=list)
    current_ranges: list[ExperienceRange] = field(default_factory=list)
    date_warnings: list[str] = field(default_factory=list)
    calculation_status: str = "met"

    def as_dict(self) -> dict:
        return {
            "confirmed_months": self.confirmed_months,
            "possible_additional_months": self.possible_additional_months,
            "uncertain_ranges": [item.as_dict() for item in self.uncertain_ranges],
            "invalid_ranges": [item.as_dict() for item in self.invalid_ranges],
            "current_ranges": [item.as_dict() for item in self.current_ranges],
            "date_warnings": self.date_warnings,
            "calculation_status": self.calculation_status,
        }


def parse_year_month(value: str | None) -> tuple[int, int] | None:
    """Parse only explicit YYYY or YYYY-MM values; malformed partial years stay unknown."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = _YEAR_MONTH_RE.fullmatch(text)
    if not match:
        return None
    year = int(match.group(1))
    month = int(match.group(2) or "1")
    if not 1 <= month <= 12:
        return None
    return year, month


def months_between(start: str | None, end: str | None, active: bool = False) -> int:
    """Backward-compatible confirmed month count for a single range."""

    start_ym = parse_year_month(start)
    end_ym = _end_year_month(end, active)
    if start_ym is None or end_ym is None:
        return 0
    return max(0, _month_index(end_ym) - _month_index(start_ym))


def total_experience_months(entries: list[ExperienceEntry]) -> int:
    return calculate_experience_duration(entries).confirmed_months


def calculate_experience_duration(
    entries: list[ExperienceEntry],
    *,
    evaluation_date: date | None = None,
) -> ExperienceDurationResult:
    evaluation_date = evaluation_date or date.today()
    intervals: list[tuple[int, int]] = []
    uncertain: list[ExperienceRange] = []
    invalid: list[ExperienceRange] = []
    current: list[ExperienceRange] = []
    warnings: list[str] = []

    for index, entry in enumerate(entries):
        start = parse_year_month(entry.start_date)
        end = _end_year_month(entry.end_date, entry.currently_active, evaluation_date)
        if entry.currently_active and start is not None and end is not None:
            current.append(
                _range_from_entry(
                    index,
                    entry,
                    "current",
                    confirmed_months=max(0, _month_index(end) - _month_index(start)),
                )
            )
        if start is None:
            warning = f"experience[{index}] has missing or ambiguous start_date"
            warnings.append(warning)
            uncertain.append(_range_from_entry(index, entry, "missing_start", warning=warning))
            continue
        if end is None:
            warning = f"experience[{index}] has missing or ambiguous end_date"
            warnings.append(warning)
            uncertain.append(_range_from_entry(index, entry, "ambiguous_end", warning=warning))
            continue
        start_month = _month_index(start)
        end_month = _month_index(end)
        if end_month < start_month:
            warning = f"experience[{index}] end_date precedes start_date"
            warnings.append(warning)
            invalid.append(_range_from_entry(index, entry, "invalid_range", warning=warning))
            continue
        intervals.append((start_month, end_month))

    confirmed = _union_months(intervals)
    possible = sum(item.possible_months for item in uncertain)
    status = "confirmed"
    if invalid:
        status = "invalid_ranges_present"
    if uncertain:
        status = "uncertain_ranges_present"
    return ExperienceDurationResult(
        confirmed_months=confirmed,
        possible_additional_months=possible,
        uncertain_ranges=uncertain,
        invalid_ranges=invalid,
        current_ranges=current,
        date_warnings=warnings,
        calculation_status=status,
    )


def _end_year_month(value: str | None, active: bool, evaluation_date: date | None = None) -> tuple[int, int] | None:
    if active and not value:
        current = evaluation_date or date.today()
        return current.year, current.month
    return parse_year_month(value)


def _month_index(year_month: tuple[int, int]) -> int:
    year, month = year_month
    return year * 12 + month


def _union_months(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return sum(end - start for start, end in merged)


def _range_from_entry(
    index: int,
    entry: ExperienceEntry,
    status: str,
    *,
    confirmed_months: int = 0,
    possible_months: int = 0,
    warning: str | None = None,
) -> ExperienceRange:
    return ExperienceRange(
        index=index,
        title=entry.title,
        organization=entry.organization,
        start_date=entry.start_date,
        end_date=entry.end_date,
        status=status,
        confirmed_months=confirmed_months,
        possible_months=possible_months,
        warning=warning,
    )
