"""Education matching helpers."""

from chandra_eval.schemas.resume import DegreeLevel, EducationEntry


_ORDER = {
    DegreeLevel.UNKNOWN: 0,
    DegreeLevel.SECONDARY: 1,
    DegreeLevel.DIPLOMA: 2,
    DegreeLevel.CERTIFICATION: 2,
    DegreeLevel.BACHELOR: 3,
    DegreeLevel.PROFESSIONAL: 4,
    DegreeLevel.MASTER: 5,
    DegreeLevel.DOCTORATE: 6,
    DegreeLevel.POSTDOCTORATE: 7,
}


def has_degree_at_least(entries: list[EducationEntry], required: DegreeLevel) -> bool:
    required_rank = _ORDER[required]
    return any(_ORDER[entry.degree_level] >= required_rank for entry in entries)
