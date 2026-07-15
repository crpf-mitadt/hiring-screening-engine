"""Resume profile loading for offline and pilot workflows."""

import json
from pathlib import Path

from chandra_eval.safety.candidate_sanitizer import build_scoring_safe_profile
from chandra_eval.schemas.resume import CandidateProfile


def load_candidate_profile(path: Path | str) -> CandidateProfile:
    path = Path(path)
    candidate = CandidateProfile.model_validate_json(path.read_text(encoding="utf-8"))
    candidate.scoring_safe_profile = build_scoring_safe_profile(candidate)
    return candidate


def write_candidate_profile(candidate: CandidateProfile, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(candidate.model_dump(mode="json"), indent=2), encoding="utf-8")
    return path
