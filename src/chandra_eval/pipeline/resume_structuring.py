"""Resume structuring placeholder for synthetic and Gemma-backed workflows."""

from pathlib import Path

from chandra_eval.ingestion.resume_document import load_candidate_profile
from chandra_eval.schemas.resume import CandidateProfile


def load_structured_resume(path: Path) -> CandidateProfile:
    return load_candidate_profile(path)
