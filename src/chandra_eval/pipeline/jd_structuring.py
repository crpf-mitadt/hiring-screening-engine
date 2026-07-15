"""JD structuring pipeline."""

from pathlib import Path

from chandra_eval.ingestion.jd_document import extract_synthetic_jd, write_structured_jd
from chandra_eval.schemas.jd import StructuredJD


def structure_jd_text_file(path: Path, job_id: str, output_path: Path) -> StructuredJD:
    jd = extract_synthetic_jd(
        path.read_text(encoding="utf-8"),
        job_id=job_id,
        source_document_id=path.name,
    )
    write_structured_jd(jd, output_path)
    return jd
