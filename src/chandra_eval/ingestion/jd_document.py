"""Job-description loading and simple synthetic extraction."""

from dataclasses import dataclass
import json
import re
from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
import pypdfium2 as pdfium

from chandra_eval.exceptions import DocumentIngestionError, ScannedPdfRequiresOcrError
from chandra_eval.schemas.common import EvidenceRef, EvidenceStatus
from chandra_eval.schemas.jd import (
    JDRequirement,
    RequirementCategory,
    RequirementType,
    ScoringCriterion,
    StructuredJD,
)


@dataclass(frozen=True)
class IngestedDocument:
    source_document_id: str
    text: str
    content_type: str
    page_count: int | None = None
    warnings: tuple[str, ...] = ()


def load_structured_jd(path: Path) -> StructuredJD:
    return StructuredJD.model_validate_json(path.read_text(encoding="utf-8"))


def write_structured_jd(jd: StructuredJD, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jd.model_dump(mode="json"), indent=2), encoding="utf-8")
    return path


def read_jd_document(path: Path | str) -> str:
    return ingest_jd_document(path).text


def ingest_jd_document(path: Path | str) -> IngestedDocument:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return IngestedDocument(
            source_document_id=path.name,
            text=read_docx_text(path),
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    if suffix in {"", ".txt", ".md"}:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise DocumentIngestionError(
                f"{path} is not valid UTF-8. Convert the text encoding or use DOCX/PDF input."
            ) from error
        return IngestedDocument(
            source_document_id=path.name,
            text=text,
            content_type="text/markdown" if suffix == ".md" else "text/plain",
        )
    if suffix == ".pdf":
        text, page_count = read_pdf_text(path)
        return IngestedDocument(
            source_document_id=path.name,
            text=text,
            content_type="application/pdf",
            page_count=page_count,
        )
    raise DocumentIngestionError(
        f"Unsupported JD input format: {suffix or '<none>'}. Supported: .docx, .txt, .md, .pdf."
    )


def read_docx_text(path: Path | str) -> str:
    document = Document(str(path))
    parts: list[str] = []

    for block in _iter_docx_blocks(document):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue
            style_name = block.style.name if block.style is not None else ""
            if style_name.lower().startswith("heading"):
                parts.append(f"# {text}")
            else:
                parts.append(text)
        elif isinstance(block, Table):
            for row in block.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))

    return "\n".join(parts)


def read_pdf_text(path: Path | str, min_text_characters: int = 20) -> tuple[str, int]:
    path = Path(path)
    document = pdfium.PdfDocument(str(path))
    pages: list[str] = []
    try:
        for index in range(len(document)):
            page = document[index]
            try:
                text_page = page.get_textpage()
                try:
                    text = text_page.get_text_range().strip()
                finally:
                    text_page.close()
            finally:
                page.close()
            pages.append(f"[Page {index + 1}]\n{text}".strip())
    finally:
        page_count = len(document)
        document.close()

    merged = "\n\n".join(page for page in pages if page.strip())
    plain_text = re.sub(r"\[Page \d+\]", "", merged).strip()
    if len(plain_text) < min_text_characters:
        raise ScannedPdfRequiresOcrError(
            f"{path} has little or no embedded text. Run Chandra OCR for scanned PDFs."
        )
    return merged, page_count


def _iter_docx_blocks(document) -> list[Paragraph | Table]:
    blocks: list[Paragraph | Table] = []
    for child in document.element.body.iterchildren():
        if child.tag.endswith("}p"):
            blocks.append(Paragraph(child, document))
        elif child.tag.endswith("}tbl"):
            blocks.append(Table(child, document))
    return blocks


def extract_synthetic_jd(text: str, job_id: str, source_document_id: str) -> StructuredJD:
    """Heuristic JD extraction for tests and demos; live pilots should use Gemma."""

    lowered = text.lower()
    skills = _extract_list_after(text, "required skills")
    preferred_skills = _extract_list_after(text, "preferred skills")
    months = _extract_experience_months(lowered)
    requirements: list[JDRequirement] = []
    for index, skill in enumerate(skills, start=1):
        requirements.append(
            JDRequirement(
                requirement_id=f"req-skill-{index}",
                category=RequirementCategory.SKILL,
                description=skill,
                requirement_type=RequirementType.MINIMUM,
                mandatory=True,
                evidence=[
                    EvidenceRef(
                        source_type="jd",
                        source_document_id=source_document_id,
                        exact_text=skill,
                        evidence_status=EvidenceStatus.SUPPORTED,
                    )
                ],
            )
        )
    if months is not None:
        requirements.append(
            JDRequirement(
                requirement_id="req-experience-months",
                category=RequirementCategory.EXPERIENCE,
                description=f"At least {months} months of relevant experience",
                requirement_type=RequirementType.MINIMUM,
                mandatory=True,
                minimum_value=months,
                unit="months",
            )
        )
    return StructuredJD(
        job_id=job_id,
        source_document_id=source_document_id,
        extraction_method="deterministic_fallback",
        job_title=_extract_title(text),
        role_summary=text.splitlines()[0].strip() if text.splitlines() else "",
        mandatory_requirements=requirements,
        required_skills=skills,
        preferred_skills=preferred_skills,
        minimum_experience_months=months,
        scoring_rubric=[
            ScoringCriterion(
                criterion_id="required-skills",
                name="Required skills",
                category=RequirementCategory.SKILL,
                description="Match required JD skills",
                weight=45,
                mandatory=True,
            ),
            ScoringCriterion(
                criterion_id="minimum-experience",
                name="Minimum experience",
                category=RequirementCategory.EXPERIENCE,
                description="Meet minimum experience duration",
                weight=35,
                mandatory=True,
            ),
            ScoringCriterion(
                criterion_id="preferred-skills",
                name="Preferred skills",
                category=RequirementCategory.SKILL,
                description="Match preferred JD skills",
                weight=20,
                mandatory=False,
            ),
        ],
        extraction_warnings=["Synthetic heuristic extraction; verify before live use."],
    )


def _extract_title(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            cleaned = line.strip().lstrip("#").strip()
            return cleaned.removeprefix("Title:").strip()
    return "Untitled role"


def _extract_list_after(text: str, label: str) -> list[str]:
    pattern = re.compile(rf"{re.escape(label)}\s*:\s*(.+)", re.IGNORECASE)
    for line in text.splitlines():
        match = pattern.search(line)
        if match:
            return [item.strip() for item in re.split(r",|;", match.group(1)) if item.strip()]
    return []


def _extract_experience_months(text: str) -> int | None:
    match = re.search(r"(\d+)\+?\s*(years|year|yrs|yr)", text)
    if match:
        return int(match.group(1)) * 12
    match = re.search(r"(\d+)\+?\s*(months|month|mos|mo)", text)
    if match:
        return int(match.group(1))
    return None
