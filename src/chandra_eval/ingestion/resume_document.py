"""Resume profile loading for offline and pilot workflows."""

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any

from chandra_eval.ingestion.chandra_html import parse_chandra_html
from chandra_eval.ocr.full_document import extract_response_content
from chandra_eval.safety.candidate_sanitizer import build_scoring_safe_profile
from chandra_eval.schemas.common import EvidenceRef, EvidenceStatus
from chandra_eval.schemas.resume import CandidateProfile


_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b")
_PHONE_RE = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")


@dataclass(frozen=True)
class OcrBlock:
    page_number: int
    block_index: int
    data_label: str
    data_bbox: list[float] | None
    exact_text: str
    normalized_text: str


@dataclass(frozen=True)
class OcrDocument:
    source_document_id: str
    blocks: list[OcrBlock]
    warnings: list[str]


def load_candidate_profile(path: Path | str) -> CandidateProfile:
    path = Path(path)
    candidate = CandidateProfile.model_validate_json(path.read_text(encoding="utf-8"))
    candidate.scoring_safe_profile = build_scoring_safe_profile(candidate)
    return candidate


def write_candidate_profile(candidate: CandidateProfile, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(candidate.model_dump(mode="json"), indent=2), encoding="utf-8")
    return path


def load_ocr_document(path: Path | str) -> OcrDocument:
    path = Path(path)
    if path.is_dir():
        return _load_ocr_directory(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return _load_raw_responses(payload, source_document_id=path.name)
        if isinstance(payload, dict) and "pages" in payload:
            return _load_document_json(payload, source_document_id=path.name)
    if path.suffix.lower() in {".html", ".htm"}:
        return _load_html_files([path])
    raise ValueError(
        f"Unsupported OCR input: {path}. Use an OCR output directory, raw responses JSON, document.json, or page HTML."
    )


def prompt_text_from_ocr(document: OcrDocument) -> str:
    lines: list[str] = []
    for block in document.blocks:
        if block.data_label.lower() == "image":
            continue
        text = block.normalized_text.strip()
        if not text or _EMAIL_RE.search(text) or _PHONE_RE.search(text):
            continue
        lines.append(f"[page={block.page_number} block={block.block_index} label={block.data_label}] {text}")
    return "\n".join(lines)


def evidence_from_block(block: OcrBlock, source_document_id: str = "chandra_ocr") -> EvidenceRef:
    return EvidenceRef(
        source_type="resume_ocr",
        source_document_id=source_document_id,
        page_number=block.page_number,
        block_index=block.block_index,
        bbox=block.data_bbox,
        exact_text=block.exact_text,
        normalized_text=block.normalized_text,
        evidence_status=EvidenceStatus.SUPPORTED,
        notes=f"data-label={block.data_label}",
    )


def _load_ocr_directory(path: Path) -> OcrDocument:
    source_map_path = path / "source-map.json"
    document_path = path / "document.json"
    raw_responses_path = path / "raw-responses.json"
    if source_map_path.is_file():
        payload = json.loads(source_map_path.read_text(encoding="utf-8"))
        return _load_source_map(payload, source_document_id=path.name)
    if document_path.is_file():
        payload = json.loads(document_path.read_text(encoding="utf-8"))
        return _load_document_json(payload, source_document_id=path.name)
    if raw_responses_path.is_file():
        payload = json.loads(raw_responses_path.read_text(encoding="utf-8"))
        return _load_raw_responses(payload, source_document_id=path.name)
    html_files = sorted(path.glob("page-*.html"))
    if html_files:
        return _load_html_files(html_files, source_document_id=path.name)
    raise ValueError(f"No OCR artifacts found in {path}")


def _load_source_map(payload: Any, source_document_id: str) -> OcrDocument:
    if not isinstance(payload, list):
        raise ValueError("source-map.json must contain a list")
    blocks = [
        OcrBlock(
            page_number=int(item["page_number"]),
            block_index=int(item["block_index"]),
            data_label=str(item.get("data_label", "Text")),
            data_bbox=item.get("data_bbox"),
            exact_text=str(item.get("exact_text", "")),
            normalized_text=str(item.get("normalized_text") or item.get("exact_text", "")),
        )
        for item in payload
        if isinstance(item, dict)
    ]
    if not blocks:
        raise ValueError("OCR source map contained no text blocks")
    return OcrDocument(source_document_id=source_document_id, blocks=blocks, warnings=[])


def _load_document_json(payload: dict[str, Any], source_document_id: str) -> OcrDocument:
    source_map: list[dict[str, Any]] = []
    for page in payload.get("pages", []):
        for block in page.get("blocks", []):
            source_map.append(block)
    return _load_source_map(source_map, source_document_id)


def _load_raw_responses(payload: list[Any], source_document_id: str) -> OcrDocument:
    blocks: list[OcrBlock] = []
    warnings: list[str] = []
    for page_number, response in enumerate(payload, start=1):
        if not isinstance(response, dict):
            warnings.append(f"page {page_number}: invalid response item")
            continue
        html, page_warnings = extract_response_content(response)
        warnings.extend(f"page {page_number}: {warning}" for warning in page_warnings)
        blocks.extend(_blocks_from_html(html, page_number))
    if not blocks:
        raise ValueError("OCR responses contained no parseable HTML text")
    return OcrDocument(source_document_id=source_document_id, blocks=blocks, warnings=warnings)


def _load_html_files(paths: list[Path], source_document_id: str | None = None) -> OcrDocument:
    blocks: list[OcrBlock] = []
    for page_number, path in enumerate(paths, start=1):
        blocks.extend(_blocks_from_html(path.read_text(encoding="utf-8"), page_number))
    if not blocks:
        raise ValueError("OCR HTML contained no parseable text blocks")
    return OcrDocument(
        source_document_id=source_document_id or paths[0].name,
        blocks=blocks,
        warnings=[],
    )


def _blocks_from_html(html: str, page_number: int) -> list[OcrBlock]:
    return [
        OcrBlock(
            page_number=page_number,
            block_index=block.index,
            data_label=block.label,
            data_bbox=block.bbox,
            exact_text=block.text,
            normalized_text=block.normalized_text,
        )
        for block in parse_chandra_html(html)
    ]
