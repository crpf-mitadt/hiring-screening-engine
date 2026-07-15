"""Parse Chandra OCR HTML into ordered blocks without losing original text."""

from dataclasses import dataclass
import re

from bs4 import BeautifulSoup


_BBOX_RE = re.compile(r"[-+]?\d*\.?\d+")
_BULLET_RE = re.compile(r"^\s*[\u2022\-*]\s+")


@dataclass(frozen=True)
class ChandraBlock:
    index: int
    label: str
    text: str
    normalized_text: str
    bbox: list[float] | None
    html: str


def parse_bbox(value: str | None) -> list[float] | None:
    if not value:
        return None
    numbers = [float(match.group(0)) for match in _BBOX_RE.finditer(value)]
    return numbers or None


def normalize_duplicate_bullet(text: str, parent_name: str | None) -> str:
    """Remove visual bullets inside semantic list items only."""

    collapsed = " ".join(text.split())
    if parent_name in {"li", "ul", "ol"}:
        return _BULLET_RE.sub("", collapsed)
    return collapsed


def parse_chandra_html(html: str) -> list[ChandraBlock]:
    soup = BeautifulSoup(html, "html.parser")
    elements = soup.find_all(attrs={"data-bbox": True})
    if not elements:
        elements = soup.find_all(["h1", "h2", "h3", "p", "li", "td", "th"])

    blocks: list[ChandraBlock] = []
    for index, element in enumerate(elements):
        label = (
            element.get("data-label")
            or element.get("class", ["Text"])[0]
            if element.get("class")
            else "Text"
        )
        text = element.get_text(" ", strip=True)
        if not text:
            continue
        blocks.append(
            ChandraBlock(
                index=index,
                label=str(label),
                text=text,
                normalized_text=normalize_duplicate_bullet(text, element.name),
                bbox=parse_bbox(element.get("data-bbox")),
                html=str(element),
            )
        )
    return blocks
