from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from docx import Document

from chandra_eval.exceptions import DocumentIngestionError, ScannedPdfRequiresOcrError
from chandra_eval.ingestion.jd_document import ingest_jd_document


def _write_pdf(path: Path, text: str) -> None:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    objects = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n",
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
        b"5 0 obj << /Length " + str(len(stream)).encode("ascii") + b" >> stream\n"
        + stream
        + b"\nendstream endobj\n",
    ]
    content = b"%PDF-1.4\n"
    offsets = []
    for item in objects:
        offsets.append(len(content))
        content += item
    xref = len(content)
    content += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii")
    for offset in offsets:
        content += f"{offset:010d} 00000 n \n".encode("ascii")
    content += (
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode("ascii")
    path.write_bytes(content)


class JDIngestionTests(unittest.TestCase):
    def test_docx_preserves_headings_paragraphs_and_tables(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "jd.docx"
            document = Document()
            document.add_heading("Marketing Coordinator", level=1)
            document.add_paragraph("Required skills: SEO, Analytics")
            table = document.add_table(rows=1, cols=2)
            table.rows[0].cells[0].text = "Department"
            table.rows[0].cells[1].text = "Marketing"
            document.save(path)

            ingested = ingest_jd_document(path)

            self.assertIn("# Marketing Coordinator", ingested.text)
            self.assertIn("Required skills: SEO, Analytics", ingested.text)
            self.assertIn("Department | Marketing", ingested.text)

    def test_txt_markdown_pdf_and_failures(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            txt = root / "jd.txt"
            md = root / "jd.md"
            pdf = root / "jd.pdf"
            empty_pdf = root / "empty.pdf"
            unsupported = root / "jd.rtf"
            bad_text = root / "bad.txt"

            txt.write_text("Title: TXT Role", encoding="utf-8")
            md.write_text("# Markdown Role", encoding="utf-8")
            _write_pdf(pdf, "Title: PDF Role with enough text")
            _write_pdf(empty_pdf, "")
            unsupported.write_text("{}", encoding="utf-8")
            bad_text.write_bytes(b"\xff\xfe\x00")

            self.assertIn("TXT Role", ingest_jd_document(txt).text)
            self.assertIn("Markdown Role", ingest_jd_document(md).text)
            self.assertIn("[Page 1]", ingest_jd_document(pdf).text)
            with self.assertRaises(ScannedPdfRequiresOcrError):
                ingest_jd_document(empty_pdf)
            with self.assertRaises(DocumentIngestionError):
                ingest_jd_document(unsupported)
            with self.assertRaises(DocumentIngestionError):
                ingest_jd_document(bad_text)


if __name__ == "__main__":
    unittest.main()
