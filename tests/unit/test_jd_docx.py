from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from docx import Document

from chandra_eval.pipeline.jd_structuring import structure_jd_document_file


class JDDocxTests(unittest.TestCase):
    def test_runtime_generated_docx_is_parsed_directly(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            docx_path = root / "job-description.docx"
            output_path = root / "structured-jd.json"

            document = Document()
            document.add_paragraph("Title: Document AI Engineer")
            document.add_paragraph("Required skills: Python, OCR, data validation")
            document.add_paragraph("Preferred skills: Pydantic, reporting")
            document.add_paragraph("Minimum experience: 2 years")
            document.save(docx_path)

            jd = structure_jd_document_file(docx_path, "job-docx-001", output_path)

            self.assertTrue(output_path.is_file())
            self.assertEqual(jd.job_title, "Document AI Engineer")
            self.assertEqual(jd.source_document_id, "job-description.docx")
            self.assertEqual(jd.required_skills, ["Python", "OCR", "data validation"])
            self.assertEqual(jd.minimum_experience_months, 24)


if __name__ == "__main__":
    unittest.main()
