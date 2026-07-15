from pathlib import Path
import unittest

from chandra_eval.ingestion.chandra_html import parse_chandra_html


class ChandraHtmlTests(unittest.TestCase):
    def test_duplicate_bullets_are_normalized_without_losing_original(self) -> None:
        html = Path("tests/fixtures/synthetic/chandra_page.html").read_text(encoding="utf-8")
        blocks = parse_chandra_html(html)

        python_block = next(block for block in blocks if block.normalized_text == "Python")
        self.assertEqual(python_block.text, "• Python")
        self.assertEqual(python_block.normalized_text, "Python")
        self.assertEqual(python_block.bbox, [0.1, 0.2, 0.8, 0.25])


if __name__ == "__main__":
    unittest.main()
