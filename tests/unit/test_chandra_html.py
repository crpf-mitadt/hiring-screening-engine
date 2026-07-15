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

    def test_tables_are_preserved_and_image_blocks_are_excluded(self) -> None:
        html = """
        <div data-label="Image" data-bbox="[0,0,1,1]">portrait</div>
        <table>
          <tr>
            <td data-label="Table" data-bbox="[0.1,0.1,0.2,0.2]">Python</td>
          </tr>
        </table>
        """

        blocks = parse_chandra_html(html)

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].label, "Table")
        self.assertEqual(blocks[0].text, "Python")


if __name__ == "__main__":
    unittest.main()
