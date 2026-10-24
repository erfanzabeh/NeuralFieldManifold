"""Check tutorial completeness without executing notebook code."""

import hashlib
import json
import unittest
from pathlib import Path


class TutorialContent(unittest.TestCase):
    def test_original_cells_and_explanations_are_preserved(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "tests/fixtures/tutorials.json").read_text())
        for name, expected in manifest.items():
            with self.subTest(notebook=name):
                notebook = json.loads((root / "docs/tutorials" / name).read_text())
                self.assertEqual(len(notebook["cells"]), expected["cells"])
                for index, digest in expected["markdown"].items():
                    cell = notebook["cells"][int(index)]
                    self.assertEqual(cell["cell_type"], "markdown")
                    source = "".join(cell["source"])
                    self.assertEqual(
                        hashlib.sha256(source.encode()).hexdigest(), digest
                    )
