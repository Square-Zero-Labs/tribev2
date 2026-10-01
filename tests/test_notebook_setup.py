"""Check Colab setup without installing packages or downloading models."""
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {"tribe_demo.ipynb": 4, "tribe_thumbnail_study.ipynb": 6}


class NotebookSetupTests(unittest.TestCase):
    def cells(self, name):
        return json.loads((ROOT / name).read_text())["cells"]

    def test_code_cells_compile(self):
        for name in NOTEBOOKS:
            for index, cell in enumerate(self.cells(name)):
                if cell["cell_type"] == "code":
                    compile("".join(cell["source"]), f"{name}:{index}", "exec")

    def test_install_resolves_together_and_requires_restart(self):
        for name, import_index in NOTEBOOKS.items():
            with self.subTest(notebook=name):
                cells = self.cells(name)
                namespace = {}
                with patch("subprocess.run") as run:
                    with self.assertRaisesRegex(SystemExit, "Restart session"):
                        exec("".join(cells[2]["source"]), namespace)
                run.assert_called_once()
                command = run.call_args.args[0]
                self.assertEqual(command[:3], [sys.executable, "-m", "pip"])
                self.assertIn("torch==2.6.0", command)
                self.assertIn("torchvision==0.21.0", command)
                self.assertIn("torchaudio==2.6.0", command)
                self.assertIn("transformers==5.17.0", command)
                self.assertIn("numpy==2.2.6", command)
                self.assertIn("scipy==1.15.3", command)
                self.assertIn("exca==0.5.20", command)
                self.assertIn("pyvista==0.46.5", command)
                self.assertTrue(run.call_args.kwargs["check"])
                self.assertFalse(any("force-reinstall" in arg for arg in command))
                with self.assertRaisesRegex(RuntimeError, "Restart session"):
                    exec("".join(cells[import_index]["source"]), namespace)

    def test_failed_install_blocks_imports(self):
        for name, import_index in NOTEBOOKS.items():
            namespace = {}
            cells = self.cells(name)
            with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "pip")):
                with self.assertRaises(subprocess.CalledProcessError):
                    exec("".join(cells[2]["source"]), namespace)
            with self.assertRaisesRegex(RuntimeError, "Restart session"):
                exec("".join(cells[import_index]["source"]), namespace)

    def test_incompatible_numpy_rejected_before_extension_imports(self):
        for name, import_index in NOTEBOOKS.items():
            with patch("importlib.metadata.version", return_value="2.0.2"):
                with self.assertRaisesRegex(RuntimeError, "Expected numpy 2.2.6"):
                    exec("".join(self.cells(name)[import_index]["source"]), {})


if __name__ == "__main__":
    unittest.main()
