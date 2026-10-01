"""Integration smoke test; run with plotting dependencies and IPython installed."""
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


@unittest.skipUnless(
    all(importlib.util.find_spec(name) is not None for name in ("IPython", "pyvista", "neuralset")),
    "Requires plotting dependencies and IPython",
)
class PlottingImportTests(unittest.TestCase):
    def test_imports_inside_ipython(self):
        # A plain Python import misses PyVista's IPython completion hooks.
        subprocess.run(
            [sys.executable, "-c", """
from IPython.core.interactiveshell import InteractiveShell
shell = InteractiveShell.instance()
from tribev2.plotting import PlotBrain, PlotBrainNilearn
import pyvista as pv
mesh = pv.PolyData([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], [3, 0, 1, 2])
assert mesh.n_points == 3
assert mesh.n_cells == 1
"""],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            timeout=180,
        )


if __name__ == "__main__":
    unittest.main()
