import ast
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_renderer_has_no_analysis_imports():
    import plot_prego_fixed_geometry_decoding as plot
    tree = ast.parse(Path(plot.__file__).read_text())
    names = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not any(n and ('prego' in n or 'sklearn' in n or 'NeuralFieldManifold' in n) for n in names)
    assert plot.COLORS['geometry'] == '#7f1209'
    assert plot.COLORS['all_bands'] == '#92918d'


def test_confusion_uses_fixed_scale_and_retains_all_labels():
    from plot_prego_fixed_geometry_decoding import confusion_plot
    matrix = np.eye(6)
    fig, ax = confusion_plot(matrix)
    assert ax.images[0].get_clim() == (0, 1)
    assert ax.get_xlabel() == 'Predicted direction'
    assert ax.get_ylabel() == 'True direction'
    assert len(ax.texts) == 36
    assert [t.get_text() for t in ax.get_xticklabels()] == list('123456')
    plt.close(fig)
