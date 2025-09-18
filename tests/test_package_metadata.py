import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mdakit


def test_dunder_all_exports_modules():
    exported = set(mdakit.__all__)
    assert {"tica", "plotting", "featurizer"}.issubset(exported)
    assert {"Featurizer", "rmsd_feat", "distance_feat"}.issubset(exported)


def test_version_string():
    assert isinstance(mdakit.__version__, str)
    assert mdakit.__version__
