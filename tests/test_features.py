import pathlib
import sys

import numpy as np
import pytest
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tica.featurizer import (
    Featurizer,
    contact_count_feat,
    dihedral_feat,
    distance_feat,
    gyration_feat,
    hbond_count_feat,
    plane_distance_feat,
    rmsd_feat,
    sasa_feat,
    secondary_structure_frac_feat,
    torsion_avg_feat,
)


def make_u(n_frames=6, seed=0):
    n_res = 3
    atoms_per_res = 4  # N, CA, C, O
    n_atoms = n_res * atoms_per_res
    atom_resindex = np.repeat(np.arange(n_res), atoms_per_res)
    coords0 = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.45, 0.1, 0.0],
            [2.90, 0.2, 0.1],
            [3.60, 0.4, 0.2],
            [4.30, 0.8, 0.3],
            [5.75, 0.9, 0.4],
            [7.20, 1.0, 0.5],
            [7.90, 1.2, 0.6],
            [8.60, 1.5, 0.7],
            [10.05, 1.6, 0.8],
            [11.50, 1.7, 0.9],
            [12.20, 1.9, 1.0],
        ],
        dtype=np.float32,
    )
    rng = np.random.default_rng(seed)
    displacements = rng.normal(scale=0.05, size=(n_frames, 3)).astype(np.float32)
    coords = np.stack([coords0 + disp for disp in displacements], axis=0)
    u = mda.Universe.empty(
        n_atoms,
        n_residues=n_res,
        atom_resindex=atom_resindex,
        residue_segindex=np.zeros(n_res, dtype=int),
        trajectory=True,
    )
    u.add_TopologyAttr("resid", np.arange(1, n_res + 1))
    u.add_TopologyAttr("resname", ["ALA"] * n_res)
    names = [name for _ in range(n_res) for name in ("N", "CA", "C", "O")]
    u.add_TopologyAttr("name", names)
    u.add_TopologyAttr("type", names)
    u.add_TopologyAttr("mass", np.ones(n_atoms))
    u.trajectory = MemoryReader(coords)
    return u


def test_array_names_and_shape():
    u = make_u()
    feat = Featurizer(u)
    feat.add("rmsd", rmsd_feat(select="all"))
    feat.add("dist", distance_feat("name N", "name CA", mass_weighted=False))
    frame0 = u.trajectory.frame
    X = feat.array()
    assert X.shape == (u.trajectory.n_frames, 2)
    assert feat.names() == ["rmsd", "dist"]
    assert u.trajectory.frame == frame0


def test_feature_length_validation():
    u = make_u()

    def bad(_u):
        return np.arange(3)

    feat = Featurizer(u)
    feat.add("bad", bad)
    with pytest.raises(ValueError):
        feat.array()


def test_no_features_raises():
    u = make_u()
    feat = Featurizer(u)
    with pytest.raises(ValueError):
        feat.array()


def test_gyration_matches_manual():
    u = make_u()
    rg_series = gyration_feat("all")(u)
    coords = u.trajectory.ts.positions
    center = coords.mean(axis=0)
    manual = np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=1)))
    assert np.isclose(rg_series[0], manual)


def test_contact_and_hbond_counts():
    u = make_u()
    contact = contact_count_feat("resid 1", "resid 2", cutoff=10.0)(u)
    expected_pairs = 4 * 4  # atoms per residue squared
    assert np.allclose(contact, expected_pairs)
    hbonds = hbond_count_feat("resid 1 and name O", "resid 2 and name N", cutoff=2.0)(u)
    assert np.all(hbonds >= 0)


def test_dihedral_and_torsion_features():
    u = make_u()
    dih = dihedral_feat("resid 1 and name N", "resid 1 and name CA", "resid 1 and name C", "resid 2 and name N")(u)
    assert dih.shape == (u.trajectory.n_frames,)
    tors = torsion_avg_feat(select="resid 1-3", angle="phi")(u)
    assert tors.shape == (u.trajectory.n_frames,)


def test_secondary_structure_fraction_and_plane_distance():
    u = make_u()
    frac = secondary_structure_frac_feat(select="resid 1-3", kind="helix")(u)
    assert frac.shape == (u.trajectory.n_frames,)
    assert np.all((frac >= 0) | np.isnan(frac))
    plane = plane_distance_feat("all", axis=2, plane_position=0.5)(u)
    assert plane.shape == (u.trajectory.n_frames,)
    assert np.all(plane >= 0)


def test_sasa_feature_runs():
    pytest.importorskip("MDAnalysis.analysis.sasa")
    u = make_u()
    sasa = sasa_feat("all")(u)
    assert sasa.shape == (u.trajectory.n_frames,)
    assert np.all(sasa >= 0)
