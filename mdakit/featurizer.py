"""Feature helpers for building tICA-ready input matrices."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np
from MDAnalysis import Universe
from MDAnalysis.analysis import rms
from MDAnalysis.lib import distances


FeatureFn = Callable[[Universe], np.ndarray]


@dataclass
class Featurizer:
    """Collects per-frame feature callables and builds stacked arrays.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Trajectory source used by each feature callable.
    features : Sequence[Tuple[str, FeatureFn]], optional
        Initial list of (name, callable) feature definitions.
    """

    universe: Universe
    features: List[Tuple[str, FeatureFn]] = field(default_factory=list)

    def add(self, name: str, func: FeatureFn) -> "Featurizer":
        """Register a feature callable and return self for chaining."""
        if not callable(func):
            raise TypeError("Feature function must be callable")
        self.features.append((name, func))
        return self

    def names(self) -> List[str]:
        """Return the registered feature names in order."""
        return [name for name, _ in self.features]

    def array(self, dtype=np.float64) -> np.ndarray:
        """Evaluate all features and return shape (frames, n_features)."""
        if not self.features:
            raise ValueError("No features registered; add() at least one feature")

        traj = self.universe.trajectory
        n_frames = traj.n_frames
        cols = []
        for name, func in self.features:
            values = np.asarray(func(self.universe), dtype=dtype)
            if values.ndim != 1:
                raise ValueError(f"Feature '{name}' must return 1D array per frame")
            if values.shape[0] != n_frames:
                raise ValueError(
                    f"Feature '{name}' length {values.shape[0]} does not match trajectory ({n_frames})"
                )
            cols.append(values[:, None])
        return np.hstack(cols)


def rmsd_feat(
    select: str = "name CA",
    ref_frame: int = 0,
    center: bool = True,
    superposition: bool = True,
) -> FeatureFn:
    """Return a feature callable computing RMSD (Å) relative to ref_frame."""

    def compute(u: Universe) -> np.ndarray:
        ag = u.select_atoms(select)
        if ag.n_atoms == 0:
            raise ValueError(f"Selection '{select}' matched no atoms")
        traj = u.trajectory
        frame0 = traj.frame
        analysis = rms.RMSD(
            ag,
            ag,
            ref_frame=ref_frame,
            center=center,
            superposition=superposition,
        )
        analysis.run()
        traj[frame0]
        data = getattr(analysis, "results", analysis)
        return np.asarray(data.rmsd[:, 2], dtype=np.float64)

    return compute


def distance_feat(
    select1: str,
    select2: str,
    mass_weighted: bool = True,
) -> FeatureFn:
    """Return a feature callable computing COM/COG distance between selections."""

    def compute(u: Universe) -> np.ndarray:
        ag1 = u.select_atoms(select1)
        ag2 = u.select_atoms(select2)
        if ag1.n_atoms == 0:
            raise ValueError(f"Selection '{select1}' matched no atoms")
        if ag2.n_atoms == 0:
            raise ValueError(f"Selection '{select2}' matched no atoms")

        get_center1 = ag1.center_of_mass if mass_weighted else ag1.center_of_geometry
        get_center2 = ag2.center_of_mass if mass_weighted else ag2.center_of_geometry

        traj = u.trajectory
        frame0 = traj.frame
        vals = np.empty(traj.n_frames, dtype=np.float64)
        for i, _ in enumerate(traj):
            vals[i] = np.linalg.norm(get_center2() - get_center1())
        traj[frame0]
        return vals

    return compute


def gyration_feat(select: str = "all", mass_weighted: bool = True) -> FeatureFn:
    """Return feature computing radius of gyration for ``select`` per frame."""

    def compute(u: Universe) -> np.ndarray:
        ag = u.select_atoms(select)
        if ag.n_atoms == 0:
            raise ValueError(f"Selection '{select}' matched no atoms")
        traj = u.trajectory
        frame0 = traj.frame
        weights = ag.masses if mass_weighted else None
        out = np.empty(traj.n_frames, dtype=np.float64)
        for i, _ in enumerate(traj):
            out[i] = ag.radius_of_gyration(weights=weights)
        traj[frame0]
        return out

    return compute


def contact_count_feat(
    select1: str,
    select2: str,
    cutoff: float = 4.5,
    pbc: bool = False,
) -> FeatureFn:
    """Count contacts within ``cutoff`` between two selections for each frame."""

    def compute(u: Universe) -> np.ndarray:
        ag1 = u.select_atoms(select1)
        ag2 = u.select_atoms(select2)
        if ag1.n_atoms == 0:
            raise ValueError(f"Selection '{select1}' matched no atoms")
        if ag2.n_atoms == 0:
            raise ValueError(f"Selection '{select2}' matched no atoms")
        traj = u.trajectory
        frame0 = traj.frame
        out = np.empty(traj.n_frames, dtype=np.float64)
        for i, ts in enumerate(traj):
            box = ts.dimensions if pbc else None
            dmat = distances.distance_array(ag1.positions, ag2.positions, box=box)
            out[i] = float(np.count_nonzero(dmat < cutoff))
        traj[frame0]
        return out

    return compute


def hbond_count_feat(
    donors_sel: str,
    acceptors_sel: str,
    cutoff: float = 3.5,
    pbc: bool = False,
) -> FeatureFn:
    """Approximate hydrogen-bond counts using a donor-acceptor distance cutoff."""

    def compute(u: Universe) -> np.ndarray:
        donors = u.select_atoms(donors_sel)
        acceptors = u.select_atoms(acceptors_sel)
        if donors.n_atoms == 0:
            raise ValueError(f"Selection '{donors_sel}' matched no atoms")
        if acceptors.n_atoms == 0:
            raise ValueError(f"Selection '{acceptors_sel}' matched no atoms")
        traj = u.trajectory
        frame0 = traj.frame
        out = np.empty(traj.n_frames, dtype=np.float64)
        for i, ts in enumerate(traj):
            box = ts.dimensions if pbc else None
            dmat = distances.distance_array(donors.positions, acceptors.positions, box=box)
            out[i] = float(np.count_nonzero(dmat < cutoff))
        traj[frame0]
        return out

    return compute


def dihedral_feat(
    atom1: str,
    atom2: str,
    atom3: str,
    atom4: str,
    deg: bool = True,
) -> FeatureFn:
    """Return dihedral angle (radians or degrees) defined by four single-atom selections."""

    def compute(u: Universe) -> np.ndarray:
        sels = [u.select_atoms(sel) for sel in (atom1, atom2, atom3, atom4)]
        if any(sel.n_atoms != 1 for sel in sels):
            raise ValueError("Each dihedral selection must match exactly one atom")
        idx = [sel.indices[0] for sel in sels]
        traj = u.trajectory
        frame0 = traj.frame
        out = np.empty(traj.n_frames, dtype=np.float64)
        for i, _ in enumerate(traj):
            coords = u.atoms.positions
            angles = distances.calc_dihedrals(
                coords[idx[0]][None, :],
                coords[idx[1]][None, :],
                coords[idx[2]][None, :],
                coords[idx[3]][None, :],
            )[0]
            out[i] = np.degrees(angles) if deg else angles
        traj[frame0]
        return out

    return compute


def _build_backbone_indices(residues: Iterable) -> Tuple[List[Tuple[int, int, int, int]], List[int], List[Tuple[int, int, int, int]], List[int]]:
    phi_quads: List[Tuple[int, int, int, int]] = []
    phi_centers: List[int] = []
    psi_quads: List[Tuple[int, int, int, int]] = []
    psi_centers: List[int] = []
    residues = list(residues)
    for i in range(1, len(residues)):
        prev_res = residues[i - 1]
        curr_res = residues[i]
        try:
            c_prev = prev_res.atoms.select_atoms("name C")[0]
            n_curr = curr_res.atoms.select_atoms("name N")[0]
            ca_curr = curr_res.atoms.select_atoms("name CA")[0]
            c_curr = curr_res.atoms.select_atoms("name C")[0]
        except IndexError:
            continue
        phi_quads.append((c_prev.index, n_curr.index, ca_curr.index, c_curr.index))
        phi_centers.append(i)
    for i in range(len(residues) - 1):
        curr_res = residues[i]
        next_res = residues[i + 1]
        try:
            n_curr = curr_res.atoms.select_atoms("name N")[0]
            ca_curr = curr_res.atoms.select_atoms("name CA")[0]
            c_curr = curr_res.atoms.select_atoms("name C")[0]
            n_next = next_res.atoms.select_atoms("name N")[0]
        except IndexError:
            continue
        psi_quads.append((n_curr.index, ca_curr.index, c_curr.index, n_next.index))
        psi_centers.append(i)
    return phi_quads, phi_centers, psi_quads, psi_centers


def _dihedral_matrix(u: Universe, quads: Sequence[Tuple[int, int, int, int]]) -> np.ndarray:
    if not quads:
        return np.empty((u.trajectory.n_frames, 0), dtype=np.float64)
    quad_arr = np.asarray(quads, dtype=np.int64)
    traj = u.trajectory
    frame0 = traj.frame
    out = np.empty((traj.n_frames, quad_arr.shape[0]), dtype=np.float64)
    for i, _ in enumerate(traj):
        coords = u.atoms.positions
        out[i, :] = distances.calc_dihedrals(
            coords[quad_arr[:, 0]],
            coords[quad_arr[:, 1]],
            coords[quad_arr[:, 2]],
            coords[quad_arr[:, 3]],
        )
    traj[frame0]
    return out


def torsion_avg_feat(
    select: str = "protein",
    angle: str = "phi",
    deg: bool = True,
) -> FeatureFn:
    """Average backbone phi/psi angle over ``select`` residues per frame."""

    angle = angle.lower()
    if angle not in {"phi", "psi"}:
        raise ValueError("angle must be 'phi' or 'psi'")

    def compute(u: Universe) -> np.ndarray:
        residues = u.select_atoms(select).residues
        if len(residues) < 2:
            raise ValueError("Need at least two residues for backbone torsions")
        phi_quads, phi_centers, psi_quads, psi_centers = _build_backbone_indices(residues)
        if angle == "phi":
            mat = _dihedral_matrix(u, phi_quads)
        else:
            mat = _dihedral_matrix(u, psi_quads)
        if mat.shape[1] == 0:
            raise ValueError("No torsions could be computed for the selection")
        vals = np.mean(mat, axis=1)
        return np.degrees(vals) if deg else vals

    return compute


def secondary_structure_frac_feat(
    select: str = "protein",
    kind: str = "helix",
) -> FeatureFn:
    """Heuristic helical/beta fraction from backbone phi/psi angles per frame."""

    kind = kind.lower()
    if kind not in {"helix", "beta"}:
        raise ValueError("kind must be 'helix' or 'beta'")

    def compute(u: Universe) -> np.ndarray:
        residues = u.select_atoms(select).residues
        if len(residues) < 3:
            raise ValueError("Need at least three residues for secondary-structure fractions")
        phi_quads, phi_centers, psi_quads, psi_centers = _build_backbone_indices(residues)
        phi = _dihedral_matrix(u, phi_quads)
        psi = _dihedral_matrix(u, psi_quads)
        n_res = len(residues)
        if phi.shape[1] == 0 or psi.shape[1] == 0:
            raise ValueError("Could not compute backbone angles for selection")
        traj = u.trajectory
        out = np.empty(traj.n_frames, dtype=np.float64)
        for i in range(traj.n_frames):
            phi_vals = np.full(n_res, np.nan)
            psi_vals = np.full(n_res, np.nan)
            for ang, center in zip(phi[i], phi_centers):
                phi_vals[center] = np.degrees(ang)
            for ang, center in zip(psi[i], psi_centers):
                psi_vals[center] = np.degrees(ang)
            mask = ~np.isnan(phi_vals) & ~np.isnan(psi_vals)
            if not np.any(mask):
                out[i] = np.nan
                continue
            phi_sel = phi_vals[mask]
            psi_sel = psi_vals[mask]
            if kind == "helix":
                cond = (
                    (phi_sel >= -120.0)
                    & (phi_sel <= -30.0)
                    & (psi_sel >= -70.0)
                    & (psi_sel <= -10.0)
                )
            else:  # beta
                cond = (
                    (phi_sel >= -180.0)
                    & (phi_sel <= -60.0)
                    & (psi_sel >= 90.0)
                    & (psi_sel <= 180.0)
                )
            out[i] = float(cond.mean())
        return out

    return compute


def sasa_feat(
    select: str = "all",
    probe_radius: float = 1.4,
    n_sphere_points: int = 960,
) -> FeatureFn:
    """Total Shrake–Rupley SASA for ``select`` per frame."""

    try:
        from MDAnalysis.analysis.sasa import ShrakeRupley
    except ImportError as err:  # pragma: no cover - optional dependency handled at runtime
        raise ImportError("sasa_feat requires MDAnalysis.analysis.sasa") from err

    def compute(u: Universe) -> np.ndarray:
        ag = u.select_atoms(select)
        if ag.n_atoms == 0:
            raise ValueError(f"Selection '{select}' matched no atoms")
        sr = ShrakeRupley(
            ag,
            probe_radius=probe_radius,
            n_sphere_points=n_sphere_points,
            mode="atom",
        )
        sr.run()
        data = getattr(sr.results, "atom_sasa", None)
        if data is None:
            data = sr.atom_sasa  # type: ignore[attr-defined]
        return np.asarray(data.sum(axis=1), dtype=np.float64)

    return compute


def plane_distance_feat(
    select: str = "all",
    axis: int = 2,
    plane_position: float = 0.0,
) -> FeatureFn:
    """Minimum absolute distance along ``axis`` to a plane at ``plane_position``."""

    def compute(u: Universe) -> np.ndarray:
        ag = u.select_atoms(select)
        if ag.n_atoms == 0:
            raise ValueError(f"Selection '{select}' matched no atoms")
        traj = u.trajectory
        frame0 = traj.frame
        out = np.empty(traj.n_frames, dtype=np.float64)
        for i, _ in enumerate(traj):
            coords = ag.positions[:, axis]
            out[i] = float(np.min(np.abs(coords - plane_position)))
        traj[frame0]
        return out

    return compute


__all__ = [
    "Featurizer",
    "rmsd_feat",
    "distance_feat",
    "gyration_feat",
    "contact_count_feat",
    "hbond_count_feat",
    "dihedral_feat",
    "torsion_avg_feat",
    "secondary_structure_frac_feat",
    "sasa_feat",
    "plane_distance_feat",
]
__all__ = ["Featurizer", "rmsd_feat", "distance_feat"]
