import numpy as np
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader

from mdakit.tica import TICA


def make_universe(n_frames=5, n_atoms=2):
    coords = np.cumsum(
        np.random.randn(n_frames, n_atoms, 3).astype(np.float32) * 0.01,
        axis=0,
    )
    u = mda.Universe.empty(n_atoms, trajectory=True)
    u.add_TopologyAttr("name", [f"A{i}" for i in range(n_atoms)])
    u.trajectory = MemoryReader(coords)
    return u


def test_build_lagged_blocks_pure():
    # Build a simple (T, D) matrix with predictable values
    T, D, lag = 10, 6, 3
    X = np.arange(T * D, dtype=float).reshape(T, D)

    # We only need a TICA instance; we'll pass X explicitly
    u = make_universe(n_frames=2, n_atoms=1)
    tica = TICA(u, lag=lag)

    X_out, X0, Xtau, n_pairs, T_out, lag_out = tica._build_lagged_blocks(X=X, lag=lag)

    assert X_out.shape == (T, D)
    assert X0.shape == (T - lag, D)
    assert Xtau.shape == (T - lag, D)
    assert n_pairs == T - lag
    assert T_out == T and lag_out == lag
    # Values: Xtau should equal X shifted by lag
    assert np.allclose(Xtau, X[lag:, :])
    assert np.allclose(X0, X[:-lag, :])
    print("_build_lagged_blocks (pure) passed.")


def test_build_lagged_blocks_with_coords():
    # Use the class's self._coords machinery
    T, D, lag = 8, 4, 2
    X = np.arange(T * D, dtype=float).reshape(T, D)

    u = make_universe(n_frames=T, n_atoms=1)
    tica = TICA(u, lag=lag)
    # Inject coords directly to mimic what _single_frame collects
    tica._coords = X.copy()

    X_out, X0, Xtau, n_pairs, T_out, lag_out = tica._build_lagged_blocks()

    assert X_out.shape == (T, D)
    assert X0.shape == (T - lag, D)
    assert Xtau.shape == (T - lag, D)
    assert n_pairs == T - lag
    assert T_out == T and lag_out == lag
    assert np.allclose(Xtau, X[lag:, :])
    assert np.allclose(X0, X[:-lag, :])
    print("_build_lagged_blocks (self._coords) passed.")


def test_center_and_scale_stub():
    # This test guides implementation; it will report TODO until implemented
    T, D, lag = 12, 5, 4
    rng = np.random.default_rng(0)
    X = rng.normal(size=(T, D))

    u = make_universe(n_frames=T, n_atoms=1)
    tica = TICA(u, lag=lag, scale=False)

    _, X0, Xtau, n_pairs, *_ = tica._build_lagged_blocks(X=X, lag=lag)
    try:
        result = tica._center_and_scale(X0, Xtau, scale=False)
    except TypeError:
        # Signature not implemented yet
        print("_center_and_scale not implemented yet (signature mismatch). TODO: implement.")
        return

    if result is None:
        print("_center_and_scale returns None. TODO: implement body to return (X0c, Xtauc, mean, std_or_none).")
        return

    X0c, Xtauc, mean, std = result
    assert X0c.shape == X0.shape and Xtauc.shape == Xtau.shape
    # Check zero mean (global) when scale=False
    stacked = np.vstack([X0c, Xtauc])
    m = stacked.mean(axis=0)
    assert np.allclose(m, 0, atol=1e-10), f"Mean not ~0: max |mean|={np.abs(m).max()}"
    assert std is None
    print("_center_and_scale (mean-only) passed.")


def test_estimate_covariance_values():
    # Construct deterministic centered blocks and verify covariance math
    rng = np.random.default_rng(42)
    n_pairs, D = 25, 7
    X0c = rng.normal(size=(n_pairs, D))
    Xtauc = rng.normal(size=(n_pairs, D))

    u = make_universe(n_frames=2, n_atoms=1)
    tica = TICA(u, lag=3)

    # Case 1: reversible=False, bessel=False
    C0, Ctau = tica._estimate_covariance(X0c, Xtauc, n_pairs=n_pairs, reversible=False, bessel=False)
    denom = n_pairs
    C0_ref = (X0c.T @ X0c) / denom
    Ctau_ref = (X0c.T @ Xtauc) / denom
    C0_ref = 0.5 * (C0_ref + C0_ref.T)
    Ctau_ref = 0.5 * (Ctau_ref + Ctau_ref.T)
    assert np.allclose(C0, C0_ref)
    assert np.allclose(Ctau, Ctau_ref)
    assert np.allclose(C0, C0.T)
    assert np.allclose(Ctau, Ctau.T)

    # Case 2: reversible=True, bessel=False
    C0, Ctau = tica._estimate_covariance(X0c, Xtauc, n_pairs=n_pairs, reversible=True, bessel=False)
    C0_ref = (X0c.T @ X0c + Xtauc.T @ Xtauc) * (0.5 / denom)
    Ctau_ref = (X0c.T @ Xtauc + Xtauc.T @ X0c) * (0.5 / denom)
    C0_ref = 0.5 * (C0_ref + C0_ref.T)
    Ctau_ref = 0.5 * (Ctau_ref + Ctau_ref.T)
    assert np.allclose(C0, C0_ref)
    assert np.allclose(Ctau, Ctau_ref)

    # Case 3: reversible=True, bessel=True
    C0, Ctau = tica._estimate_covariance(X0c, Xtauc, n_pairs=n_pairs, reversible=True, bessel=True)
    denom = n_pairs - 1
    C0_ref = (X0c.T @ X0c + Xtauc.T @ Xtauc) * (0.5 / denom)
    Ctau_ref = (X0c.T @ Xtauc + Xtauc.T @ X0c) * (0.5 / denom)
    C0_ref = 0.5 * (C0_ref + C0_ref.T)
    Ctau_ref = 0.5 * (Ctau_ref + Ctau_ref.T)
    assert np.allclose(C0, C0_ref)
    assert np.allclose(Ctau, Ctau_ref)
    print("_estimate_covariance passed for reversible/non-reversible and Bessel options.")


def test_regularize_stub():
    # Prepare a nearly-symmetric PSD matrix and check diagonal loading
    rng = np.random.default_rng(7)
    D = 6
    A = rng.normal(size=(D, D))
    C0 = A.T @ A  # symmetric PSD
    C0_asy = C0.copy()
    C0_asy[0, 1] += 1e-6  # introduce tiny asymmetry

    u = make_universe(n_frames=2, n_atoms=1)
    tica = TICA(u, lag=3)

    try:
        C0_reg = tica._regularize(C0_asy, 1e-3)
    except TypeError:
        print("_regularize not implemented yet (signature mismatch). TODO: implement.")
        return

    if C0_reg is None:
        print("_regularize returns None. TODO: implement to return C0_reg.")
        return

    # Expected: symmetrize + eps * I, without modifying input
    C0_sym = 0.5 * (C0_asy + C0_asy.T)
    expected = C0_sym + 1e-3 * np.eye(D)
    assert np.allclose(C0_reg, expected)
    assert np.allclose(C0_reg, C0_reg.T)

    # Check eps=0 produces pure symmetrization
    C0_reg0 = tica._regularize(C0_asy, 0.0)
    assert np.allclose(C0_reg0, C0_sym)
    print('regularization passed')


def main():
    test_build_lagged_blocks_pure()
    test_build_lagged_blocks_with_coords()
    test_center_and_scale_stub()
    test_estimate_covariance_values()
    test_regularize_stub()
    print("All helper tests finished.")


if __name__ == "__main__":
    main()
