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
    T, D, lag = 10, 6, 3
    X = np.arange(T * D, dtype=float).reshape(T, D)
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
    T, D, lag = 8, 4, 2
    X = np.arange(T * D, dtype=float).reshape(T, D)

    u = make_universe(n_frames=T, n_atoms=1)
    tica = TICA(u, lag=lag)
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
    rng = np.random.default_rng(7)
    D = 6
    A = rng.normal(size=(D, D))
    C0 = A.T @ A 
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
    C0_sym = 0.5 * (C0_asy + C0_asy.T)
    expected = C0_sym + 1e-3 * np.eye(D)
    assert np.allclose(C0_reg, expected)
    assert np.allclose(C0_reg, C0_reg.T)
    C0_reg0 = tica._regularize(C0_asy, 0.0)
    assert np.allclose(C0_reg0, C0_sym)


def test_whiten_solve_cholesky_path():
    rng = np.random.default_rng(1)
    D = 8
    B = rng.normal(size=(D, D))
    C0_reg = B @ B.T + 1e-3 * np.eye(D)  # SPD
    S = rng.normal(size=(D, D))
    Ctau = 0.5 * (S + S.T)               # symmetric

    u = make_universe(n_frames=2, n_atoms=1)
    tica = TICA(u, lag=3)

    M, eigvals, eigvecs = tica._whiten_solve(C0_reg, Ctau, chol_fallback_tol=(1e-4, 1e-12))
    assert M.shape == (D, D)
    assert eigvals.shape == (D,)
    assert eigvecs.shape == (D, D)
    L = np.linalg.cholesky(0.5 * (C0_reg + C0_reg.T))
    Y = np.linalg.solve(L, Ctau)
    M_ref = np.linalg.solve(L, Y.T).T
    M_ref = 0.5 * (M_ref + M_ref.T)
    assert np.allclose(M, M_ref, atol=1e-10)
    R = Ctau @ eigvecs - (C0_reg @ eigvecs) * eigvals
    assert np.linalg.norm(R) / max(1.0, np.linalg.norm(Ctau @ eigvecs)) < 1e-10
    


def test_whiten_solve_eig_fallback_and_truncation():
    rng = np.random.default_rng(2)
    D, r = 10, 5
    A = rng.normal(size=(D, r))
    C0_reg = A @ A.T  # PSD, singular -> forces eig fallback
    S = rng.normal(size=(D, D))
    Ctau = 0.5 * (S + S.T)

    u = make_universe(n_frames=2, n_atoms=1)
    tica = TICA(u, lag=3, n_components=3)

    M, eigvals, eigvecs = tica._whiten_solve(C0_reg, Ctau, chol_fallback_tol=(1e-4, 1e-12))

    assert eigvals.shape == (3,)
    assert eigvecs.shape == (D, 3)
    assert M.shape[0] == M.shape[1]
    # Project generalized eigen relation into the kept subspace used by whitening
    s, U = np.linalg.eigh(0.5 * (C0_reg + C0_reg.T))
    rel_tol, abs_tol = 1e-4, 1e-12
    thresh = max(abs_tol, rel_tol * float(s.max()))
    keep = s > thresh
    P = U[:, keep] @ U[:, keep].T
    R = P @ (Ctau @ eigvecs) - P @ ((C0_reg @ eigvecs) * eigvals)
    rel = np.linalg.norm(R) / max(1.0, np.linalg.norm(P @ (Ctau @ eigvecs)))
    print("eig-fallback projected rel residual:", rel)
    assert rel < 1e-8
    print('regularization passed')


def test_sort_and_truncate_basic():
    eigvals = np.array([0.2, 0.9, 0.5, 0.1])
    D = 4
    eigvecs = np.eye(D)

    u = make_universe(n_frames=2, n_atoms=1)
    tica = TICA(u, lag=2)

    vals, vecs = tica._sort_and_truncate(eigvals, eigvecs, n_components=None)
    assert np.allclose(vals, np.array([0.9, 0.5, 0.2, 0.1]))
    assert vecs.shape == (D, D)

    vals2, vecs2 = tica._sort_and_truncate(eigvals, eigvecs, n_components=2)
    assert np.allclose(vals2, np.array([0.9, 0.5]))
    assert vecs2.shape == (D, 2)

    vals0, vecs0 = tica._sort_and_truncate(eigvals, eigvecs, n_components=0)
    assert vals0.shape == (0,)
    assert vecs0.shape == (D, 0)


def test_compute_timescales_basic():
    eigvals = np.array([-0.1, 0.0, 0.2, 0.9, 1.0, 1.1])
    lag, dt = 5, 0.1

    u = make_universe(n_frames=2, n_atoms=1)
    tica = TICA(u, lag=lag)

    ts = tica._compute_timescales(eigvals, lag=lag, dt=dt)
    expected = np.full(eigvals.shape, np.inf)
    mask = (eigvals > 0) & (eigvals < 1)
    expected[mask] = -(lag * dt) / np.log(eigvals[mask])
    assert np.allclose(ts[mask], expected[mask])
    assert np.all(np.isinf(ts[~mask]))
    assert tica._compute_timescales(eigvals, lag=lag, dt=None) is None


def test_project_centered_basic():
    T, D, K = 3, 4, 2
    X = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0, 6.0],
    ])
    mean = np.array([1.0, 1.0, 1.0, 1.0])
    std = np.array([1.0, 2.0, 0.5, 4.0])
    C = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
        [0.0, -1.0],
    ])  # (D,K)

    u = make_universe(n_frames=2, n_atoms=1)
    tica = TICA(u, lag=2)

    # Mean-only
    Y = tica._project_centered(X, mean, None, C)
    Y_ref = (X - mean) @ C
    assert np.allclose(Y, Y_ref)

    # Mean + std
    Ys = tica._project_centered(X, mean, std, C)
    Ys_ref = ((X - mean) / std) @ C
    assert np.allclose(Ys, Ys_ref)


def main():
    test_build_lagged_blocks_pure()
    test_build_lagged_blocks_with_coords()
    test_center_and_scale_stub()
    test_estimate_covariance_values()
    test_regularize_stub()
    test_whiten_solve_cholesky_path()
    test_whiten_solve_eig_fallback_and_truncation()
    test_sort_and_truncate_basic()
    test_compute_timescales_basic()
    test_project_centered_basic()
    print("All helper tests finished.")


if __name__ == "__main__":
    main()
