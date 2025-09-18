import pathlib
import sys

import numpy as np
import pytest
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tica.tica import TICA


def make_universe(n_frames=6, n_atoms=3, seed=0):
    """Build a tiny in-memory MDAnalysis Universe with reproducible coordinates."""
    rng = np.random.default_rng(seed)
    coords = rng.normal(scale=0.01, size=(n_frames, n_atoms, 3)).astype(np.float32)
    coords = np.cumsum(coords, axis=0)
    u = mda.Universe.empty(n_atoms, trajectory=True)
    u.add_TopologyAttr("name", [f"A{i}" for i in range(n_atoms)])
    u.trajectory = MemoryReader(coords)
    return u


def test_blocks_arr():
    T, D, lag = 10, 6, 3
    X = np.arange(T * D, dtype=float).reshape(T, D)
    u = make_universe(n_frames=2, n_atoms=1)
    tica = TICA(u, lag=lag)

    X_out, X0, Xtau, n_pairs, T_out, lag_out = tica._build_lagged_blocks(X=X, lag=lag)

    assert X_out.shape == (T, D)
    assert X0.shape == (T - lag, D)
    assert Xtau.shape == (T - lag, D)
    assert n_pairs == T - lag
    assert T_out == T
    assert lag_out == lag
    assert np.allclose(X0, X[:-lag, :])
    assert np.allclose(Xtau, X[lag:, :])


def test_blocks_coords():
    T, D, lag = 8, 4, 2
    X = np.arange(T * D, dtype=float).reshape(T, D)

    u = make_universe(n_frames=T + 1, n_atoms=2)
    tica = TICA(u, lag=lag)
    tica._coords = X.copy()

    X_out, X0, Xtau, n_pairs, T_out, lag_out = tica._build_lagged_blocks()

    assert X_out.shape == (T, D)
    assert X0.shape == (T - lag, D)
    assert Xtau.shape == (T - lag, D)
    assert n_pairs == T - lag
    assert T_out == T
    assert lag_out == lag
    assert np.allclose(X0, X[:-lag, :])
    assert np.allclose(Xtau, X[lag:, :])


def test_center_mean():
    T, D, lag = 12, 5, 4
    rng = np.random.default_rng(0)
    X = rng.normal(size=(T, D))

    u = make_universe(n_frames=T, n_atoms=1)
    tica = TICA(u, lag=lag, scale=False)

    _, X0, Xtau, *_ = tica._build_lagged_blocks(X=X, lag=lag)
    X0c, Xtauc, mean, std = tica._center_and_scale(X0, Xtau, scale=False)

    stacked = np.vstack([X0c, Xtauc])
    assert np.allclose(stacked.mean(axis=0), 0.0)
    assert std is None
    expected_mean = np.vstack([X[:-lag], X[lag:]]).mean(axis=0)
    assert np.allclose(mean, expected_mean)


def test_center_scale():
    T, D, lag = 16, 3, 3
    rng = np.random.default_rng(1)
    X = rng.normal(size=(T, D))

    u = make_universe(n_frames=T, n_atoms=1)
    tica = TICA(u, lag=lag, scale=True)

    _, X0, Xtau, *_ = tica._build_lagged_blocks(X=X, lag=lag)
    X0c, Xtauc, mean, std = tica._center_and_scale(X0, Xtau, scale=True)

    stacked = np.vstack([X0c, Xtauc])
    assert np.allclose(stacked.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(stacked.std(axis=0, ddof=1), 1.0, atol=1e-12)
    assert std is not None
    expected_mean = np.vstack([X[:-lag], X[lag:]]).mean(axis=0)
    expected_std = np.vstack([X[:-lag], X[lag:]]).std(axis=0, ddof=1)
    assert np.allclose(mean, expected_mean)
    assert np.allclose(std, expected_std)


def test_cov():
    rng = np.random.default_rng(42)
    n_pairs, D = 25, 7
    X0c = rng.normal(size=(n_pairs, D))
    Xtauc = rng.normal(size=(n_pairs, D))

    u = make_universe(n_frames=2, n_atoms=1)
    tica = TICA(u, lag=3)

    C0, Ctau = tica._estimate_covariance(X0c, Xtauc, n_pairs=n_pairs, reversible=False, bessel=False)
    denom = n_pairs
    C0_ref = (X0c.T @ X0c) / denom
    Ctau_ref = (X0c.T @ Xtauc) / denom
    C0_ref = 0.5 * (C0_ref + C0_ref.T)
    Ctau_ref = 0.5 * (Ctau_ref + Ctau_ref.T)
    assert np.allclose(C0, C0_ref)
    assert np.allclose(Ctau, Ctau_ref)

    C0_rev, Ctau_rev = tica._estimate_covariance(X0c, Xtauc, n_pairs=n_pairs, reversible=True, bessel=True)
    denom = n_pairs - 1
    C0_ref = (X0c.T @ X0c + Xtauc.T @ Xtauc) * (0.5 / denom)
    Ctau_ref = (X0c.T @ Xtauc + Xtauc.T @ X0c) * (0.5 / denom)
    C0_ref = 0.5 * (C0_ref + C0_ref.T)
    Ctau_ref = 0.5 * (Ctau_ref + Ctau_ref.T)
    assert np.allclose(C0_rev, C0_ref)
    assert np.allclose(Ctau_rev, Ctau_ref)


def test_reg_matrix():
    rng = np.random.default_rng(7)
    D = 6
    A = rng.normal(size=(D, D))
    C0 = A.T @ A
    C0_asy = C0.copy()
    C0_asy[0, 1] += 1e-6

    u = make_universe(n_frames=2, n_atoms=1)
    tica = TICA(u, lag=3)

    C0_reg = tica._regularize(C0_asy, 1e-3)
    C0_sym = 0.5 * (C0_asy + C0_asy.T)
    expected = C0_sym + 1e-3 * np.eye(D)
    assert np.allclose(C0_reg, expected)
    assert np.allclose(C0_reg, C0_reg.T)
    assert C0_reg.dtype == expected.dtype

    C0_reg0 = tica._regularize(C0_asy, 0.0)
    assert np.allclose(C0_reg0, C0_sym)


def test_reg_neg():
    u = make_universe(n_frames=3, n_atoms=1)
    tica = TICA(u, lag=1)
    with pytest.raises(ValueError):
        tica._regularize(np.eye(3), -1e-5)


def test_whiten_chol():
    rng = np.random.default_rng(1)
    D = 8
    B = rng.normal(size=(D, D))
    C0_reg = B @ B.T + 1e-3 * np.eye(D)
    S = rng.normal(size=(D, D))
    Ctau = 0.5 * (S + S.T)

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
    residual = Ctau @ eigvecs - (C0_reg @ eigvecs) * eigvals
    assert np.linalg.norm(residual) / max(1.0, np.linalg.norm(Ctau @ eigvecs)) < 1e-10


def test_whiten_eig():
    rng = np.random.default_rng(2)
    D, r = 10, 5
    A = rng.normal(size=(D, r))
    C0_reg = A @ A.T
    S = rng.normal(size=(D, D))
    Ctau = 0.5 * (S + S.T)

    u = make_universe(n_frames=2, n_atoms=1)
    tica = TICA(u, lag=3, n_components=3)

    M, eigvals, eigvecs = tica._whiten_solve(C0_reg, Ctau, chol_fallback_tol=(1e-4, 1e-12))

    assert eigvals.shape == (3,)
    assert eigvecs.shape == (D, 3)
    s, U = np.linalg.eigh(0.5 * (C0_reg + C0_reg.T))
    rel_tol, abs_tol = 1e-4, 1e-12
    thresh = max(abs_tol, rel_tol * float(s.max()))
    keep = s > thresh
    P = U[:, keep] @ U[:, keep].T
    residual = P @ (Ctau @ eigvecs) - P @ ((C0_reg @ eigvecs) * eigvals)
    rel_norm = np.linalg.norm(residual) / max(1.0, np.linalg.norm(P @ (Ctau @ eigvecs)))
    assert rel_norm < 1e-8


def test_sort_trunc():
    eigvals = np.array([0.2, 0.9, 0.5, 0.1])
    eigvecs = np.eye(4)

    u = make_universe(n_frames=3, n_atoms=1)
    tica = TICA(u, lag=2)

    vals, vecs = tica._sort_and_truncate(eigvals, eigvecs, n_components=None)
    assert np.allclose(vals, np.array([0.9, 0.5, 0.2, 0.1]))
    assert vecs.shape == (4, 4)

    vals2, vecs2 = tica._sort_and_truncate(eigvals, eigvecs, n_components=2)
    assert np.allclose(vals2, np.array([0.9, 0.5]))
    assert vecs2.shape == (4, 2)

    vals0, vecs0 = tica._sort_and_truncate(eigvals, eigvecs, n_components=0)
    assert vals0.shape == (0,)
    assert vecs0.shape == (4, 0)


def test_timescales():
    eigvals = np.array([-0.1, 0.0, 0.2, 0.9, 1.0, 1.1])
    lag, dt = 5, 0.1

    u = make_universe(n_frames=3, n_atoms=1)
    tica = TICA(u, lag=lag)

    ts = tica._compute_timescales(eigvals, lag=lag, dt=dt)
    expected = np.full(eigvals.shape, np.inf)
    mask = (eigvals > 0) & (eigvals < 1)
    expected[mask] = -(lag * dt) / np.log(eigvals[mask])
    assert np.allclose(ts[mask], expected[mask])
    assert np.all(np.isinf(ts[~mask]))
    assert tica._compute_timescales(eigvals, lag=lag, dt=None) is None


def test_project_center():
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
    ])

    u = make_universe(n_frames=3, n_atoms=1)
    tica = TICA(u, lag=2)

    Y = tica._project_centered(X, mean, None, C)
    Y_ref = (X - mean) @ C
    assert np.allclose(Y, Y_ref)

    Ys = tica._project_centered(X, mean, std, C)
    Ys_ref = ((X - mean) / std) @ C
    assert np.allclose(Ys, Ys_ref)


def test_run():
    u = make_universe(n_frames=12, n_atoms=4)
    tica = TICA(u, lag=2, n_components=2, regularization=1e-4).run()
    d = u.select_atoms("all").n_atoms * 3
    assert tica.components_.shape == (d, 2)
    assert tica.proj_.shape == (u.trajectory.n_frames, 2)
    assert tica.eigenvalues_.shape == (2,)
    assert tica.timescales_ is None or tica.timescales_.shape == (2,)


def test_fit():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 6))
    u = make_universe(n_frames=5, n_atoms=2)
    tica = TICA(u, lag=2, n_components=2, scale=True)
    tica.fit_from_array(X, lag=3, scale=True, n_components=2, dt=0.5)
    assert tica.components_.shape == (6, 2)
    assert tica.proj_.shape == (40, 2)
    assert tica.timescales_.shape == (2,)
    assert np.all(np.isfinite(tica.eigenvalues_))


def test_transform():
    u = make_universe(n_frames=10, n_atoms=4)
    tica = TICA(u, lag=2, n_components=3).run()

    original_proj = tica.proj_.copy()
    traj = u.trajectory

    traj[3]
    ag = u.select_atoms("all")
    proj_ag = tica.transform(ag)
    assert np.allclose(proj_ag, original_proj)
    assert traj.frame == 3

    traj[5]
    proj_u = tica.transform(u)
    assert np.allclose(proj_u, original_proj)
    assert traj.frame == 5
    assert proj_u.shape == (u.trajectory.n_frames, tica.components_.shape[1])

    proj_arr = tica.transform(tica.X)
    assert np.allclose(proj_arr, original_proj)
