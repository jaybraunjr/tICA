import numpy as np
from collections import deque
from typing import Optional, Union

from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.align import _fit_to



class TICA(AnalysisBase):
    """Time-lagged Independent Component Analysis (tICA)."""

    def __init__(
        self,
        universe,
        select: str = "all",
        lag: int = 1,
        n_components: int = None,
        regularization: float = 0.0,
        scale: bool = False,
        reversible: bool = True,
        dt: float = None,
        align: bool=False,
        **kwargs,
    ):
        super().__init__(universe.trajectory, **kwargs)
        self._u = universe
        self._select = select
        self.lag = int(lag)
        if self.lag < 1:
            raise ValueError("lag must be >= 1")

        self._n_components = n_components
        self._regularization = float(regularization)
        self._scale = bool(scale)
        self._reversible = bool(reversible)
        self._dt = dt
        self.align = bool(align)


    def _prepare(self):
        self._atoms = self._u.select_atoms(self._select)
        D = self._atoms.n_atoms * 3
        self._X = np.empty((self.n_frames, D), dtype=np.float64)
        self._i = 0
        if getattr(self, 'align', False):
            ref = self._atoms
            self._ref_atom_positions = ref.positions.copy()
            self._ref_cog = ref.center_of_geometry()


    def _single_frame(self):
        if getattr(self, 'align', False):
            mobile_cog = self._atoms.center_of_geometry()
            mobile_atoms, _ = _fit_to(
                self._atoms.positions - mobile_cog,
                self._ref_atom_positions - self._ref_cog,
                self._atoms,
                mobile_com=mobile_cog,
                ref_com=self._ref_cog,
            )
            pos = mobile_atoms.positions
        else:
            pos = self._atoms.positions
        self._X[self._i, :] = pos.ravel()
        self._i += 1

    def _conclude(self):
        X, X0, Xtau, n_pairs, T, lag = self._build_lagged_blocks()
        X0c, Xtauc, mean, std = self._center_and_scale(X0, Xtau, scale=self._scale)
        C0, Ctau = self._estimate_covariance(
            X0c, Xtauc, n_pairs, reversible=self._reversible, bessel=False
        )
        C0_reg = self._regularize(C0, self._regularization)
        try:
            self.C0_eigvals = np.linalg.eigvalsh(C0_reg)
        except Exception:
            self.C0_eigvals = None
        M, lam, V = self._whiten_solve(C0_reg, Ctau, chol_fallback_tol=(1e-4, 1e-12))
        lam, V = self._sort_and_truncate(lam, V, self._n_components)
        self.X = X
        self.C0 = C0
        self.Ctau = Ctau
        self.M = M
        self.M_eigvals = lam
        self.eigenvalues_ = lam
        self.components_ = V
        self._mean_ = mean
        self._std_ = std
        self.proj_ = self._project_centered(X, mean, std, V)
        dt = self._dt if self._dt is not None else getattr(self._trajectory, "dt", None)
        self.timescales_ = self._compute_timescales(lam, lag, dt)

    def transform(self, data, n_components=None):
        """Project data into the learned tICA space.

        Parameters
        - data: ndarray (T, D) or MDAnalysis AtomGroup or Universe
            If ndarray, it must have the same feature dimension D used in fit.
            If AtomGroup/Universe, frames are iterated and the same selection
            used during fit is applied (for Universe) or validated (for AG).
            If ``self.align`` is True and a reference was set during fit, frames
            are aligned to that reference before projection.
        - n_components: int | None
            Number of components to return; defaults to the model's size.

        Returns
        - Y: ndarray (T, m)
            Projection onto the first m components.
        """
        if not hasattr(self, "components_") or not hasattr(self, "_mean_"):
            raise ValueError("TICA must be run() before calling transform")

        C = self.components_
        if n_components is not None:
            m = int(n_components)
            if m < 1:
                return np.empty((0, 0), dtype=np.float64)
            C = C[:, :m]

        if isinstance(data, np.ndarray):
            X = np.asarray(data, dtype=np.float64)
            if X.ndim != 2:
                raise ValueError("Input array must be 2D (T, D)")
            if X.shape[1] != self.components_.shape[0]:
                raise ValueError("Feature dimension mismatch for projection")
            return self._project_centered(X, self._mean_, self._std_, C)

        try:
            from MDAnalysis import Universe
            from MDAnalysis.core.groups import AtomGroup
        except Exception:
            Universe = None
            AtomGroup = None

        ag = None
        traj = None
        if Universe is not None and isinstance(data, Universe):
            ag = data.select_atoms(self._select)
            traj = data.trajectory
        elif AtomGroup is not None and isinstance(data, AtomGroup):
            ag = data
            traj = ag.universe.trajectory
        else:
            raise TypeError("data must be a numpy array, AtomGroup, or Universe")

        D = self.components_.shape[0]
        if ag.n_atoms * 3 != D:
            raise ValueError("AtomGroup size does not match fitted selection")
        T = len(traj)
        X = np.empty((T, D), dtype=np.float64)

        i = 0
        if self.align and hasattr(self, "_ref_atom_positions"):
            ref_pos = self._ref_atom_positions
            ref_cog = self._ref_cog
            for ts in traj:
                mobile_cog = ag.center_of_geometry()
                mobile_atoms, _ = _fit_to(
                    ag.positions - mobile_cog,
                    ref_pos - ref_cog,
                    ag,
                    mobile_com=mobile_cog,
                    ref_com=ref_cog,
                )
                X[i, :] = mobile_atoms.positions.ravel()
                i += 1
        else:
            for ts in traj:
                X[i, :] = ag.positions.ravel()
                i += 1

        return self._project_centered(X, self._mean_, self._std_, C)


    def _build_lagged_blocks(self, X=None, lag=None):
        """Build lagged data blocks X0 and Xtau from stacked coordinates.

        Parameters
        ----------
        X : array-like of shape (T, D), optional
            Stacked coordinates. If None, uses self._coords.
        lag : int, optional
            Lag in frames. If None, uses self.lag.

        Returns
        -------
        X : ndarray of shape (T, D)
        X0 : ndarray of shape (T - lag, D)
        Xtau : ndarray of shape (T - lag, D)
        n_pairs : int
            Number of lagged pairs (T - lag).
        T : int
            Number of frames.
        lag : int
            Resolved lag value.
        """
        if X is None:
            if hasattr(self, "_X") and self._i == self.n_frames:
                X = self._X
            else:
                # Fallback to any collected coords list
                X = np.asarray(getattr(self, "_coords", []), dtype=np.float64)
        else:
            X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (T, D)")

        lag = self.lag if lag is None else int(lag)
        if lag < 1:
            raise ValueError("lag must be >= 1")

        T = X.shape[0]
        if T <= lag:
            raise ValueError(f"Not enough frames (T={T}) for lag={lag}")

        X0 = X[:-lag, :]
        Xtau = X[lag:, :]
        n_pairs = T - lag

        return X, X0, Xtau, n_pairs, T, lag
    


    def _center_and_scale(self, X0, Xtau, scale, mean=None, std=None):
        """Center (and optionally z‑score) lagged blocks.

        Parameters (expected when you implement this)
        - X0: ndarray, shape (T - lag, D)
        - Xtau: ndarray, shape (T - lag, D)
        - scale: bool, optional (default: self._scale)
        - mean: ndarray (D,), optional
        - std: ndarray (D,), optional

        Returns (what this function should return)
        - X0c: ndarray, shape (T - lag, D)
        - Xtauc: ndarray, shape (T - lag, D)
        - mean: ndarray (D,)
        - std_or_none: ndarray (D,) or None
            The std used when `scale=True`, else None.

        """
        X0 = np.asarray(X0, dtype=np.float64)
        Xtau = np.asarray(Xtau, dtype=np.float64)
        use_scale = self._scale if scale is None else bool(scale)
        need_stack = (mean is None) or (use_scale and std is None)
        if need_stack:
            X = np.vstack([X0, Xtau])
        if mean is None:
            mean = X.mean(axis=0)

        if use_scale:
            if std is None:
                std = X.std(axis=0, ddof=1)
            std = np.asarray(std, dtype=np.float64)
            std[std == 0] = 1.0
            X0c = (X0 - mean) / std
            Xtauc = (Xtau - mean) / std
            std_out = std
        else:
            X0c = X0 - mean
            Xtauc = Xtau - mean
            std_out = None

        self._mean_ = mean
        self._std_ = std_out

        return X0c, Xtauc, mean, std_out


    def _estimate_covariance(self, X0c, Xtauc, n_pairs, reversible, bessel):
          """Compute instantaneous and lagged covariances from centered blocks.

          Parameters
          ----------
          X0c : ndarray, shape (n_pairs, D)
              Centered (or z-scored) pre-lag block.
          Xtauc : ndarray, shape (n_pairs, D)
              Centered (or z-scored) lagged block, normalized with the same
              mean/std as ``X0c``.

          Returns
          -------
          C0 : ndarray, shape (D, D)
              Instantaneous covariance matrix (symmetric; PSD up to numerical noise).
          Ctau : ndarray, shape (D, D)
              Lagged covariance matrix (symmetric if reversible estimator is used).

          Notes
          -----
          - Uses biased normalization by default (divide by ``n_pairs``) unless
            ``bessel=True``.
          - With a reversible estimator, covariances are symmetrized using both
            X0 and Xtau blocks; otherwise a one-sided estimate is used.
          - Inputs must be mean-free (and optionally scaled) using the same
            global mean/std for both blocks. See ``_center_and_scale``.
          """

          X0c = np.asarray(X0c, dtype=np.float64)
          Xtauc = np.asarray(Xtauc, dtype=np.float64)
          denom = int(n_pairs) - (1 if bessel else 0)
          if denom <= 0:
              raise ValueError("Invalid denominator for covariance normalization")

          if reversible:
              C0 = (X0c.T @ X0c + Xtauc.T @ Xtauc) * (0.5 / denom)
              Ctau = (X0c.T @ Xtauc + Xtauc.T @ X0c) * (0.5 / denom)
          else:
              C0 = (X0c.T @ X0c) / denom
              Ctau = (X0c.T @ Xtauc) / denom

          C0 = 0.5 * (C0 + C0.T)
          Ctau = 0.5 * (Ctau + Ctau.T)

          return C0, Ctau

    def _regularize(self, C0, eps):
        """Add diagonal regularization and ensure symmetry.

        Implement with signature (C0, eps)->C0_reg

        Parameters
        - C0: ndarray (D, D)
            Instantaneous covariance matrix (symmetric, PSD up to noise).
        - eps: float
            Non‑negative regularization strength for Tikhonov/diagonal loading.

        Returns
        - C0_reg: ndarray (D, D)
            Regularized, explicitly symmetrized covariance: C0 + eps*I.

        Notes
        - Ensure exact symmetry via 0.5*(A + A.T) before/after loading.
        - Do not modify input array in place.
        """
        C0_sym = 0.5 * (C0 + C0.T)
        C0_reg = C0_sym + (eps * np.eye(C0.shape[0]) if eps > 0 else 0)
        # Ensure exact symmetry on return
        return 0.5 * (C0_reg + C0_reg.T)
        

    def _whiten_solve(self, C0_reg, Ctau, chol_fallback_tol):
        """Solve the generalized eigenproblem via whitening.

        Implement with signature (C0_reg, Ctau, chol_fallback_tol)->(M, eigvals, eigvecs)

        Parameters
        - C0_reg: ndarray (D, D)
            Regularized covariance used for whitening.
        - Ctau: ndarray (D, D)
            Lagged covariance matrix.
        - chol_fallback_tol: tuple[float, float]
            (rel_tol, abs_tol) for eigen‑based whitening when Cholesky fails.

        Returns
        - M: ndarray (D, D) or (d_kept, d_kept)
            Symmetric whitened operator L^{-1} Ctau L^{-T} (or reduced form).
        - eigvals: ndarray (k,)
            Eigenvalues sorted descending.
        - eigvecs: ndarray (D, k)
            Right eigenvectors in original coordinates (back‑transformed).

        Notes
        - Prefer Cholesky for speed/stability; fall back to eig‑whitening with
          small‑eigenvalue filtering using provided tolerances.
        """
        C0_reg = 0.5 * (C0_reg + C0_reg.T)
        Ctau = 0.5 * (Ctau + Ctau.T)

        try:
            L = np.linalg.cholesky(C0_reg)
            Y = np.linalg.solve(L, Ctau)        # Y = L^{-1} Ctau
            M = np.linalg.solve(L, Y.T).T       # M = Y L^{-T}
            M = 0.5 * (M + M.T)
            lam, Q = np.linalg.eigh(M)
            V = np.linalg.solve(L.T, Q)
        except np.linalg.LinAlgError:
            s, U = np.linalg.eigh(C0_reg)

            if chol_fallback_tol is None:
                rel_tol, abs_tol = 1e-4, 1e-12
            else:
                rel_tol, abs_tol = chol_fallback_tol
            thresh = max(abs_tol, rel_tol * float(s.max()))
            keep = s > thresh
            s_kept = s[keep]
            U_kept = U[:, keep]
            if s_kept.size == 0:
                raise RuntimeError(
                    "C0 is singular even after regularization; increase regularization or check data"
                )
            S_inv_sqrt = np.diag(1.0 / np.sqrt(s_kept))
            M = S_inv_sqrt @ U_kept.T @ Ctau @ U_kept @ S_inv_sqrt
            M = 0.5 * (M + M.T)
            lam, E = np.linalg.eigh(M)
            V = U_kept @ S_inv_sqrt @ E

        idx = np.argsort(lam)[::-1]
        lam = lam[idx]
        V = V[:, idx]

        # Default to full set
        eigvals, eigvecs = lam, V
        if self._n_components is not None:
            eigvals = lam[: self._n_components]
            eigvecs = V[:, : self._n_components]

        return M, eigvals, eigvecs
        


    def _sort_and_truncate(self, eigvals, eigvecs, n_components):
        """Sort eigenpairs and optionally truncate to target dimensionality.

        Implement with signature (eigvals, eigvecs, n_components)->(vals, vecs)

        Parameters
        - eigvals: ndarray (k,)
            Unsorted eigenvalues.
        - eigvecs: ndarray (D, k)
            Corresponding eigenvectors (columns).
        - n_components: int | None
            If given, keep the top n components after sorting.

        Returns
        - vals: ndarray (m,)
        - vecs: ndarray (D, m)
            Sorted (descending by eigenvalue) and possibly truncated.
        """
        eigvals = np.asanyarray(eigvals)
        eigvecs = np.asanyarray(eigvecs)

        idx = np.argsort(eigvals)[::-1]
        vals = eigvals[idx]
        vecs = eigvecs[:, idx]

        if n_components is not None:
            n = int(n_components)
            if n < 1:
                return vals[:0], vecs[:, :0]
            vals = vals[:n]
            vecs = vecs[:, :n]

        return vals, vecs
 

    def _compute_timescales(self, eigvals, lag, dt):
        """Convert eigenvalues to implied timescales.

        Implement with signature (eigvals, lag, dt)->timescales

        Parameters
        - eigvals: ndarray (k,)
            tICA eigenvalues in (0, 1]. Values outside (0,1) map to inf.
        - lag: int
            Lag in frames used for the estimator.
        - dt: float | None
            Time per frame. If None, return None.

        Returns
        - timescales: ndarray (k,) | None
            tau_i = -(lag*dt)/log(lambda_i) for 0 < lambda_i < 1; else inf.
        """
        if dt is None:
            return None
        vals = np.asarray(eigvals, dtype=np.float64)
        tau = np.full(vals.shape, np.inf, dtype=np.float64)
        mask = (vals > 0.0) & (vals < 1.0)
        if np.any(mask):
            tau[mask] = -(float(lag) * float(dt)) / np.log(vals[mask])
        return tau

    def _project_centered(self, X, mean, std, components):
        """Project centered (and optionally scaled) data onto components.

        Implement with signature (X, mean, std, components)->Y

        Parameters
        - X: ndarray (T, D)
            Full stacked coordinates.
        - mean: ndarray (D,)
            Mean used for centering.
        - std: ndarray (D,) | None
            Standard deviation for scaling; if None, skip scaling.
        - components: ndarray (D, k)
            Projection matrix (tICA components).

        Returns
        - Y: ndarray (T, k)
            Projected trajectory in tICA space using the same preprocessing.
        """
        X = np.asarray(X, dtype=np.float64)
        mean = np.asarray(mean, dtype=np.float64)
        C = np.asarray(components, dtype=np.float64)
        if std is not None:
            std = np.asarray(std, dtype=np.float64)
            Z = (X - mean) / std
        else:
            Z = X - mean
        return Z @ C
