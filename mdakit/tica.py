import numpy as np
from collections import deque
from typing import Optional, Union

from MDAnalysis.analysis.base import AnalysisBase



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


    def _prepare(self):
        # select atoms
        self._atoms = self._u.select_atoms(self._select)
        # print(f"DEBUG: selection '{self._select}' → {self._atoms.n_atoms} atoms")
        self._coords = []

    def _single_frame(self):
        # collect positions
        pos = self._atoms.positions
        # print(f"DEBUG: frame {self._ts.frame}, positions shape {pos.shape}")
        self._coords.append(pos.copy().ravel())

    def _conclude(self):
        # turn coords into numpy array
        X = np.array(self._coords)
        lag = self.lag
        T = X.shape[0]
        if T <= lag:
            raise ValueError(f"Not enough frames ({T}) for lag={lag}")

        # build lagged blocks
        X0 = X[:-lag]
        Xtau = X[lag:]

        # Centering/scaling to match estimator choice
        mean = X.mean(0)
        if self._scale:
            std = X.std(axis=0, ddof=1)
            std[std == 0] = 1.0
            X0c = (X0 - mean) / std
            Xtauc = (Xtau - mean) / std
            self._std_ = std
        else:
            X0c = X0 - mean
            Xtauc = Xtau - mean
            self._std_ = None

        self._mean_ = mean


        # covariance estimators
        n_pairs = T - lag
        if self._reversible:
            # Symmetric (reversible) estimators like PyEMMA's LaggedCovariance
            C0 = 0.5 * ((X0c.T @ X0c) + (Xtauc.T @ Xtauc)) / n_pairs
            Ctau = 0.5 * ((X0c.T @ Xtauc) + (Xtauc.T @ X0c)) / n_pairs
        else:
            C0 = (X0c.T @ X0c) / n_pairs
            Ctau = (X0c.T @ Xtauc) / n_pairs

        # symmetrize
        C0 = 0.5 * (C0 + C0.T)
        Ctau = 0.5 * (Ctau + Ctau.T)

        # regularization
        eps = max(self._regularization, 0.0)
        C0_reg = C0 + (eps * np.eye(C0.shape[0]) if eps > 0 else 0)

        # Try Cholesky-based whitening for numerical stability
        try:
            L = np.linalg.cholesky(C0_reg)
            # M = L^{-1} Ctau L^{-T}
            Y = np.linalg.solve(L, Ctau)        # Y = L^{-1} Ctau
            M = np.linalg.solve(L, Y.T).T       # M = Y L^{-T}
            M = 0.5 * (M + M.T)
            lam, Q = np.linalg.eigh(M)
            # Back-transform eigenvectors: v = L^{-T} y
            V = np.linalg.solve(L.T, Q)
        except np.linalg.LinAlgError:
            # Fallback: eigen-based whitening with relative cutoff
            s, U = np.linalg.eigh(C0_reg)
            # filter out small eigenvalues with relative tolerance
            rel_tol = 1e-4
            abs_tol = 1e-12
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

        # sort
        idx = np.argsort(lam)[::-1]
        lam = lam[idx]
        V = V[:, idx]

        if self._n_components is not None:
            lam = lam[: self._n_components]
            V = V[:, : self._n_components]

        # save everything for debugging
        self.X = X
        self.C0 = C0
        self.Ctau = Ctau
        # Report eigenvalues of regularized C0 for diagnostics
        try:
            self.C0_eigvals = np.linalg.eigvalsh(C0 + (eps * np.eye(C0.shape[0]) if eps > 0 else 0))
        except Exception:
            self.C0_eigvals = None
        self.M = M
        self.M_eigvals = lam

        # final results
        self.eigenvalues_ = lam
        self.components_ = V
        # IMPORTANT: use same centering/scaling for projection
        if self._std_ is not None:
            self.proj_ = ((X - mean) / self._std_) @ V
        else:
            self.proj_ = (X - mean) @ V

        # timescales
        dt = self._dt if self._dt is not None else getattr(self._trajectory, "dt", None)
        if dt is not None:
            mask = (lam > 0) & (lam < 1)
            timescales = np.full(lam.shape, np.inf)
            timescales[mask] = -(lag * dt) / np.log(lam[mask])
            self.timescales_ = timescales
        else:
            self.timescales_ = None


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
            X = np.asarray(self._coords, dtype=np.float64)
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
        # ensure float64 for stability
        X0 = np.asarray(X0, dtype=np.float64)
        Xtau = np.asarray(Xtau, dtype=np.float64)
        use_scale = self._scale if scale is None else bool(scale)

        # stack if we need to compute stats
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
  
        C0_reg = C0 + (eps * np.eye(C0.shape[0]) if eps > 0 else 0)

        return C0_reg
        

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
        pass

    def _sort_and_truncate():
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
        pass

    def _compute_timescales():
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
        pass

    def _project_centered():
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
        pass
