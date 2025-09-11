TICA (MDAnalysis-based)
=========================================

This repo contains a compact, fast implementation of Time‑lagged Independent Component Analysis (tICA) on top of MDAnalysis.

Contents
- mdakit/tica.py — tICA implementation as an MDAnalysis AnalysisBase
- mdakit/plotting.py — plotting helpers (heatmap + marginals, spectra, etc.)


Math Overview (tICA)
- Data and lagged blocks:

  $$
  X \in \mathbb{R}^{T\times D},\quad X_0 = X_{0:T-\tau},\quad X_\tau = X_{\tau:T}
  $$

  After centering (and optional z‑scoring) with the same mean/std for both blocks we obtain $X_0^c$ and $X_\tau^c$.

- Covariances (biased normalization by default):

  Non‑reversible
  $$
  C_0 = \frac{X_0^{c\,\top} X_0^c}{T-\tau},\qquad
  C_\tau = \frac{X_0^{c\,\top} X_\tau^c}{T-\tau}
  $$

  Reversible (symmetrized)
  $$
  C_0 = \frac{1}{2(T-\tau)}\big(X_0^{c\,\top} X_0^c + X_\tau^{c\,\top} X_\tau^c\big),\qquad
  C_\tau = \frac{1}{2(T-\tau)}\big(X_0^{c\,\top} X_\tau^c + X_\tau^{c\,\top} X_0^c\big)
  $$

- Generalized eigenproblem and regularization:

  $$
  C_\tau\, v = \lambda\, C_0\, v,\qquad C_0 \leftarrow C_0 + \varepsilon I
  $$

- Whitening and eigensolve:

  Cholesky (preferred)
  $$
  C_0 = L L^\top,\qquad M = L^{-1} C_\tau L^{-\top},\qquad M y = \lambda y,\qquad v = L^{-\top} y
  $$

  Eigen‑whitening fallback (with $S=\mathrm{diag}(s)$)
  $$
  C_0 = U\, \mathrm{diag}(s)\, U^\top,\quad M = S^{-1/2} U^\top C_\tau U S^{-1/2},\quad v = U S^{-1/2} y
  $$

- Projection of full data:

  $$
  Y = (X - \mu)\, C\quad (\text{or } Y = ((X-\mu) / \sigma)\, C \text{ if scaling})
  $$

- Implied timescales (if frame time $\Delta t$ is known):

  $$
  t_i = -\frac{\tau\, \Delta t}{\ln \lambda_i},\qquad 0 < \lambda_i < 1
  $$


Installation/Requirements
- Python 3.8+
- MDAnalysis
- NumPy, Matplotlib (for plotting)

Quick Start (tICA)
1) Load your trajectory with MDAnalysis and run tICA:

    import MDAnalysis as mda
    from mdakit.tica import TICA

    u = mda.Universe("input.gro", "traj.xtc")
    tica = TICA(
        u,
        select="name CA",   # atom selection
        lag=20,              # frames
        n_components=3,
        regularization=1e-5,
        scale=False,         # z‑score features (optional)
        reversible=True,
        dt=None,             # set to frame time to get timescales
        align=False          # optional per‑frame alignment to reference
    ).run()

    # Results
    Y = tica.proj_            # (T, n_components) projection (ICs)
    C = tica.components_      # (D, n_components) components
    lam = tica.eigenvalues_   # eigenvalues
    ts = tica.timescales_     # implied timescales (if dt was provided)

2) Project another trajectory or an array into the same tICA space:

    # From an AtomGroup/Universe (uses same selection, optional alignment)
    Y2 = tica.transform(u, n_components=3)

    # From a prebuilt array X2 of shape (T2, D)
    Y2 = tica.transform(X2)

Plotting
- Heatmap + marginals of IC1 vs IC2:

    from mdakit.plotting import heat_hist2d_from_proj
    fig, *_ = heat_hist2d_from_proj(tica.proj_, i=0, j=1, bins=150, title="tICA: IC1 vs IC2")
    fig.savefig("tica_ic12_heat.png", dpi=150)

- Scree plot and timescales:

    from mdakit.plotting import eigen_spectrum, timescales_bar
    eigen_spectrum(tica.eigenvalues_, title="tICA Eigenvalues")[0].savefig("tica_eigs.png")
    if tica.timescales_ is not None:
        timescales_bar(tica.timescales_, title="tICA Timescales")[0].savefig("tica_timescales.png")


Tips & Best Practices
- Alignment: for Cartesian coordinates, rigid‑body alignment often improves interpretability (removes translation/rotation). Either use `align=True` in TICA or pre‑align with MDAnalysis (AlignTraj or transformations pipeline). For internal coordinates (distances/dihedrals), alignment is unnecessary.
- Scaling: enable `scale=True` to z‑score features when needed. This can help when units or scales differ among coordinates.
- Regularization: small diagonal loading (e.g., `1e-5`) can stabilize whitening for near‑singular C₀.
- Reversible estimator: set `reversible=True` to symmetrize estimators, often preferable for equilibrium MD.
