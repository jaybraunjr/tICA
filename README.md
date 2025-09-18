TICA (MDAnalysis-based)
=========================================

This repo contains a compact, fast implementation of Time‑lagged Independent Component Analysis (tICA) on top of MDAnalysis.

Contents
- mdakit/tica.py — tICA implementation as an MDAnalysis AnalysisBase
- mdakit/plotting.py — plotting helpers (heatmap + marginals, spectra, etc.)


Math Overview (tICA)

tICA takes a trajectory matrix $X \in \mathbb{R}^{T \times D}$, forms two lagged blocks $X_0 = X_{0:T-\tau}$ and $X_\tau = X_{\tau:T}$, centers (and optionally scales) them, and estimates the instantaneous and time-lagged covariances

$$
C_0 = \langle X_0^c X_0^{c\top} \rangle, \qquad C_\tau = \langle X_0^c X_\tau^{c\top} \rangle.
$$

We regularize $C_0$ (add $\varepsilon I$ if requested) and solve the generalized eigenproblem $C_\tau v = \lambda C_0 v$. The eigenvectors form the tICA components, the projections are $Y = (X - \mu) C$, and optional timescales follow from $\tau_i = -\tau \Delta t / \ln \lambda_i$ for $0 < \lambda_i < 1$.


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

- The default heatmap colormap mirrors PyEMMA's `nipy_spectral` free-energy palette; pass `cmap="viridis"` (or any Matplotlib colormap) if you prefer another look.

- Scree plot and timescales:

    from mdakit.plotting import eigen_spectrum, timescales_bar
    eigen_spectrum(tica.eigenvalues_, title="tICA Eigenvalues")[0].savefig("tica_eigs.png")
    if tica.timescales_ is not None:
        timescales_bar(tica.timescales_, title="tICA Timescales")[0].savefig("tica_timescales.png")


Feature Engineering
- Build collective-variable matrices with the featurizer:

    from mdakit.featurizer import Featurizer, rmsd_feat, distance_feat

    feats = (
        Featurizer(u)
        .add("rmsd", rmsd_feat(select="name CA"))
        .add("dist", distance_feat("resid 1 and name CA", "resid 20 and name CA"))
    )
    X = feats.array()  # shape (n_frames, 2)
    tica = TICA(u, lag=20, n_components=2).fit_from_array(X, lag=20, scale=True)

- Additional helpers: `gyration_feat`, `contact_count_feat`, `hbond_count_feat`,
  `dihedral_feat`, `torsion_avg_feat`, `secondary_structure_frac_feat`,
  `sasa_feat`, and `plane_distance_feat` for common CVs.


Tips & Best Practices
- Alignment: for Cartesian coordinates, rigid‑body alignment often improves interpretability (removes translation/rotation). Either use `align=True` in TICA or pre-align with MDAnalysis (AlignTraj or transformations pipeline). For internal coordinates (distances/dihedrals), alignment is unnecessary.
- Scaling: enable `scale=True` to z-score features when needed. This can help when units or scales differ among coordinates.
- Regularization: small diagonal loading (e.g., `1e-5`) can stabilize whitening for near-singular C₀.
- Reversible estimator: set `reversible=True` to symmetrize estimators, often preferable for equilibrium MD.

Examples & Tests
- Usage demos and benchmarking scripts now live in `examples/` (they expect the sample trajectory files referenced above).
- Unit tests run on synthetic data; execute `pytest` from the repository root to validate core math and IO helpers.
- Rendered docs (after `python3 -m sphinx -b html docs docs/_build/html`) are available via [`docs/_build/html/index.html`](docs/_build/html/index.html).
