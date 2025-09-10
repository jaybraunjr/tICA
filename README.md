TICA and PCA Utilities (MDAnalysis-based)
=========================================

This repo contains a compact, fast implementation of Time‑lagged Independent Component Analysis (tICA) on top of MDAnalysis.

Contents
- mdakit/tica.py — tICA implementation as an MDAnalysis AnalysisBase
- mdakit/plotting.py — plotting helpers (heatmap + marginals, spectra, etc.)


Math Overview (tICA)
- Data: stacked Cartesian coordinates X ∈ R^{T×D}, with frames t = 0..T−1 and feature dimension D (e.g., 3×n_atoms after selection).
- Lagged blocks: for lag τ ≥ 1, define X₀ = X[0:T−τ], X_τ = X[τ:T]. After centering (and optional z‑scoring) with the same mean/std for both blocks we obtain X₀ᶜ and X_τᶜ.
- Covariances (biased normalization by default):
  - Non‑reversible:
    - C₀ = (X₀ᶜᵀ X₀ᶜ) / (T−τ)
    - C_τ = (X₀ᶜᵀ X_τᶜ) / (T−τ)
  - Reversible (symmetrized):
    - C₀ = 0.5[(X₀ᶜᵀ X₀ᶜ) + (X_τᶜᵀ X_τᶜ)] / (T−τ)
    - C_τ = 0.5[(X₀ᶜᵀ X_τᶜ) + (X_τᶜᵀ X₀ᶜ)] / (T−τ)
- Generalized eigenproblem: C_τ v = λ C₀ v. We regularize C₀ → C₀+εI and solve via whitening.
- Whitening and eigensolve:
  - Cholesky if possible: C₀ = LLᵀ, M = L^{-1} C_τ L^{-T}, solve M y = λ y, back‑transform v = L^{-T} y.
  - Fallback: eig‑whitening C₀ = U diag(s) Uᵀ, drop small s, M = s^{-1/2} Uᵀ C_τ U s^{-1/2}, back‑transform v = U s^{-1/2} y.
- Sort λ descending and (optionally) truncate to the top n components.
- Projection of full data: Y = (X−μ) C (or z‑scored if scaling), where columns of C are the learned components.
- Implied timescales (if frame time Δt is known): τ_i = −(lag·Δt) / ln(λ_i) for 0 < λ_i < 1; otherwise ∞.


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

PCA (MDAnalysis) on the same trajectory

    from MDAnalysis.analysis.pca import PCA
    p = PCA(u, select="name CA", n_components=3).run()
    pc = p.results.p_components     # (3N, 3)
    proj = p.transform(u.select_atoms("name CA"), n_components=3)

Benchmarks
- Run the timing script on the same CA‑only trajectory used above:

    python3 test_perf.py

You’ll see end‑to‑end timings for:
- TICA (this implementation)
- PCA (MDAnalysis)
- tICA (PyEMMA) — requires mdtraj and pyemma; the script includes a small NumPy compatibility shim for older PyEMMA releases.

Tips & Best Practices
- Alignment: for Cartesian coordinates, rigid‑body alignment often improves interpretability (removes translation/rotation). Either use `align=True` in TICA or pre‑align with MDAnalysis (AlignTraj or transformations pipeline). For internal coordinates (distances/dihedrals), alignment is unnecessary.
- Scaling: enable `scale=True` to z‑score features when needed. This can help when units or scales differ among coordinates.
- Regularization: small diagonal loading (e.g., `1e-5`) can stabilize whitening for near‑singular C₀.
- Reversible estimator: set `reversible=True` to symmetrize estimators, often preferable for equilibrium MD.

