User guide
==========

This guide walks through the most common ways to use the tica package, from
running tICA on a molecular dynamics trajectory to analysing engineered
collective variables (CVs).

Setup
-----

1. Install the project in editable mode::

       python3 -m pip install -e .

2. Generate example CVs using the demo script::

       python3 examples/trajectory_demo.py

   The script performs a trajectory-driven tICA run and writes engineered CVs to
   feature_outputs/.

Running tICA on a trajectory
----------------------------

The standard workflow pulls coordinates directly from an MDAnalysis.Universe
and lets the tica.tica.TICA class handle lagged block construction,
centering/scaling, whitening, projection, and diagnostics::

    import MDAnalysis as mda
    from tica.tica import TICA

    u = mda.Universe("input.gro", "traj.xtc")
    model = TICA(u, select="name CA", lag=20, n_components=3).run()

    projections = model.proj_
    components = model.components_
    eigenvalues = model.eigenvalues_
    timescales = model.timescales_

Projecting new data
-------------------

After fitting, you can project other trajectories or NumPy arrays::

    other = mda.Universe("other.gro", "other.xtc")
    other_ic = model.transform(other, n_components=2)

    import numpy as np
    features = np.load("feature_outputs/tica_features.npy")
    array_ic = model.transform(features)

Engineered features
-------------------

The tica.featurizer module builds CV matrices from an MDAnalysis
Universe. Example::

    from tica.featurizer import (
        Featurizer,
        rmsd_feat,
        distance_feat,
        torsion_avg_feat,
    )

    feats = (
        Featurizer(u)
        .add("rmsd", rmsd_feat("name CA"))
        .add("dist", distance_feat("resid 1", "resid 20", mass_weighted=False))
        .add("phi", torsion_avg_feat("protein", angle="phi"))
    )
    X = feats.array()

    engineered_model = TICA(u, lag=20, n_components=2)
    engineered_model.fit_from_array(X, scale=True)

Per-feature tICA
----------------

To diagnose individual CVs, run a two-dimensional tICA on each feature (feature
value and its lagged partner). examples/trajectory_demo.py already produces
such plots under feature_outputs/tica_per_feature_ic2/. The essential steps
are::

    import numpy as np
    from pathlib import Path
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader
    from tica.tica import TICA

    X = np.load("feature_outputs/tica_features.npy")
    lag = 20

    dummy = mda.Universe.empty(1, trajectory=True)
    dummy.trajectory = MemoryReader(np.zeros((X.shape[0] - lag, 1, 3), dtype=np.float32))

    for i, name in enumerate(["example"]):
        Xi = np.column_stack([X[:-lag, i], X[lag:, i]])
        model = TICA(dummy, lag=lag, n_components=2)
        model.fit_from_array(Xi, scale=True)
        np.savetxt(f"feature_outputs/tica_per_feature_ic2/tica_{name}.csv", model.proj_, delimiter=",")

Visualisation
-------------

examples/trajectory_demo.py writes IC1 vs IC2 plots for the combined model
and each engineered CV. The tica_cv_analysis.ipynb notebook (created in the
repository root) offers an interactive way to explore the saved CVs and rerun
projections.

Next steps
----------

* Integrate tICA components with clustering or Markov state modelling.
* Experiment with alternate lag times and reversible vs. non-reversible
  estimators by adjusting TICA.fit_from_array arguments.
* Extend the featurizer with custom CV definitions tailored to your system.
