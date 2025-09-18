Quick start
===========

Install the package in editable mode and run the demo workflow:

.. code-block:: bash

   python3 -m pip install -e .
   python3 examples/trajectory_demo.py

The script generates collective-variable (CV) time series under
feature_outputs/ along with pre-made tICA plots. See the notebook
tica_cv_analysis.ipynb (created in the repository root) for an
interactive walkthrough that loads those CVs, plots them, and runs tICA
on demand.

Key modules
-----------

mdakit.tica
    Implementation of the MDAnalysis-based tICA engine (TICA class).

mdakit.featurizer
    Helpers for building CV pipelines: RMSD, distances, contact counts,
    torsion averages, secondary-structure fractions, SASA, etc.

mdakit.plotting
    Convenience plots for tICA projections, eigen spectra, and derived statistics.
