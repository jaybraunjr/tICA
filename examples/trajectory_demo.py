import MDAnalysis as mda
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from mdakit.featurizer import (
    Featurizer,
    contact_count_feat,
    dihedral_feat,
    distance_feat,
    gyration_feat,
    plane_distance_feat,
    secondary_structure_frac_feat,
    rmsd_feat,
    torsion_avg_feat,
)
from mdakit.tica import TICA


u = mda.Universe("../../mlx.gro", "../../rep1_skip10.xtc")

# --- tICA on Cartesian coordinates (align=False by default) --- #
tica = TICA(
    u,
    select="name CA",
    lag=50,
    n_components=3,
    regularization=1e-5,
).run(step=1)

proj = tica.proj_

plt.figure(figsize=(6, 5))
plt.scatter(proj[:, 0], proj[:, 1], s=10, alpha=0.5)
plt.xlabel("TIC 1")
plt.ylabel("TIC 2")
plt.title("Projection onto first two TICA components")
plt.tight_layout()
plt.savefig("tica_proj.png")
plt.close()

plt.figure()
plt.plot(tica.eigenvalues_, "o-")
plt.xlabel("Component")
plt.ylabel("Eigenvalue")
plt.title("TICA eigenvalues")
plt.savefig("tica_eigenvalues.png")
plt.close()

print("X shape:", tica.X.shape)
print("C0 shape:", tica.C0.shape, "Cτ shape:", tica.Ctau.shape)
print("Eigenvalues of M (all):", tica.M_eigvals)


# --- Build CVs with the featurizer and fit tICA on them --- #
feats = (
    Featurizer(u)
    .add("rmsd_ca", rmsd_feat(select="name CA"))
    .add(
        "end_to_end",
        distance_feat("resid 1 and name CA", "resid 20 and name CA", mass_weighted=False),
    )
    .add(
        "protein_lipid_sep",
        distance_feat("protein and name CA", "resname POPC and name P"),
    )
    .add("rg_ca", gyration_feat("name CA"))
    .add(
        "lipid_plane_dist",
        plane_distance_feat("resname POPC and name P", axis=2, plane_position=0.0),
    )
    .add(
        "contacts",
        contact_count_feat(
            "protein and name CA",
            "resname POPC and name P",
            cutoff=6.0,
            pbc=True,
        ),
    )
    .add(
        "phi_avg",
        torsion_avg_feat(select="protein", angle="phi", deg=True),
    )
    .add(
        "psi_avg",
        torsion_avg_feat(select="protein", angle="psi", deg=True),
    )
    .add(
        "helix_fraction",
        secondary_structure_frac_feat(select="protein", kind="helix"),
    )
    .add(
        "phi12",
        dihedral_feat(
            "resid 1 and name C",
            "resid 2 and name N",
            "resid 2 and name CA",
            "resid 2 and name C",
        ),
    )
)

X = feats.array()
names = feats.names()
print("Feature matrix shape:", X.shape, "names:", names)

out_dir = Path("feature_outputs")
out_dir.mkdir(exist_ok=True)

# Persist raw collective variables for downstream analysis
np.save(out_dir / "tica_features.npy", X)
np.savetxt(
    out_dir / "tica_features.csv",
    X,
    delimiter=",",
    header=",".join(names),
    comments="",
)
print(f"Saved feature matrix to {out_dir}/tica_features.npy and .csv")

for idx, name in enumerate(names):
    column = X[:, idx]
    np.save(out_dir / f"feature_{name}.npy", column)
    np.savetxt(
        out_dir / f"feature_{name}.csv",
        column,
        delimiter=",",
        header=name,
        comments="",
    )
print(f"Saved individual feature series in {out_dir}/feature_<name>.*")

tica_feats = TICA(u, lag=20, n_components=2)
tica_feats.fit_from_array(X, lag=20, scale=True, n_components=2)

plt.figure(figsize=(6, 5))
plt.scatter(tica_feats.proj_[:, 0], tica_feats.proj_[:, 1], s=8, alpha=0.5)
plt.xlabel("IC 1")
plt.ylabel("IC 2")
plt.title("tICA on engineered CVs")
plt.tight_layout()
plt.savefig("tica_proj_cv.png")
plt.close()

plt.figure()
plt.plot(tica_feats.eigenvalues_, "o-")
plt.xlabel("Component")
plt.ylabel("Eigenvalue")
plt.title("tICA eigenvalues (CVs)")
plt.savefig("tica_eigenvalues_cv.png")
plt.close()

print("CV tICA eigenvalues:", tica_feats.eigenvalues_)
