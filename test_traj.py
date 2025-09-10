import MDAnalysis as mda
import numpy as np
from mdakit.tica import TICA   
import matplotlib
matplotlib.use("Agg")   # use a non-GUI backend
import matplotlib.pyplot as plt




u = mda.Universe("../../mlx.gro", "../../rep1_skip10.xtc")


# Use a small regularization to stabilize whitening; MDAnalysis units are Å
tica = TICA(
    u,
    select="name CA",
    lag=20,
    n_components=3,
    regularization=1e-5,
).run(step=1)


proj = tica.proj_

# Scatter plot
plt.figure(figsize=(6,5))
plt.scatter(proj[:,0], proj[:,1], s=10, alpha=0.5)
plt.xlabel("TIC 1")
plt.ylabel("TIC 2")
plt.title("Projection onto first two TICA components")
plt.tight_layout()
plt.savefig("tica_proj.png")
plt.close()

# Eigenvalues
plt.figure()
plt.plot(tica.eigenvalues_, "o-")
plt.xlabel("Component")
plt.ylabel("Eigenvalue")
plt.title("TICA eigenvalues")
plt.savefig("tica_eigenvalues.png")
plt.close()

print("X shape:", tica.X.shape)
print("C0 shape:", tica.C0.shape, "Cτ shape:", tica.Ctau.shape)
print("C0 (first 3x3):\n", tica.C0[:3,:3])
print("Cτ (first 3x3):\n", tica.Ctau[:3,:3])

print("Eigenvalues of C0 (first 10):", tica.C0_eigvals[:10])
print("Whitened M (first 3x3):\n", tica.M[:3,:3])
print("Eigenvalues of M (all):", tica.M_eigvals)

