import mdtraj as md
import pyemma
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load the trajectory with MDTraj (PyEMMA needs mdtraj-style input)
traj = md.load_xtc("../../rep1_skip10.xtc", top="../../mlx.gro")
# MDAnalysis uses Å by default while MDTraj uses nm; scale to Å for comparison
traj.xyz *= 10.0

# Select Cα atoms
ca_indices = traj.topology.select("name CA")
X = traj.atom_slice(ca_indices).xyz.reshape(traj.n_frames, -1)

# Run TICA with PyEMMA
tica_obj = pyemma.coordinates.tica(data=X, lag=20, dim=3)

# Project trajectory onto TICA space
Y = tica_obj.get_output()[0]   # (n_frames, n_components)

# Scatter plot
plt.figure(figsize=(6,5))
plt.scatter(Y[:,0], Y[:,1], s=10, alpha=0.5)
plt.xlabel("TIC 1")
plt.ylabel("TIC 2")
plt.title("PyEMMA: Projection onto first two TICA components")
plt.tight_layout()
plt.savefig("pyemma_tica_proj.png")
plt.close()

# Eigenvalues
plt.figure()
plt.plot(tica_obj.eigenvalues, "o-")
plt.xlabel("Component")
plt.ylabel("Eigenvalue")
plt.title("PyEMMA TICA eigenvalues")
plt.savefig("pyemma_tica_eigenvalues.png")
plt.close()


print("PyEMMA C0 shape:", tica_obj.cov.shape)
print("PyEMMA C0 (first 3x3):\n", tica_obj.cov[:3,:3])

print("PyEMMA Cτ (first 3x3):\n", tica_obj.cov_tau[:3,:3])

print("PyEMMA eigenvalues (all):", tica_obj.eigenvalues)
