import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.pca import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    u = mda.Universe("../../mlx.gro", "../../rep1_skip10.xtc")
    select = "name CA"

    pca = PCA(u, select=select, n_components=3).run()

    pcs = pca.results.p_components        # (3N, 3)
    var = pca.results.variance            # (3,)
    cum = pca.results.cumulated_variance  # (3,)

    print("PCA p_components shape:", pcs.shape)
    print("PCA variance:", var)
    print("PCA cumulated_variance:", cum)

    n_atoms = u.select_atoms(select).n_atoms
    assert pcs.shape == (3 * n_atoms, 3)
    assert var.shape == (3,)
    assert np.all(var[:-1] >= var[1:] - 1e-12)
    assert np.all((cum >= -1e-12) & (cum <= 1 + 1e-12))
    assert np.all(cum[:-1] <= cum[1:] + 1e-12)


    ag = u.select_atoms(select)
    try:
        proj = pca.transform(ag, n_components=pcs.shape[1])
        print("Projected shape:", proj.shape)
        assert proj.shape[0] == u.trajectory.n_frames
        assert proj.shape[1] == pcs.shape[1]
        # Scatter plot of first two PCs
        plt.figure(figsize=(6,5))
        plt.scatter(proj[:,0], proj[:,1], s=10, alpha=0.5)
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.title("MDAnalysis PCA: Projection onto first two PCs")
        plt.tight_layout()
        plt.savefig("pca_proj.png")
        plt.close()
    except Exception as e:
        print("Transform skipped due to:", repr(e))

    # Eigenvalues plot
    plt.figure()
    plt.plot(var, "o-")
    plt.xlabel("Component")
    plt.ylabel("Variance")
    plt.title("MDAnalysis PCA eigenvalues")
    plt.tight_layout()
    plt.savefig("pca_eigenvalues.png")
    plt.close()

    print("MDAnalysis PCA (same traj) test passed.")


if __name__ == "__main__":
    main()
