import time
import numpy as np

import MDAnalysis as mda
from tica.tica import TICA
from MDAnalysis.analysis.pca import PCA as MDAPCA


def time_it(fn, repeat=1, warmup=True, label="task"):
    if warmup:
        try:
            fn()
        except Exception:
            pass
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    print(f"{label}: {avg:.3f}s (avg over {repeat})")
    return avg


def bench_tica_mdanalysis():
    u = mda.Universe("../../mlx.gro", "../../rep1_skip10.xtc")
    def run():
        TICA(
            u,
            select="name CA",
            lag=20,
            n_components=3,
            regularization=1e-5,
        ).run(step=1)
    return time_it(run, repeat=1, label="TICA (ours, MDAnalysis)")


def bench_pca_mdanalysis():
    u = mda.Universe("../../mlx.gro", "../../rep1_skip10.xtc")
    def run():
        MDAPCA(u, select="name CA", n_components=3).run()
    return time_it(run, repeat=1, label="PCA (MDAnalysis)")


def bench_tica_pyemma():
    try:
        import mdtraj as md
        import pyemma
    except Exception as e:
        print("PyEMMA bench skipped:", repr(e))
        return None

    def run():
        import numpy as _np
        if not hasattr(_np, 'bool'):
            # Work around older PyEMMA expecting np.bool
            setattr(_np, 'bool', _np.bool_)
        traj = md.load_xtc("../../rep1_skip10.xtc", top="../../mlx.gro")
        traj.xyz *= 10.0  # nm -> Å for parity
        ca_indices = traj.topology.select("name CA")
        X = traj.atom_slice(ca_indices).xyz.reshape(traj.n_frames, -1)
        _ = pyemma.coordinates.tica(data=X, lag=20, dim=3)
    return time_it(run, repeat=1, label="tICA (PyEMMA)")


def main():
    print("Benchmarking on ../../mlx.gro + ../../rep1_skip10.xtc (name CA)")
    t1 = bench_tica_mdanalysis()
    t2 = bench_pca_mdanalysis()
    t3 = bench_tica_pyemma()
    print("\nSummary (seconds):")
    print(f"- TICA (ours): {t1:.3f}")
    print(f"- PCA (MDAnalysis): {t2:.3f}")
    if t3 is not None:
        print(f"- tICA (PyEMMA): {t3:.3f}")


if __name__ == "__main__":
    main()
