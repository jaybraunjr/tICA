__all__ = [
    "tica",
    "plotting",
    "featurizer",
]

try:
    from importlib.metadata import version, PackageNotFoundError
except Exception:  # pragma: no cover
    try:
        from importlib_metadata import version, PackageNotFoundError  # type: ignore
    except Exception:  # pragma: no cover
        version = None
        PackageNotFoundError = Exception

try:
    __version__ = version("tica")  # type: ignore[arg-type]
except Exception:
    __version__ = "0.1.0"

from .featurizer import (  # noqa: E402
    Featurizer,
    contact_count_feat,
    dihedral_feat,
    distance_feat,
    gyration_feat,
    hbond_count_feat,
    plane_distance_feat,
    rmsd_feat,
    sasa_feat,
    secondary_structure_frac_feat,
    torsion_avg_feat,
)

__all__ += [
    "Featurizer",
    "rmsd_feat",
    "distance_feat",
    "gyration_feat",
    "contact_count_feat",
    "hbond_count_feat",
    "dihedral_feat",
    "torsion_avg_feat",
    "secondary_structure_frac_feat",
    "sasa_feat",
    "plane_distance_feat",
]
