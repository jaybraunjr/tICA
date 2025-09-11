__all__ = [
    "tica",
    "plotting",
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
    __version__ = version("mdakit")  # type: ignore[arg-type]
except Exception:
    __version__ = "0.1.0"

