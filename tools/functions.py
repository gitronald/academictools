"""Convenience module for short imports: import tools.functions as fn."""

from tools import dataframes, io, plots, stats, utils

_SUBPACKAGES = (dataframes, io, plots, stats, utils)

# FilePath flows in via io.__all__ (it lives in tools.io.data)
__all__ = [
    *(name for pkg in _SUBPACKAGES for name in pkg.__all__),
]


def __getattr__(name: str):
    """Lazily resolve a flattened name from one of the subpackages."""
    for pkg in _SUBPACKAGES:
        if name in pkg.__all__:
            return getattr(pkg, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return __all__
