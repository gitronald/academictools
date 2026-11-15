from importlib.metadata import version

from . import dataframes, io, plots, stats, utils

__version__ = version("academictools")

__all__ = [
    "dataframes",
    "io",
    "plots",
    "stats",
    "utils",
]
