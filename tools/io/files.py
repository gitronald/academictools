"""File system utilities."""

import atexit
import os
import shutil
import sys


def generate_temp_dir(temp_dir: str) -> None:
    """Create a temporary directory and register automatic cleanup on exit.

    Removes any existing directory at the path before creating a new one.

    Args:
        temp_dir: Path for the temporary directory
    """
    remove_temp_dir(temp_dir)
    os.makedirs(temp_dir)
    atexit.register(remove_temp_dir, temp_dir)


def remove_temp_dir(temp_dir: str) -> None:
    """Remove a temporary directory if it exists.

    Args:
        temp_dir: Path to the directory to remove
    """
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def get_size_in_gb(obj: object) -> float:
    """Get the memory size of a Python object in gigabytes.

    Args:
        obj: Python object to measure

    Returns:
        Size in GB as a float
    """
    size_in_bytes = sys.getsizeof(obj)
    size_in_gb = size_in_bytes / (1024**3)
    return size_in_gb
