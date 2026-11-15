import tempfile

import matplotlib
import polars as pl
import pytest

# Use a headless backend so plot tests never try to open a GUI window.
matplotlib.use("Agg")


@pytest.fixture
def sample_dataframe():
    """Create a sample polars DataFrame for testing."""
    return pl.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "city": ["New York", "London", "Tokyo"],
        }
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
