import subprocess
import sys

import pytest

BANNED = ("pandas", "seaborn", "statsmodels")

MODULES = [
    "tools",
    "tools.functions",
    "tools.dataframes",
    "tools.io",
    "tools.plots",
    "tools.stats",
    "tools.utils",
]

PROBE = """
import importlib
import sys

module = importlib.import_module({name!r})
# resolve every flattened name too, so lazy lookups are covered
for attr in getattr(module, "__all__", []):
    getattr(module, attr)
print(",".join(m for m in {banned!r} if m in sys.modules))
"""


@pytest.mark.parametrize("name", MODULES)
def test_import_loads_no_pandas_seaborn_or_statsmodels(name):
    # a fresh interpreter, so modules imported by other tests do not leak in
    result = subprocess.run(
        [sys.executable, "-c", PROBE.format(name=name, banned=BANNED)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == ""


def test_version_matches_package_metadata():
    from importlib.metadata import version

    import tools

    assert tools.__version__ == version("academictools")
