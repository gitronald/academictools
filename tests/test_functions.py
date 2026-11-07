import pytest

import tools.functions as fn
from tools import dataframes, io, plots, stats, utils


class TestFlattenedNamespace:
    def test_all_aggregates_every_subpackage(self):
        expected = {
            name
            for pkg in (dataframes, io, plots, stats, utils)
            for name in pkg.__all__
        }
        assert set(fn.__all__) == expected

    def test_dir_matches_all(self):
        # dir() sorts and dedupes, so compare as sets
        assert set(dir(fn)) == set(fn.__all__)

    @pytest.mark.parametrize(
        ("name", "pkg"),
        [
            ("crosstab", dataframes),
            ("load_xsv", io),
            ("get_modal_category", stats),
            ("print_line", utils),
        ],
    )
    def test_getattr_resolves_to_subpackage(self, name, pkg):
        assert getattr(fn, name) is getattr(pkg, name)

    def test_unknown_name_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="no attribute 'does_not_exist'"):
            fn.does_not_exist
